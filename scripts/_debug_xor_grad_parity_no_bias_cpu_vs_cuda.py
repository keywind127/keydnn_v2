"""
scripts/debug_xor_grad_parity_no_bias_cpu_vs_cuda.py

Sanity: XOR grad parity CPU vs CUDA with *bias disabled*.

Goal
----
If CUDA gradients become much closer to CPU when bias=False, the bug is likely
in bias backward (db reduction) or bias add forward/backward.

This script:
- Builds an identical XOR MLP on CPU and CUDA (same weights copied over)
- Runs one forward + loss + backward on both
- Reports loss diff and param grad max|diff|, plus MAX across params

Usage
-----
python scripts/debug_xor_grad_parity_no_bias_cpu_vs_cuda.py
python scripts/debug_xor_grad_parity_no_bias_cpu_vs_cuda.py --loss mean
python scripts/debug_xor_grad_parity_no_bias_cpu_vs_cuda.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np


# -------------------------
# Make repo_root/src importable
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _cuda_available() -> bool:
    try:
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _cuda_sync_best_effort(device_str: str) -> None:
    if not device_str.startswith("cuda"):
        return

    # Try wrapper sync functions
    for mod_path in (
        "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
        "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
        "keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
    ):
        try:
            m = __import__(mod_path, fromlist=["cuda_synchronize"])
            if hasattr(m, "cuda_synchronize"):
                # Most wrappers take lib, but some accept anything; best effort.
                try:
                    m.cuda_synchronize(m.load_keydnn_cuda_native())  # type: ignore[attr-defined]
                except Exception:
                    # fallback: some wrappers expose a different name / signature
                    pass
                return
        except Exception:
            continue

    # If no sync found, do nothing.


def _copy_params_cpu_to_cuda(cpu_model, cuda_model) -> None:
    """
    Copy CPU parameters -> CUDA parameters via numpy round-trip.

    Assumes model.parameters() returns tensors that support:
      - to_numpy()
      - copy_from_numpy()
    """
    cpu_ps = list(cpu_model.parameters())
    cuda_ps = list(cuda_model.parameters())
    assert len(cpu_ps) == len(cuda_ps), "CPU/CUDA parameter count mismatch"

    for i, (p_cpu, p_cuda) in enumerate(zip(cpu_ps, cuda_ps)):
        w = p_cpu.to_numpy()
        # ensure contiguous float32
        w = np.ascontiguousarray(w, dtype=np.float32)
        p_cuda.copy_from_numpy(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--loss",
        choices=["sum", "mean"],
        default="sum",
        help="use sum-loss (recommended for diagnosing mean/reduction bugs) or mean-loss",
    )
    args = ap.parse_args()

    cuda_str = args.device.strip().lower()
    if cuda_str.startswith("cuda") and not _cuda_available():
        raise SystemExit("CUDA requested but CUDA native DLL/wrappers not available.")

    from keydnn.infrastructure._models import Sequential
    from keydnn.infrastructure.fully_connected._linear import Linear
    from keydnn.infrastructure._activations import Sigmoid
    from keydnn.infrastructure.tensor._tensor import Tensor
    from keydnn.domain.device._device import Device

    np.random.seed(args.seed)

    # ---------------- Dataset ----------------
    x_np = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    # ---------------- Build CPU baseline model (bias=False) ----------------
    cpu_dev = Device("cpu")
    x_cpu = Tensor(shape=x_np.shape, device=cpu_dev)
    x_cpu.copy_from_numpy(x_np)
    y_cpu = Tensor(shape=y_np.shape, device=cpu_dev)
    y_cpu.copy_from_numpy(y_np)

    hidden = int(args.hidden)
    model_cpu = Sequential(
        Linear(2, hidden, bias=False, device=cpu_dev),
        Sigmoid(),
        Linear(hidden, 1, bias=False, device=cpu_dev),
        Sigmoid(),
    )

    # ---------------- Build CUDA model (bias=False) ----------------
    cuda_dev = Device(cuda_str) if cuda_str.startswith("cuda") else Device("cpu")
    x_cuda = Tensor(shape=x_np.shape, device=cuda_dev)
    x_cuda.copy_from_numpy(x_np)
    y_cuda = Tensor(shape=y_np.shape, device=cuda_dev)
    y_cuda.copy_from_numpy(y_np)

    model_cuda = Sequential(
        Linear(2, hidden, bias=False, device=cuda_dev),
        Sigmoid(),
        Linear(hidden, 1, bias=False, device=cuda_dev),
        Sigmoid(),
    )

    # Copy CPU params -> CUDA params for identical starting point
    if cuda_str.startswith("cuda"):
        _copy_params_cpu_to_cuda(model_cpu, model_cuda)

    # ---------------- Loss ----------------
    def loss_fn(pred, target):
        diff = pred - target
        sq = diff * diff
        if args.loss == "mean":
            if hasattr(sq, "mean"):
                return sq.mean()
            # fallback: mean = sum / numel
            return sq.sum() * (1.0 / float(np.prod(target.shape)))
        else:
            # sum loss (no scaling) is best for diagnosing reduction/bias bugs
            return sq.sum()

    # ---------------- CPU forward/backward ----------------
    pred_cpu = model_cpu(x_cpu)
    loss_cpu = loss_fn(pred_cpu, y_cpu)
    loss_cpu.backward()

    # ---------------- CUDA forward/backward ----------------
    pred_cuda = model_cuda(x_cuda)
    loss_cuda = loss_fn(pred_cuda, y_cuda)
    loss_cuda.backward()

    if cuda_str.startswith("cuda"):
        _cuda_sync_best_effort(cuda_str)

    # ---------------- Report ----------------
    lc = float(loss_cpu.to_numpy())
    lg = float(loss_cuda.to_numpy()) if hasattr(loss_cuda, "to_numpy") else float("nan")
    print(f"loss cpu={lc:.8f}  cuda={lg:.8f}  diff={abs(lc-lg):.3e}")

    cpu_ps = list(model_cpu.parameters())
    cuda_ps = list(model_cuda.parameters())
    max_all = 0.0

    for i, (p_cpu, p_cuda) in enumerate(zip(cpu_ps, cuda_ps)):
        g_cpu = (
            p_cpu.grad.to_numpy() if getattr(p_cpu, "grad", None) is not None else None
        )
        g_cuda = (
            p_cuda.grad.to_numpy()
            if getattr(p_cuda, "grad", None) is not None
            else None
        )

        if g_cpu is None or g_cuda is None:
            print(
                f"param[{i}] grad missing (cpu={g_cpu is None}, cuda={g_cuda is None})"
            )
            continue

        g_cpu = np.asarray(g_cpu, dtype=np.float32)
        g_cuda = np.asarray(g_cuda, dtype=np.float32)
        md = float(np.max(np.abs(g_cpu - g_cuda)))
        max_all = max(max_all, md)
        print(f"param[{i}] grad max|diff| = {md:.6e}  shape={g_cpu.shape}")

    print(f"MAX grad max|diff| across params = {max_all:.6e}")


if __name__ == "__main__":
    main()
