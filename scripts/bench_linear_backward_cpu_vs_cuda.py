"""
scripts/bench_linear_backward_cpu_vs_cuda.py

CPU vs CUDA Linear backward microbenchmark (NOT a unit test) for KeyDNN.

What this measures
------------------
- Backward-only cost of Linear, excluding:
  - HtoD/DtoH transfers (done once per case)
  - forward time (forward is executed once to produce a Context)
- It times: ctx.backward_fn(grad_out) repeatedly.

Notes
-----
- Requires CUDA wrappers + DLL to be importable for CUDA path.
- Synchronizes on CUDA after each backward to ensure kernels complete.
- If your Linear CUDA backward still falls back to host reduction for db,
  then bias=True backward will include that overhead.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

# -------------------------
# Make repo_root/src importable
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from keydnn.domain.device._device import Device
from keydnn.infrastructure._linear import Linear
from keydnn.infrastructure.tensor._tensor import Tensor

# CUDA utils (try-import; match your other bench scripts style)
try:
    from keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as cuda_utils

    _CUDA_IMPORT_OK = True
except Exception:
    _CUDA_IMPORT_OK = False


def _unwrap_param(obj):
    """
    Try to unwrap Parameter-like objects to a Tensor.

    Supports:
    - Tensor directly
    - Parameter with `.data` pointing to a Tensor
    """
    if isinstance(obj, Tensor):
        return obj
    if hasattr(obj, "data") and isinstance(getattr(obj, "data"), Tensor):
        return getattr(obj, "data")
    return None


def _get_linear_tensors(layer: Linear) -> tuple[Tensor, Optional[Tensor]]:
    """
    Best-effort accessors for Linear weight/bias tensors.

    Tries common attribute names used across implementations.
    """
    # weight
    for wname in ("weight", "W", "w"):
        if hasattr(layer, wname):
            wt = _unwrap_param(getattr(layer, wname))
            if wt is not None:
                break
    else:
        raise AttributeError(
            "Could not locate Linear weight tensor on layer (weight/W/w)."
        )

    # bias (optional)
    bt = None
    for bname in ("bias", "b"):
        if hasattr(layer, bname):
            cand = _unwrap_param(getattr(layer, bname))
            if cand is not None:
                bt = cand
                break

    return wt, bt


def _copy_tensor_cpu_to_cuda(
    dst_cuda: Tensor, src_cpu: Tensor, *, cuda_ctx: _CudaContext
) -> None:
    """
    Copy a CPU Tensor's data into an existing CUDA Tensor.
    """
    arr = src_cpu.to_numpy()
    _ensure_cuda_alloc_and_h2d(dst_cuda, arr, ctx=cuda_ctx)


def _sync_linear_params_cpu_to_cuda(
    layer_cpu: Linear, layer_cuda: Linear, *, cuda_ctx: _CudaContext
) -> None:
    """
    Overwrite CUDA layer parameters to exactly match CPU layer parameters.
    """
    w_cpu, b_cpu = _get_linear_tensors(layer_cpu)
    w_cuda, b_cuda = _get_linear_tensors(layer_cuda)

    _copy_tensor_cpu_to_cuda(w_cuda, w_cpu, cuda_ctx=cuda_ctx)
    if b_cpu is not None and b_cuda is not None:
        _copy_tensor_cpu_to_cuda(b_cuda, b_cpu, cuda_ctx=cuda_ctx)

    cuda_ctx.sync()


def _time_one(fn: Callable[[], None], *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    ts: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return ts


def _median(xs: list[float]) -> float:
    return statistics.median(xs)


def _fmt_seconds(x: float) -> str:
    if x < 1e-6:
        return f"{x*1e9:.2f} ns"
    if x < 1e-3:
        return f"{x*1e6:.2f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def _speedup(cpu_s: float, cuda_s: float) -> float:
    return (cpu_s / cuda_s) if cuda_s > 0 else float("inf")


def _dtype_from_str(s: str) -> np.dtype:
    if s == "float32":
        return np.float32
    if s == "float64":
        return np.float64
    raise ValueError(f"unsupported dtype: {s}")


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    in_features: int
    out_features: int
    bias: bool


class _CudaContext:
    def __init__(self) -> None:
        if not _CUDA_IMPORT_OK:
            raise RuntimeError(
                "CUDA ctypes modules failed to import. "
                "Build / install CUDA native DLL and python wrappers first."
            )
        self.lib = cuda_utils.load_keydnn_cuda_native()

    def sync(self) -> None:
        cuda_utils.cuda_synchronize(self.lib)

    def h2d(self, dst_dev: int, src_host: np.ndarray) -> None:
        cuda_utils.cudaMemcpyHtoD(
            self.lib, int(dst_dev), src_host, int(src_host.nbytes)
        )


def _get_ctx(y: Tensor):
    """
    Retrieve legacy Context object from a Tensor.

    Adjust this helper if your Tensor stores ctx under a different attribute.
    """
    # Common patterns in your code: y._set_ctx(ctx) => likely y._ctx
    if hasattr(y, "_ctx"):
        return getattr(y, "_ctx")
    if hasattr(y, "ctx"):
        return getattr(y, "ctx")
    raise AttributeError("Could not find Context on Tensor (expected y._ctx or y.ctx).")


def _ensure_cuda_alloc_and_h2d(
    t: Tensor, arr: np.ndarray, *, ctx: _CudaContext
) -> None:
    """
    Ensure a CUDA tensor has allocated dev buffer, then HtoD copy from arr.
    """
    if not t.device.is_cuda():
        raise TypeError("expected CUDA tensor")
    if hasattr(t, "_ensure_cuda_alloc"):
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    dst = int(getattr(t, "data", 0))
    if dst == 0:
        raise RuntimeError("CUDA tensor has no device buffer (data == 0)")
    ctx.h2d(dst, arr)


def bench_backward_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    device_ordinal: int,
    rng_seed: int,
    cuda_ctx: Optional[_CudaContext],
) -> None:
    rng = np.random.default_rng(rng_seed)

    # Host init once
    x_np = rng.standard_normal((case.batch, case.in_features)).astype(dtype, copy=False)

    # -------------------------
    # CPU setup
    # -------------------------
    cpu_dev = Device("cpu")
    layer_cpu = Linear(
        case.in_features,
        case.out_features,
        bias=case.bias,
        device=cpu_dev,
    )
    x_cpu = Tensor(
        shape=x_np.shape, device=cpu_dev, requires_grad=True, dtype=np.dtype(dtype)
    )
    x_cpu.copy_from_numpy(x_np)

    # forward once (not timed)
    y_cpu = layer_cpu.forward(x_cpu)
    ctx_cpu = _get_ctx(y_cpu)

    # grad_out once (not timed)
    go_cpu = Tensor(
        shape=y_cpu.shape, device=cpu_dev, requires_grad=False, dtype=np.dtype(dtype)
    )
    go_cpu.fill(1.0)

    def cpu_bwd() -> None:
        grads = ctx_cpu.backward_fn(go_cpu)
        # Touch something to keep it "used"
        _ = grads[0] if isinstance(grads, tuple) else grads

    # -------------------------
    # CUDA setup
    # -------------------------
    if cuda_ctx is None:
        raise RuntimeError("cuda_ctx is required for CUDA benchmark")

    cuda_dev = Device("cuda:0")
    layer_cuda = Linear(
        case.in_features,
        case.out_features,
        bias=case.bias,
        device=cuda_dev,
    )

    # IMPORTANT: make CUDA params match CPU params (for sanity correctness)
    _sync_linear_params_cpu_to_cuda(layer_cpu, layer_cuda, cuda_ctx=cuda_ctx)

    # x on CUDA
    x_cuda = Tensor(
        shape=x_np.shape,
        device=cuda_dev,
        requires_grad=True,
        dtype=np.dtype(dtype),
    )
    _ensure_cuda_alloc_and_h2d(x_cuda, x_np, ctx=cuda_ctx)

    # forward once (not timed)
    y_cuda = layer_cuda.forward(x_cuda)
    ctx_cuda = _get_ctx(y_cuda)

    # grad_out on CUDA (not timed)
    go_cuda = Tensor(
        shape=y_cuda.shape, device=cuda_dev, requires_grad=False, dtype=np.dtype(dtype)
    )
    # Use your device-aware fill (you have fill kernel)
    go_cuda.fill(1.0)
    cuda_ctx.sync()

    def cuda_bwd() -> None:
        grads = ctx_cuda.backward_fn(go_cuda)
        cuda_ctx.sync()
        _ = grads[0] if isinstance(grads, tuple) else grads

    # -------------------------
    # Optional sanity check (small)
    # -------------------------
    if sanity:
        g_cpu = ctx_cpu.backward_fn(go_cpu)
        g_cuda = ctx_cuda.backward_fn(go_cuda)
        cuda_ctx.sync()

        # Compare grad_x + grad_w (+ grad_b if present)
        # (We pull CUDA grads back to numpy via to_numpy())
        for i, (gc, gg) in enumerate(zip(g_cpu, g_cuda)):
            if gc is None or gg is None:
                continue
            np.testing.assert_allclose(
                gg.to_numpy(),
                gc.to_numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"grad mismatch at index {i}",
            )

    # -------------------------
    # Timing
    # -------------------------
    t_cpu = _time_one(cpu_bwd, warmup=warmup, repeats=repeats)
    t_cuda = _time_one(cuda_bwd, warmup=warmup, repeats=repeats)

    cpu_med = _median(t_cpu)
    cuda_med = _median(t_cuda)

    print(
        f"{case.name}: batch={case.batch} in={case.in_features} out={case.out_features} "
        f"bias={case.bias} dtype={dtype} | "
        f"cpu_bwd={_fmt_seconds(cpu_med):>10}  cuda_bwd={_fmt_seconds(cuda_med):>10}  "
        f"speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--in_features", type=int, default=1024)
    ap.add_argument("--out_features", type=int, default=1024)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--bias", action="store_true", help="Enable bias (default: False)")
    ap.add_argument(
        "--no-bias", action="store_true", help="Disable bias (overrides --bias)"
    )
    ap.add_argument(
        "--sanity", action="store_true", help="Compare CPU vs CUDA grads (not timed)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--presets", action="store_true", help="Run a preset suite.")
    args = ap.parse_args()

    dtype = _dtype_from_str(args.dtype)

    if args.no_bias:
        bias = False
    else:
        bias = (
            bool(args.bias)
            if ("--bias" in sys.argv or "--no-bias" in sys.argv)
            else True
        )

    if not _CUDA_IMPORT_OK:
        raise SystemExit(
            "CUDA wrappers not importable. Build / install CUDA native DLL and python wrappers first."
        )

    cuda_ctx = _CudaContext()

    print("\n" + "=" * 110)
    print(
        f"Linear CPU vs CUDA BACKWARD benchmark | dtype={dtype} "
        f"(warmup={args.warmup}, repeats={args.repeats}, device={args.device}, sanity={args.sanity})"
    )
    print("=" * 110)

    if args.presets:
        cases = [
            Case("small", 256, 512, 512, bias),
            Case("mid", 1024, 1024, 1024, bias),
            Case("big", 1024, 2048, 2048, bias),
            Case("huge", 1024, 4096, 4096, bias),
        ]
        for c in cases:
            bench_backward_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                sanity=args.sanity,
                device_ordinal=args.device,
                rng_seed=args.seed,
                cuda_ctx=cuda_ctx,
            )
    else:
        c = Case("single", args.batch, args.in_features, args.out_features, bias)
        bench_backward_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            sanity=args.sanity,
            device_ordinal=args.device,
            rng_seed=args.seed,
            cuda_ctx=cuda_ctx,
        )


if __name__ == "__main__":
    main()
