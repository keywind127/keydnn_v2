"""
scripts/bench_linear_cpu_vs_cuda.py

CPU vs CUDA Linear microbenchmark (NOT a unit test) for KeyDNN.

Benchmarks forward-only computation for:
- Linear.forward(x) = x @ W^T (+ b)

Timing policy
-------------
- Excludes HtoD/DtoH transfers (performed once per case).
- Times only CPU forward or CUDA forward.
- Uses warmup iterations before timed repeats.
- Measures *forward only* (no backward).

Notes
-----
- CUDA path depends on your CUDA-enabled Tensor ops used by Linear:
  matmul / transpose / stack / add.
- Bias expansion uses Tensor.stack; that cost is part of the measured forward.
- Uses device-aware parameter initialization in Linear; for CUDA it performs
  HtoD copies during module initialization (outside timed region).
- If CUDA wrappers / DLL are not available, the script exits with a clear message.

Usage
-----
python scripts/bench_linear_cpu_vs_cuda.py --presets --dtype float32
python scripts/bench_linear_cpu_vs_cuda.py --batch 1024 --in_features 4096 --out_features 4096 --dtype float32
python scripts/bench_linear_cpu_vs_cuda.py --presets --sanity
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

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
from keydnn.infrastructure.tensor._tensor import Tensor
from keydnn.infrastructure.fully_connected._linear import Linear  # adjust if your path differs


# -------------------------
# Optional CUDA imports
# -------------------------
try:
    from keydnn.infrastructure.ops.pool2d_cuda import (
        _load_cuda_lib,
        cuda_set_device,
    )

    _CUDA_IMPORT_OK = True
except Exception:
    _CUDA_IMPORT_OK = False


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


def _dtype_str(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "float32"
    if dtype == np.float64:
        return "float64"
    return str(dtype)


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


def _get_cuda_device(index: int = 0) -> Device:
    """Best-effort helper to obtain a CUDA Device instance across possible Device APIs."""
    if hasattr(Device, "cuda") and callable(getattr(Device, "cuda")):
        return Device.cuda(index)  # type: ignore[attr-defined]

    try:
        return Device("cuda", index)  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(f"cuda:{index}")  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(kind="cuda", index=index)  # type: ignore[call-arg]
    except Exception as e:
        raise RuntimeError(
            "Unable to construct a CUDA Device; update _get_cuda_device()."
        ) from e


def _make_tensor_from_numpy_cpu(x: np.ndarray) -> Tensor:
    dev = Device("cpu")
    # Prefer public API if it exists
    if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
        return Tensor.from_numpy(x, device=dev)  # type: ignore[call-arg]
    t = Tensor(shape=x.shape, device=dev, requires_grad=False)
    t.copy_from_numpy(x)
    return t


def _make_tensor_from_numpy_cuda(x: np.ndarray, *, cuda_device: Device) -> Tensor:
    """
    Create a CUDA Tensor and copy host -> device using Tensor APIs already present in your repo.
    We rely on Tensor.from_numpy if available, otherwise fall back to:
      - allocate CUDA tensor
      - _ensure_cuda_alloc
      - cudaMemcpyHtoD via known wrapper path
    """
    x_c = np.ascontiguousarray(x)

    # Preferred: Tensor.from_numpy(..., device=cuda)
    if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
        return Tensor.from_numpy(x_c, device=cuda_device)  # type: ignore[call-arg]

    # Fallback: manual allocate + HtoD
    t = Tensor(shape=x_c.shape, device=cuda_device, requires_grad=False, dtype=np.dtype(x_c.dtype))  # type: ignore[call-arg]
    if hasattr(t, "_ensure_cuda_alloc") and callable(getattr(t, "_ensure_cuda_alloc")):
        t._ensure_cuda_alloc(dtype=np.dtype(x_c.dtype))
    dst = int(getattr(t, "data", 0))
    if dst == 0:
        raise RuntimeError("CUDA tensor allocation failed: t.data == 0")

    # Robust import for cudaMemcpyHtoD
    def _import_cudaMemcpyHtoD():
        try:
            from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        try:
            from keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        try:
            from keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        raise ImportError(
            "Could not import cudaMemcpyHtoD from known native_cuda ctypes modules."
        )

    cudaMemcpyHtoD = _import_cudaMemcpyHtoD()

    if hasattr(t, "_get_cuda_lib") and callable(getattr(t, "_get_cuda_lib")):
        lib = t._get_cuda_lib()
    else:
        lib = Tensor._get_cuda_lib()

    cudaMemcpyHtoD(lib, dst, x_c, int(x_c.nbytes))
    return t


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    in_features: int
    out_features: int
    bias: bool


def bench_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    seed: int,
    device_index: int,
) -> None:
    rng = np.random.default_rng(seed)

    x_np = rng.standard_normal((case.batch, case.in_features)).astype(dtype, copy=False)

    # -------------------------
    # CPU setup (outside timing)
    # -------------------------
    cpu_dev = Device("cpu")
    x_cpu = _make_tensor_from_numpy_cpu(x_np)

    lin_cpu = Linear(
        in_features=case.in_features,
        out_features=case.out_features,
        bias=case.bias,
        device=cpu_dev,
    )

    # -------------------------
    # CUDA setup (outside timing)
    # -------------------------
    if not _CUDA_IMPORT_OK:
        raise RuntimeError(
            "CUDA wrappers not importable. Build/install CUDA native DLL and python wrappers first."
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device_index))
    cuda_dev = _get_cuda_device(device_index)

    x_cuda = _make_tensor_from_numpy_cuda(x_np, cuda_device=cuda_dev)

    lin_cuda = Linear(
        in_features=case.in_features,
        out_features=case.out_features,
        bias=case.bias,
        device=cuda_dev,
    )

    # -------------------------
    # Sanity check (not timed)
    # -------------------------
    if sanity:
        y_cpu = lin_cpu.forward(x_cpu).to_numpy()
        y_cuda_t = lin_cuda.forward(x_cuda)
        y_cuda = y_cuda_t.to_numpy()  # relies on your DtoH path for CUDA tensors
        # float64 tolerance tighter; float32 more relaxed
        if dtype == np.float64:
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-10, atol=1e-10)
        else:
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-4, atol=1e-4)

    # -------------------------
    # Timed regions (forward only)
    # -------------------------
    def cpu_fwd() -> None:
        _ = lin_cpu.forward(x_cpu)

    def cuda_fwd() -> None:
        _ = lin_cuda.forward(x_cuda)

    t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
    t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

    cpu_med = _median(t_cpu)
    cuda_med = _median(t_cuda)

    print(
        f"{case.name}: "
        f"batch={case.batch} in={case.in_features} out={case.out_features} bias={case.bias} "
        f"dtype={_dtype_str(dtype)} | "
        f"cpu={_fmt_seconds(cpu_med):>10}  cuda={_fmt_seconds(cuda_med):>10}  "
        f"speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--in_features", type=int, default=1024)
    ap.add_argument("--out_features", type=int, default=1024)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--no-bias", dest="bias", action="store_false")
    ap.set_defaults(bias=True)

    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--presets", action="store_true")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.presets:
        if args.big:
            cases = [
                Case("big-1k", 2048, 1024, 1024, True),
                Case("big-2k", 2048, 2048, 2048, True),
                Case("big-4k", 1024, 4096, 4096, True),
            ]
        else:
            cases = [
                Case("small-256", 2048, 256, 256, True),
                Case("mid-1k", 1024, 1024, 1024, True),
                Case("wide", 512, 4096, 1024, True),
                Case("tall", 4096, 256, 2048, True),
                Case("no-bias", 1024, 1024, 1024, False),
            ]

        print("\n" + "=" * 110)
        print(
            f"Linear CPU vs CUDA benchmark | dtype={_dtype_str(dtype)} "
            f"(warmup={args.warmup}, repeats={args.repeats}, device={args.device}, sanity={args.sanity})"
        )
        print("=" * 110)

        for c in cases:
            bench_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                sanity=args.sanity,
                seed=args.seed,
                device_index=args.device,
            )
    else:
        c = Case("single", args.batch, args.in_features, args.out_features, args.bias)
        print("\n" + "=" * 110)
        print(
            f"Linear CPU vs CUDA benchmark | dtype={_dtype_str(dtype)} "
            f"(warmup={args.warmup}, repeats={args.repeats}, device={args.device}, sanity={args.sanity})"
        )
        print("=" * 110)

        bench_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            sanity=args.sanity,
            seed=args.seed,
            device_index=args.device,
        )


if __name__ == "__main__":
    main()
