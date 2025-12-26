"""
scripts/bench_matmul_cpu_vs_cuda.py

CPU vs CUDA matmul microbenchmark (NOT a unit test) for KeyDNN.

Benchmarks forward-only matrix multiplication for:
- CPU: NumPy-backed Tensor.matmul (via `@`)
- CUDA: Tensor.matmul CUDA path (via `@`)

Timing policy
-------------
- Excludes HtoD/DtoH transfers (performed once per case).
- Times only the matmul forward (CPU or CUDA).
- Preallocates output buffers and device buffers outside timed region.
- Uses CUDA synchronize to avoid async timing artifacts.

Notes
-----
This benchmark is aligned with the current Tensor.matmul CUDA contract:

- Both tensors must be on the same CUDA device.
- CUDA inputs MUST already have allocated device buffers (data != 0).
- This script therefore allocates inputs explicitly and copies host data to device
  using the native CUDA ctypes helpers (cudaMalloc + HtoD).

Usage
-----
python scripts/bench_matmul_cpu_vs_cuda.py --presets --dtype float32
python scripts/bench_matmul_cpu_vs_cuda.py --M 1024 --K 1024 --N 1024 --dtype float32
python scripts/bench_matmul_cpu_vs_cuda.py --M 4096 --K 4096 --N 4096 --dtype float32 --warmup 5 --repeats 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# -------------------------
# Make repo_root/src importable
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -------------------------
# Tensor / Device imports
# -------------------------
from keydnn.infrastructure.tensor._tensor import Tensor
from keydnn.domain.device._device import Device

# -------------------------
# CUDA ctypes imports
# -------------------------
try:
    # Your project already uses this pattern in bench_pool2d_cpu_vs_cuda.py
    from keydnn.infrastructure.native_cuda.python import avgpool2d_ctypes as cuda_utils

    _CUDA_IMPORT_OK = True
except Exception:
    _CUDA_IMPORT_OK = False


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


def _speedup(a: float, b: float) -> float:
    return (a / b) if b > 0 else float("inf")


def _dtype_str(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "float32"
    if dtype == np.float64:
        return "float64"
    return str(dtype)


@dataclass(frozen=True)
class Case:
    name: str
    M: int
    K: int
    N: int


class _CudaContext:
    """
    Minimal CUDA context wrapper for malloc/free and memcpy.

    Reuses the same ctypes helper module used by other benchmarks
    (avgpool2d_ctypes exposes load_keydnn_cuda_native, cuda_malloc/free,
    and memcpy helpers in this repo).
    """

    def __init__(self) -> None:
        if not _CUDA_IMPORT_OK:
            raise RuntimeError(
                "CUDA ctypes modules failed to import. "
                "Ensure keydnn CUDA DLL is built and python wrappers are importable."
            )
        self.lib = cuda_utils.load_keydnn_cuda_native()

    def malloc(self, nbytes: int) -> int:
        return int(cuda_utils.cuda_malloc(self.lib, int(nbytes)))

    def free(self, dev_ptr: int) -> None:
        if dev_ptr != 0:
            cuda_utils.cuda_free(self.lib, int(dev_ptr))

    def h2d(self, dst_dev: int, src_host: np.ndarray) -> None:
        cuda_utils.cudaMemcpyHtoD(
            self.lib, int(dst_dev), src_host, int(src_host.nbytes)
        )

    def d2h(self, dst_host: np.ndarray, src_dev: int) -> None:
        cuda_utils.cudaMemcpyDtoH(
            self.lib, dst_host, int(src_dev), int(dst_host.nbytes)
        )

    def sync(self) -> None:
        if hasattr(self.lib, "keydnn_cuda_synchronize"):
            fn = self.lib.keydnn_cuda_synchronize
            fn.argtypes = []
            fn.restype = np.int32  # type: ignore[assignment]
            _ = int(fn())


def _bench_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    rng_seed: int,
    device_index: int,
    ctx: _CudaContext,
) -> None:
    rng = np.random.default_rng(rng_seed)
    M, K, N = int(case.M), int(case.K), int(case.N)

    # Host inputs
    A_np = rng.standard_normal((M, K)).astype(dtype, copy=False)
    B_np = rng.standard_normal((K, N)).astype(dtype, copy=False)

    # -------------------------
    # CPU tensors
    # -------------------------
    A_cpu = Tensor((M, K), Device("cpu"), requires_grad=False, dtype=dtype)
    B_cpu = Tensor((K, N), Device("cpu"), requires_grad=False, dtype=dtype)
    A_cpu.copy_from_numpy(A_np)
    B_cpu.copy_from_numpy(B_np)

    # -------------------------
    # CUDA tensors (must have allocated devptrs)
    # -------------------------
    A_cuda = Tensor(
        (M, K), Device(f"cuda:{device_index}"), requires_grad=False, dtype=dtype
    )
    B_cuda = Tensor(
        (K, N), Device(f"cuda:{device_index}"), requires_grad=False, dtype=dtype
    )

    # Allocate device buffers explicitly to satisfy contract (data != 0)
    A_cuda._ensure_cuda_alloc(dtype=dtype)
    B_cuda._ensure_cuda_alloc(dtype=dtype)

    a_dev = int(A_cuda.data)
    b_dev = int(B_cuda.data)
    if a_dev == 0 or b_dev == 0:
        raise RuntimeError("Failed to allocate CUDA input buffers for benchmark.")

    # Copy host -> device once (excluded from timed region)
    ctx.h2d(a_dev, A_np)
    ctx.h2d(b_dev, B_np)
    ctx.sync()

    # Preallocate outputs outside timed region
    # CPU output: allocate a Tensor and overwrite its data each run
    Y_cpu = Tensor((M, N), Device("cpu"), requires_grad=False, dtype=dtype)

    # CUDA output: allocate a Tensor and reuse it each run.
    # We call ops-layer matmul directly to keep output allocation out of timing.
    from keydnn.infrastructure.ops.matmul_cuda import matmul2d_cuda

    Y_cuda = Tensor(
        (M, N), Device(f"cuda:{device_index}"), requires_grad=False, dtype=dtype
    )
    Y_cuda._ensure_cuda_alloc(dtype=dtype)
    y_dev = int(Y_cuda.data)
    if y_dev == 0:
        raise RuntimeError("Failed to allocate CUDA output buffer for benchmark.")

    # Also allocate host output for optional sanity check (excluded from timing)
    Y_cuda_host = np.empty((M, N), dtype=dtype)

    # -------------------------
    # Timed kernels
    # -------------------------
    def cpu_fwd() -> None:
        # Use Tensor.matmul API for CPU (keeps benchmark aligned with framework behavior)
        y = A_cpu @ B_cpu
        # Copy into preallocated buffer to avoid allocating new arrays in analysis code
        Y_cpu.copy_from_numpy(y.to_numpy())

    def cuda_fwd() -> None:
        # Use ops-layer matmul to avoid timing Tensor output allocation.
        matmul2d_cuda(
            ctx.lib,
            a_dev=int(a_dev),
            b_dev=int(b_dev),
            c_dev=int(y_dev),
            n=int(M),  # rows of A
            k=int(K),  # inner dim
            m=int(N),  # cols of B
            dtype=dtype,
            sync=True,
            device_index=int(device_index),
        )

    # -------------------------
    # Sanity check (not timed)
    # -------------------------
    if sanity:
        cpu_ref = (A_np @ B_np).astype(dtype, copy=False)
        cuda_fwd()
        ctx.d2h(Y_cuda_host, y_dev)
        np.testing.assert_allclose(Y_cuda_host, cpu_ref, rtol=1e-4, atol=1e-4)

    # -------------------------
    # Timing
    # -------------------------
    t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
    t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

    cpu_med = _median(t_cpu)
    cuda_med = _median(t_cuda)

    # Approx FLOPs for GEMM: 2*M*K*N
    flops = 2.0 * M * K * N
    cpu_gflops = (flops / cpu_med) / 1e9 if cpu_med > 0 else float("inf")
    cuda_gflops = (flops / cuda_med) / 1e9 if cuda_med > 0 else float("inf")

    print(
        f"matmul (M={M} K={K} N={N})  "
        f"cpu={_fmt_seconds(cpu_med):>10} ({cpu_gflops:8.2f} GFLOP/s)  "
        f"cuda={_fmt_seconds(cuda_med):>10} ({cuda_gflops:8.2f} GFLOP/s)  "
        f"speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--presets", action="store_true", help="Run a preset suite.")
    ap.add_argument("--big", action="store_true", help="Use larger preset shapes.")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Sanity-check CUDA output vs NumPy (not timed).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index.")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if not _CUDA_IMPORT_OK:
        raise SystemExit(
            "CUDA wrappers not importable. Build / install CUDA native DLL and python wrappers first."
        )

    ctx = _CudaContext()

    print("\n" + "=" * 98)
    print(
        f"KeyDNN matmul CPU vs CUDA benchmark  dtype={_dtype_str(dtype)}  "
        f"(warmup={args.warmup}, repeats={args.repeats}, device=cuda:{args.device})"
    )
    print("=" * 98)

    if args.presets:
        if args.big:
            cases = [
                Case("big-1024", 1024, 1024, 1024),
                Case("big-2048", 2048, 2048, 2048),
                Case("big-4096", 4096, 4096, 4096),
            ]
        else:
            cases = [
                Case("small-256", 256, 256, 256),
                Case("mid-512", 512, 512, 512),
                Case("mid-1024", 1024, 1024, 1024),
                Case("rect-1024x2048x512", 1024, 2048, 512),
            ]
        for c in cases:
            _bench_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                sanity=args.sanity,
                rng_seed=args.seed,
                device_index=args.device,
                ctx=ctx,
            )
    else:
        c = Case("single", args.M, args.K, args.N)
        _bench_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            sanity=args.sanity,
            rng_seed=args.seed,
            device_index=args.device,
            ctx=ctx,
        )


if __name__ == "__main__":
    main()
