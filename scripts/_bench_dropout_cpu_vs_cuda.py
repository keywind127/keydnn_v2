"""
scripts/bench_dropout_cpu_vs_cuda.py

CPU vs CUDA Dropout microbenchmark (NOT a unit test) for KeyDNN.

Benchmarks forward-only for:
- Dropout(p)

Timing policy
-------------
- Excludes HtoD/DtoH transfers (performed once per case).
- Times only CPU forward or CUDA forward.
- For CUDA, optionally synchronizes after each forward so timings are accurate.
- Uses the Dropout layer implementation (so it exercises broadcast_to + mask ops).

Notes
-----
- Dropout is stochastic; for sanity checks we seed NumPy before each forward
  so CPU and CUDA should produce identical masks if your RNG is shared.
  If CUDA uses a different RNG source, sanity may fail; you can disable it.

Usage
-----
python scripts/bench_dropout_cpu_vs_cuda.py --presets --dtype float32 --p 0.5
python scripts/bench_dropout_cpu_vs_cuda.py --N 64 --C 256 --H 56 --W 56 --p 0.2 --dtype float32
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

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
# KeyDNN imports
# -------------------------
from keydnn.domain.device._device import Device
from keydnn.infrastructure.layers._dropout import Dropout
from keydnn.infrastructure.tensor._tensor import Tensor


# CUDA availability (same pattern as your tests)
def _cuda_available() -> bool:
    try:
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


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


def _get_ctx(t):
    return getattr(t, "_ctx", None) or getattr(t, "ctx", None)


# -------------------------
# CUDA memcpy helpers (match your clone test style)
# -------------------------
def _bind_memcpy(lib, mc):
    # pick whichever names exist in your tree
    def htod(dst_dev: int, src_host: np.ndarray, nbytes: int, sync: bool) -> None:
        if hasattr(mc, "memcpy_htod"):
            mc.memcpy_htod(
                lib, dst_dev=dst_dev, src_host=src_host, nbytes=nbytes, sync=sync
            )
            return
        if hasattr(mc, "cuda_memcpy_h2d"):
            mc.cuda_memcpy_h2d(
                lib, dst_dev=dst_dev, src_host=src_host, nbytes=nbytes, sync=sync
            )
            return
        raise RuntimeError("memcpy_ctypes missing memcpy_htod/cuda_memcpy_h2d")

    def dtoh(dst_host: np.ndarray, src_dev: int, nbytes: int, sync: bool) -> None:
        if hasattr(mc, "memcpy_dtoh"):
            mc.memcpy_dtoh(
                lib, dst_host=dst_host, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        if hasattr(mc, "cuda_memcpy_d2h"):
            mc.cuda_memcpy_d2h(
                lib, dst_host=dst_host, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        raise RuntimeError("memcpy_ctypes missing memcpy_dtoh/cuda_memcpy_d2h")

    return htod, dtoh


def _cuda_tensor_from_numpy(
    arr: np.ndarray, *, dev: Device, TensorCls, lib, htod
) -> Tensor:
    arr = np.ascontiguousarray(arr)
    t = TensorCls(shape=arr.shape, device=dev, requires_grad=False, ctx=None)
    t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    if int(t.data) == 0 and arr.size != 0:
        raise RuntimeError("CUDA tensor has no allocated devptr (data == 0)")
    htod(dst_dev=int(t.data), src_host=arr, nbytes=int(arr.nbytes), sync=True)
    return t


@dataclass(frozen=True)
class Case:
    name: str
    N: int
    C: int
    H: int
    W: int


def bench_case(
    case: Case,
    *,
    p: float,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    seed: int,
    sync_cuda: bool,
) -> None:
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((case.N, case.C, case.H, case.W)).astype(
        dtype, copy=False
    )

    dev_cpu = Device("cpu")
    x_cpu = Tensor(shape=x_np.shape, device=dev_cpu, requires_grad=False, ctx=None)
    x_cpu.copy_from_numpy(
        x_np.astype(np.float32 if dtype == np.float32 else np.float64, copy=False)
    )

    d_cpu = Dropout(p=p)
    d_cpu.training = True

    # CPU bench function
    def cpu_fwd() -> None:
        np.random.seed(seed)  # deterministic mask per call
        y = d_cpu(x_cpu)
        # keep a use to avoid being optimized away (not that Python does, but still)
        _ = y.shape

    # CUDA setup (optional)
    if not _cuda_available():
        print("[skip] CUDA not available")
        return

    dev_cuda = Device("cuda:0")

    # initialize CUDA lib + memcpy
    lib = Tensor._get_cuda_lib()
    from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import cuda_set_device  # type: ignore

    cuda_set_device(lib, 0)

    from keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc  # type: ignore

    htod, dtoh = _bind_memcpy(lib, mc)

    x_cuda = _cuda_tensor_from_numpy(
        x_np, dev=dev_cuda, TensorCls=Tensor, lib=lib, htod=htod
    )

    d_cuda = Dropout(p=p)
    d_cuda.training = True

    # CUDA bench function
    def cuda_fwd() -> None:
        np.random.seed(seed)  # try to match CPU mask generation
        y = d_cuda(x_cuda)
        if sync_cuda:
            # if your Tensor exposes a sync, use it; else call cuda_synchronize via a known wrapper
            try:
                from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import cuda_synchronize  # type: ignore

                cuda_synchronize(lib)
            except Exception:
                pass
        _ = y.shape

    print("\n" + "=" * 98)
    print(
        f"{case.name}: N={case.N} C={case.C} H={case.H} W={case.W} "
        f"p={p} dtype={_dtype_str(dtype)} (warmup={warmup}, repeats={repeats}, sync_cuda={sync_cuda})"
    )
    print("-" * 98)

    # Sanity: compare outputs (best-effort)
    if sanity:
        np.random.seed(seed)
        y_cpu = d_cpu(x_cpu).to_numpy()

        np.random.seed(seed)
        y_cuda_t = d_cuda(x_cuda)

        y_cuda = np.empty(y_cpu.shape, dtype=np.dtype(y_cuda_t.dtype))
        dtoh(
            dst_host=y_cuda,
            src_dev=int(y_cuda_t.data),
            nbytes=int(y_cuda.nbytes),
            sync=True,
        )

        np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-5, atol=1e-5)

    t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
    t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

    cpu_med = _median(t_cpu)
    cuda_med = _median(t_cuda)

    print(
        f"dropout fwd  cpu={_fmt_seconds(cpu_med):>10}  "
        f"cuda={_fmt_seconds(cuda_med):>10}  speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--H", type=int, default=56)
    ap.add_argument("--W", type=int, default=56)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--presets", action="store_true")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no_sync_cuda",
        action="store_true",
        help="Do not sync after each CUDA forward",
    )
    args = ap.parse_args()

    if not (0.0 <= args.p < 1.0):
        raise SystemExit("p must be in [0, 1).")

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.presets:
        if args.big:
            cases = [
                Case("big-112", 32, 256, 112, 112),
                Case("big-224", 16, 256, 224, 224),
            ]
        else:
            cases = [
                Case("small-28", 128, 64, 28, 28),
                Case("mid-56", 64, 256, 56, 56),
                Case("mid-112", 32, 256, 112, 112),
            ]
        for c in cases:
            bench_case(
                c,
                p=float(args.p),
                dtype=dtype,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                sanity=bool(args.sanity),
                seed=int(args.seed),
                sync_cuda=not bool(args.no_sync_cuda),
            )
    else:
        c = Case("single", int(args.N), int(args.C), int(args.H), int(args.W))
        bench_case(
            c,
            p=float(args.p),
            dtype=dtype,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            sanity=bool(args.sanity),
            seed=int(args.seed),
            sync_cuda=not bool(args.no_sync_cuda),
        )


if __name__ == "__main__":
    main()
