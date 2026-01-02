"""
scripts/bench_pool2d_cpu_vs_cuda.py

CPU vs CUDA pooling microbenchmark (NOT a unit test) for KeyDNN.

Benchmarks forward-only kernels for:
- maxpool2d
- avgpool2d
- global_avgpool2d

Timing policy
-------------
- Excludes HtoD/DtoH transfers (performed once per case).
- Times only CPU forward or CUDA forward (with sync=True if supported).
- Preallocates output buffers and device buffers outside timed region.

Notes
-----
This benchmark is aligned with the current CUDA pooling wrapper semantics:

- maxpool2d/avgpool2d kernels expect a *padded* input buffer on device
  (x_pad_dev) and explicit (H_pad, W_pad, H_out, W_out).
- Padding is performed on GPU via pad2d_cuda:
  - maxpool2d pads with -inf
  - avgpool2d pads with 0.0
- maxpool2d argmax indices are int64 on device (matches pool2d_cuda_ext.py).

Usage
-----
python scripts/bench_pool2d_cpu_vs_cuda.py --presets --dtype float32
python scripts/bench_pool2d_cpu_vs_cuda.py --N 8 --C 32 --H 64 --W 64 --k 2 --s 2 --p 0 --dtype float32
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

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
# CPU ops imports
# -------------------------
from keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    avgpool2d_forward_cpu,
    global_avgpool2d_forward_cpu,
)

# -------------------------
# CUDA ctypes imports
# -------------------------
try:
    from keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as cuda_utils
    from keydnn.infrastructure.ops.pool2d_cuda import (
        pad2d_cuda,
        maxpool2d_forward_cuda,
        avgpool2d_forward_cuda,
        global_avgpool2d_forward_cuda,
    )

    _CUDA_IMPORT_OK = True
except Exception:
    _CUDA_IMPORT_OK = False


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def _out_hw_pool2d(
    H: int, W: int, k: Tuple[int, int], s: Tuple[int, int], p: Tuple[int, int]
) -> Tuple[int, int]:
    k_h, k_w = k
    s_h, s_w = s
    p_h, p_w = p
    H_out = (H + 2 * p_h - k_h) // s_h + 1
    W_out = (W + 2 * p_w - k_w) // s_w + 1
    return H_out, W_out


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
    N: int
    C: int
    H: int
    W: int
    k: int | Tuple[int, int]
    s: int | Tuple[int, int]
    p: int | Tuple[int, int]


class _CudaContext:
    def __init__(self) -> None:
        if not _CUDA_IMPORT_OK:
            raise RuntimeError(
                "CUDA pooling ctypes modules failed to import. "
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


# -----------------------------------------------------------------------------
# CUDA wrapper adapters
# -----------------------------------------------------------------------------
def _try_calls(
    fn: Any, call_variants: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    last_err: Optional[BaseException] = None
    for args, kwargs in call_variants:
        try:
            fn(*args, **kwargs)
            return
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("No call variants provided.")


def _cuda_maxpool2d_fwd(
    *,
    lib: Any,
    x_pad_dev: int,
    y_dev: int,
    argmax_idx_dev: int,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k: Tuple[int, int],
    s: Tuple[int, int],
    dtype: np.dtype,
) -> None:
    k_h, k_w = k
    s_h, s_w = s

    maxpool2d_forward_cuda(
        lib,
        x_pad_dev=int(x_pad_dev),
        y_dev=int(y_dev),
        argmax_idx_dev=int(argmax_idx_dev),
        N=int(N),
        C=int(C),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        H_out=int(H_out),
        W_out=int(W_out),
        k_h=int(k_h),
        k_w=int(k_w),
        s_h=int(s_h),
        s_w=int(s_w),
        dtype=dtype,
        sync=True,
    )


def _cuda_avgpool2d_fwd(
    *,
    lib: Any,
    x_pad_dev: int,
    y_dev: int,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k: Tuple[int, int],
    s: Tuple[int, int],
    dtype: np.dtype,
) -> None:
    k_h, k_w = k
    s_h, s_w = s

    avgpool2d_forward_cuda(
        lib,
        x_pad_dev=int(x_pad_dev),
        y_dev=int(y_dev),
        N=int(N),
        C=int(C),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        H_out=int(H_out),
        W_out=int(W_out),
        k_h=int(k_h),
        k_w=int(k_w),
        s_h=int(s_h),
        s_w=int(s_w),
        dtype=dtype,
        sync=True,
    )


def _cuda_global_avgpool2d_fwd(
    *,
    lib: Any,
    x_dev: int,
    y_dev: int,
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: np.dtype,
) -> None:
    variants: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((lib, int(x_dev), int(y_dev), int(N), int(C), int(H), int(W), dtype), {}),
        (
            (lib, int(x_dev), int(y_dev), int(N), int(C), int(H), int(W), dtype),
            {"sync": True},
        ),
        (
            (lib,),
            {
                "x": int(x_dev),
                "y": int(y_dev),
                "N": int(N),
                "C": int(C),
                "H": int(H),
                "W": int(W),
                "dtype": dtype,
                "sync": True,
            },
        ),
        (
            (lib,),
            {
                "x_dev": int(x_dev),
                "y_dev": int(y_dev),
                "N": int(N),
                "C": int(C),
                "H": int(H),
                "W": int(W),
                "dtype": dtype,
                "sync": True,
            },
        ),
    ]
    _try_calls(global_avgpool2d_forward_cuda, variants)


# -----------------------------------------------------------------------------
# Bench kernels
# -----------------------------------------------------------------------------
def _bench_maxpool2d_case(
    *,
    ctx: _CudaContext,
    x: np.ndarray,
    k: Tuple[int, int],
    s: Tuple[int, int],
    p: Tuple[int, int],
    warmup: int,
    repeats: int,
    sanity: bool,
) -> None:
    N, C, H, W = x.shape
    H_out, W_out = _out_hw_pool2d(H, W, k, s, p)
    if H_out <= 0 or W_out <= 0:
        print("[skip] invalid output shape for maxpool2d")
        return

    p_h, p_w = p
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    y_cpu = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    y_cuda = np.empty_like(y_cpu)

    # NOTE: argmax indices are int64 (matches pool2d_cuda_ext.py)
    argmax = np.empty((N, C, H_out, W_out), dtype=np.int64)

    x_dev = ctx.malloc(x.nbytes)
    x_pad_dev = ctx.malloc(int(N * C * H_pad * W_pad * x.dtype.itemsize))
    y_dev = ctx.malloc(y_cpu.nbytes)
    argmax_dev = ctx.malloc(argmax.nbytes)

    try:
        ctx.h2d(x_dev, x)

        # Pad on GPU once per case (excluded from timed region).
        pad2d_cuda(
            ctx.lib,
            x_dev=int(x_dev),
            y_pad_dev=int(x_pad_dev),
            N=int(N),
            C=int(C),
            H=int(H),
            W=int(W),
            p_h=int(p_h),
            p_w=int(p_w),
            pad_value=float(-np.inf),
            dtype=x.dtype,
            device=0,
            sync=True,
        )

        def cpu_fwd() -> None:
            y, _ = maxpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
            y_cpu[...] = y

        def cuda_fwd() -> None:
            _cuda_maxpool2d_fwd(
                lib=ctx.lib,
                x_pad_dev=x_pad_dev,
                y_dev=y_dev,
                argmax_idx_dev=argmax_dev,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k=k,
                s=s,
                dtype=x.dtype,
            )

        if sanity:
            cpu_fwd()
            cuda_fwd()
            ctx.d2h(y_cuda, y_dev)
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-5, atol=1e-5)

        t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
        t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

        cpu_med = _median(t_cpu)
        cuda_med = _median(t_cuda)
        print(
            f"maxpool2d fwd  cpu={_fmt_seconds(cpu_med):>10}  "
            f"cuda={_fmt_seconds(cuda_med):>10}  speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
        )
    finally:
        ctx.free(x_dev)
        ctx.free(x_pad_dev)
        ctx.free(y_dev)
        ctx.free(argmax_dev)


def _bench_avgpool2d_case(
    *,
    ctx: _CudaContext,
    x: np.ndarray,
    k: Tuple[int, int],
    s: Tuple[int, int],
    p: Tuple[int, int],
    warmup: int,
    repeats: int,
    sanity: bool,
) -> None:
    N, C, H, W = x.shape
    H_out, W_out = _out_hw_pool2d(H, W, k, s, p)
    if H_out <= 0 or W_out <= 0:
        print("[skip] invalid output shape for avgpool2d")
        return

    p_h, p_w = p
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    y_cpu = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    y_cuda = np.empty_like(y_cpu)

    x_dev = ctx.malloc(x.nbytes)
    x_pad_dev = ctx.malloc(int(N * C * H_pad * W_pad * x.dtype.itemsize))
    y_dev = ctx.malloc(y_cpu.nbytes)

    try:
        ctx.h2d(x_dev, x)

        # Pad on GPU once per case (excluded from timed region).
        pad2d_cuda(
            ctx.lib,
            x_dev=int(x_dev),
            y_pad_dev=int(x_pad_dev),
            N=int(N),
            C=int(C),
            H=int(H),
            W=int(W),
            p_h=int(p_h),
            p_w=int(p_w),
            pad_value=0.0,
            dtype=x.dtype,
            device=0,
            sync=True,
        )

        def cpu_fwd() -> None:
            y = avgpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
            y_cpu[...] = y

        def cuda_fwd() -> None:
            _cuda_avgpool2d_fwd(
                lib=ctx.lib,
                x_pad_dev=x_pad_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k=k,
                s=s,
                dtype=x.dtype,
            )

        if sanity:
            cpu_fwd()
            cuda_fwd()
            ctx.d2h(y_cuda, y_dev)
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-5, atol=1e-5)

        t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
        t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

        cpu_med = _median(t_cpu)
        cuda_med = _median(t_cuda)
        print(
            f"avgpool2d fwd  cpu={_fmt_seconds(cpu_med):>10}  "
            f"cuda={_fmt_seconds(cuda_med):>10}  speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
        )
    finally:
        ctx.free(x_dev)
        ctx.free(x_pad_dev)
        ctx.free(y_dev)


def _bench_global_avgpool2d_case(
    *,
    ctx: _CudaContext,
    x: np.ndarray,
    warmup: int,
    repeats: int,
    sanity: bool,
) -> None:
    N, C, H, W = x.shape
    y_cpu = np.empty((N, C, 1, 1), dtype=x.dtype)
    y_cuda = np.empty_like(y_cpu)

    x_dev = ctx.malloc(x.nbytes)
    y_dev = ctx.malloc(y_cpu.nbytes)

    try:
        ctx.h2d(x_dev, x)

        def cpu_fwd() -> None:
            y = global_avgpool2d_forward_cpu(x)
            y_cpu[...] = y

        def cuda_fwd() -> None:
            _cuda_global_avgpool2d_fwd(
                lib=ctx.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                dtype=x.dtype,
            )

        if sanity:
            cpu_fwd()
            cuda_fwd()
            ctx.d2h(y_cuda, y_dev)
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-5, atol=1e-5)

        t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
        t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

        cpu_med = _median(t_cpu)
        cuda_med = _median(t_cuda)
        print(
            f"gavgpool2d fwd cpu={_fmt_seconds(cpu_med):>10}  "
            f"cuda={_fmt_seconds(cuda_med):>10}  speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
        )
    finally:
        ctx.free(x_dev)
        ctx.free(y_dev)


def bench_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    rng_seed: int,
    ctx: _CudaContext,
) -> None:
    rng = np.random.default_rng(rng_seed)

    k = _pair(case.k)
    s = _pair(case.s)
    p = _pair(case.p)

    x = rng.standard_normal((case.N, case.C, case.H, case.W)).astype(dtype, copy=False)

    print("\n" + "=" * 98)
    print(
        f"{case.name}: N={case.N} C={case.C} H={case.H} W={case.W} "
        f"k={k} s={s} p={p} dtype={_dtype_str(dtype)} "
        f"(warmup={warmup}, repeats={repeats})"
    )
    print("-" * 98)

    _bench_maxpool2d_case(
        ctx=ctx, x=x, k=k, s=s, p=p, warmup=warmup, repeats=repeats, sanity=sanity
    )
    _bench_avgpool2d_case(
        ctx=ctx, x=x, k=k, s=s, p=p, warmup=warmup, repeats=repeats, sanity=sanity
    )
    # _bench_global_avgpool2d_case(
    #     ctx=ctx, x=x, warmup=warmup, repeats=repeats, sanity=sanity
    # )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--presets", action="store_true", help="Run a preset suite.")
    ap.add_argument("--big", action="store_true", help="Use larger preset shapes.")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Sanity-check CPU vs CUDA outputs (not timed).",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="RNG seed for input generation."
    )
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if not _CUDA_IMPORT_OK:
        raise SystemExit(
            "CUDA wrappers not importable. Build / install CUDA native DLL and python wrappers first."
        )

    ctx = _CudaContext()

    if args.presets:
        if args.big:
            cases = [
                Case("big-64", 8, 32, 64, 64, 2, 2, 0),
                Case("big-128", 8, 64, 128, 128, 2, 2, 0),
                Case("big-224", 4, 64, 224, 224, 2, 2, 0),
            ]
        else:
            cases = [
                Case("mnist-ish", 16, 8, 28, 28, 2, 2, 0),
                Case("small-32", 16, 16, 32, 32, 2, 2, 0),
                Case("mid-64", 8, 32, 64, 64, 2, 2, 0),
                Case("stride1-pad1", 8, 32, 64, 64, 3, 1, 1),
            ]
        for c in cases:
            bench_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                sanity=args.sanity,
                rng_seed=args.seed,
                ctx=ctx,
            )
    else:
        c = Case("single", args.N, args.C, args.H, args.W, args.k, args.s, args.p)
        bench_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            sanity=args.sanity,
            rng_seed=args.seed,
            ctx=ctx,
        )


if __name__ == "__main__":
    main()
