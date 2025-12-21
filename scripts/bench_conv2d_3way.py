"""
scripts/bench_conv2d_3way.py

3-way Conv2D microbenchmark (NOT a unit test) to produce a clear proof of speedup.

It benchmarks three implementations (forward only for clarity / stable comparisons):

1) "pure_py"  : pure Python inner multiply-accumulate (no NumPy reductions)
               -> true "Python loops" baseline and intentionally slow.

2) "numpy_ref": Python outer loops + NumPy per-patch reductions (np.sum(patch * w))
               -> matches the style of the original reference implementation.

3) "native"   : C++ kernels via ctypes (float32/float64), matching KeyDNN semantics.

Tuning for practicality
-----------------------
Conv2D is expensive; the "pure_py" baseline can be extremely slow for larger shapes.

This script defaults to small shapes and fewer repeats. You can also:
- --skip-pure-py          to only compare numpy_ref vs native
- --pure-py-repeats K     to run fewer repeats for pure_py than others

What is timed
-------------
- Precomputes padded inputs outside timing (no np.pad in timed region)
- Preallocates outputs outside timing (no allocation in timed region)
- Loads the shared library once outside timing
- Times only the kernel work per iteration

Usage examples
--------------
# Fast preset suite
python scripts/bench_conv2d_3way.py --presets --dtype float32

# Fast preset suite without pure Python baseline (recommended for quick iteration)
python scripts/bench_conv2d_3way.py --presets --skip-pure-py --dtype float32

# Single case
python scripts/bench_conv2d_3way.py --N 1 --C_in 8 --H 28 --W 28 --C_out 8 --K 3 --s 1 --p 1 --dtype float32

# Sanity check (not timed) between pure_py, numpy_ref, and native
python scripts/bench_conv2d_3way.py --presets --sanity
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
# Native ctypes imports
# -------------------------
try:
    from keydnn.infrastructure.native.python.conv2d_ctypes import (
        load_keydnn_native,
        conv2d_forward_f32_ctypes,
        conv2d_forward_f64_ctypes,
    )

    _NATIVE_IMPORT_OK = True
except Exception:
    _NATIVE_IMPORT_OK = False


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def _out_hw(
    H: int, W: int, K_h: int, K_w: int, s: tuple[int, int], p: tuple[int, int]
) -> tuple[int, int]:
    s_h, s_w = s
    p_h, p_w = p
    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1
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


def _fmt_seconds(x: float) -> str:
    if x < 1e-6:
        return f"{x*1e9:.2f} ns"
    if x < 1e-3:
        return f"{x*1e6:.2f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def _median(xs: list[float]) -> float:
    return statistics.median(xs)


def _speedup(a: float, b: float) -> float:
    return (a / b) if b > 0 else float("inf")


# -----------------------------------------------------------------------------
# 1) PURE PYTHON kernel (no NumPy reductions)
# -----------------------------------------------------------------------------
def _conv2d_forward_pure_py(
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
    N: int,
    C_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
) -> None:
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    acc = 0.0
                    for ci in range(C_in):
                        for kh in range(K_h):
                            h = h0 + kh
                            for kw in range(K_w):
                                ww = w0 + kw
                                acc += float(x_pad[n, ci, h, ww]) * float(
                                    w[co, ci, kh, kw]
                                )
                    if b is not None:
                        acc += float(b[co])
                    y[n, co, i, j] = acc


# -----------------------------------------------------------------------------
# 2) NUMPY-REF kernel (Python loops + NumPy reductions inside)
# -----------------------------------------------------------------------------
def _conv2d_forward_numpy_ref(
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
    N: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
) -> None:
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, :, h0 : h0 + K_h, w0 : w0 + K_w]
                    y[n, co, i, j] = np.sum(patch * w[co])
            if b is not None:
                y[n, co, :, :] += b[co]


# -----------------------------------------------------------------------------
# 3) NATIVE kernel via ctypes
# -----------------------------------------------------------------------------
def _make_native_conv2d_fwd(
    lib: "object",
    *,
    dtype: np.dtype,
    N: int,
    C_in: int,
    H_pad: int,
    W_pad: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    x_pad: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
) -> Callable[[], None]:
    if dtype == np.float32:
        return lambda: conv2d_forward_f32_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            w=w,
            b=b,
            y=y,
            N=N,
            C_in=C_in,
            H_pad=H_pad,
            W_pad=W_pad,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
        )
    if dtype == np.float64:
        return lambda: conv2d_forward_f64_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            w=w,
            b=b,
            y=y,
            N=N,
            C_in=C_in,
            H_pad=H_pad,
            W_pad=W_pad,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
        )
    raise TypeError("native benchmark supports only float32/float64")


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    N: int
    C_in: int
    H: int
    W: int
    C_out: int
    K: int
    s: int
    p: int


def bench_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    pure_py_repeats: int,
    do_sanity_check: bool,
    use_bias: bool,
    skip_pure_py: bool,
) -> None:
    rng = np.random.default_rng(0)

    K_h, K_w = _pair(case.K)
    s_h, s_w = _pair(case.s)
    p_h, p_w = _pair(case.p)

    N, C_in, H, W = case.N, case.C_in, case.H, case.W
    C_out = case.C_out
    H_out, W_out = _out_hw(H, W, K_h, K_w, (s_h, s_w), (p_h, p_w))

    if H_out <= 0 or W_out <= 0:
        print(f"[skip] invalid output shape for {case}")
        return

    x = rng.standard_normal((N, C_in, H, W)).astype(dtype, copy=False)
    w = rng.standard_normal((C_out, C_in, K_h, K_w)).astype(dtype, copy=False)

    b = None
    if use_bias:
        b = rng.standard_normal((C_out,)).astype(dtype, copy=False)

    # Pre-pad outside timing
    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    # Preallocate outputs
    y = np.empty((N, C_out, H_out, W_out), dtype=dtype)

    # ----- Prepare callables -----
    def pure_py() -> None:
        _conv2d_forward_pure_py(
            x_pad=x_pad,
            w=w,
            b=b,
            y=y,
            N=N,
            C_in=C_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
        )

    def numpy_ref() -> None:
        _conv2d_forward_numpy_ref(
            x_pad=x_pad,
            w=w,
            b=b,
            y=y,
            N=N,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
        )

    native_lib = None
    native: Optional[Callable[[], None]] = None

    if _NATIVE_IMPORT_OK:
        try:
            native_lib = load_keydnn_native()  # type: ignore[name-defined]
            native = _make_native_conv2d_fwd(
                native_lib,
                dtype=dtype,
                N=N,
                C_in=C_in,
                H_pad=H_pad,
                W_pad=W_pad,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                x_pad=x_pad,
                w=w,
                b=b,
                y=y,
            )
        except Exception as e:
            print(f"[warn] native library unavailable for this run: {e}")
            native = None

    # ----- Optional sanity check (not timed) -----
    if do_sanity_check:
        numpy_ref()
        y_ref = y.copy()

        if not skip_pure_py:
            pure_py()
            np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)

        if native is not None:
            native()
            np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)

    # ----- Time it -----
    t_pure = None
    if not skip_pure_py:
        t_pure = _time_one(pure_py, warmup=warmup, repeats=pure_py_repeats)

    t_np = _time_one(numpy_ref, warmup=warmup, repeats=repeats)

    t_native = None
    if native is not None:
        t_native = _time_one(native, warmup=warmup, repeats=repeats)

    print("\n" + "=" * 98)
    bias_s = "on" if use_bias else "off"
    print(
        f"{case.name}: N={N} C_in={C_in} H={H} W={W}  C_out={C_out}  K={case.K} s={case.s} p={case.p}  "
        f"bias={bias_s}  dtype={dtype.__name__} (warmup={warmup}, repeats={repeats}, pure_py_repeats={pure_py_repeats})"
    )
    print("-" * 98)

    def row(a_label: str, a: float, b_label: str, b: float) -> None:
        print(
            f"conv2d fwd     {a_label}={_fmt_seconds(a):>10}  {b_label}={_fmt_seconds(b):>10}  "
            f"speedup={_speedup(a, b):>7.2f}x"
        )

    np_med = _median(t_np)
    if t_pure is not None:
        pure_med = _median(t_pure)
        row("pure_py", pure_med, "numpy_ref", np_med)
    else:
        print("conv2d fwd     pure_py=SKIPPED")

    if t_native is not None:
        native_med = _median(t_native)
        if t_pure is not None:
            row("pure_py", _median(t_pure), "native", native_med)
        row("numpy_ref", np_med, "native", native_med)
    else:
        print("conv2d fwd     native=SKIPPED (library unavailable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--C_in", type=int, default=8)
    ap.add_argument("--H", type=int, default=28)
    ap.add_argument("--W", type=int, default=28)
    ap.add_argument("--C_out", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument(
        "--pure-py-repeats",
        type=int,
        default=3,
        help="Repeat count for the pure_py baseline (often much slower).",
    )
    ap.add_argument("--presets", action="store_true", help="Run a preset suite.")
    ap.add_argument("--big", action="store_true", help="Use larger preset shapes.")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Run correctness checks (not timed) between numpy_ref/native and pure_py.",
    )
    ap.add_argument(
        "--no-bias",
        action="store_true",
        help="Disable bias in all benchmarked cases.",
    )
    ap.add_argument(
        "--skip-pure-py",
        action="store_true",
        help="Skip the pure_py baseline (recommended for quick iteration).",
    )
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    use_bias = not args.no_bias

    if args.presets:
        if args.big:
            # Still "small enough" to run, but noticeably heavier.
            cases = [
                Case("big-32", 1, 16, 32, 32, 16, 3, 1, 1),
                Case("big-64", 1, 16, 64, 64, 16, 3, 1, 1),
            ]
        else:
            # Fast presets (MNIST-ish / tiny CNN-ish)
            cases = [
                Case("mnist-ish", 1, 8, 28, 28, 8, 3, 1, 1),
                Case("tiny", 1, 8, 16, 16, 8, 3, 1, 1),
                Case("small", 1, 16, 32, 32, 16, 3, 1, 1),
                Case("downsample", 1, 8, 28, 28, 8, 3, 2, 1),
            ]
        for c in cases:
            bench_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                pure_py_repeats=args.pure_py_repeats,
                do_sanity_check=args.sanity,
                use_bias=use_bias,
                skip_pure_py=args.skip_pure_py,
            )
    else:
        c = Case(
            "single",
            args.N,
            args.C_in,
            args.H,
            args.W,
            args.C_out,
            args.K,
            args.s,
            args.p,
        )
        bench_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            pure_py_repeats=args.pure_py_repeats,
            do_sanity_check=args.sanity,
            use_bias=use_bias,
            skip_pure_py=args.skip_pure_py,
        )


if __name__ == "__main__":
    main()
