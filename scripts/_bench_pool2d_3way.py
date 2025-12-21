"""
scripts/bench_pool2d_3way.py

3-way Pool2D microbenchmark (NOT a unit test) to produce a clear proof of speedup.

It benchmarks three implementations:

1) "pure_py"  : pure Python inner scan / accumulation (no NumPy reductions)
               -> this is the true "Python loops" baseline and is intentionally slow.

2) "numpy_ref": Python outer loops + NumPy per-patch reductions (np.argmax / np.sum)
               -> this matches the style of your original reference path.

3) "native"   : C++ kernels via ctypes (float32/float64), matching KeyDNN semantics.

Why this is useful
------------------
Your earlier "native vs numpy fallback" benchmark can show tiny gains because the
NumPy fallback already uses NumPy C-level reductions inside the loops. That means
the inner heavy work is already compiled, so switching to C++ may only help a bit.

This script provides the "desirable proof":
- pure_py vs native should show a big speedup (proving C++ loop matters)
- numpy_ref vs native may be small (explaining your earlier results)

What is timed
-------------
This script isolates the kernel as much as practical:
- Precomputes padded inputs outside timing (no np.pad in timed region)
- Preallocates outputs outside timing (no allocation in timed region)
- Loads the shared library once outside timing
- Times only the kernel work per iteration

Benchmarked ops (forward only for clarity / stable comparisons):
- MaxPool2D forward (k x k, stride s, padding p)
- AvgPool2D forward (k x k, stride s, padding p)

Notes
-----
- Run multiple times; use median.
- For huge shapes, lower repeats.
- dtype choices: float32 or float64.
- If the native library cannot be loaded, "native" will be skipped with a warning.

Usage examples
--------------
# Presets
python scripts/bench_pool2d_3way.py --presets --dtype float32

# Single case
python scripts/bench_pool2d_3way.py --N 8 --C 32 --H 128 --W 128 --k 2 --s 2 --p 0 --dtype float32

# Bigger cases with fewer repeats
python scripts/bench_pool2d_3way.py --presets --big --warmup 2 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import ctypes
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
    from keydnn.infrastructure.native.python.maxpool2d_ctypes import (
        load_keydnn_native,
        maxpool2d_forward_f32_ctypes,
        maxpool2d_forward_f64_ctypes,
    )
    from keydnn.infrastructure.native.python.avgpool2d_ctypes import (
        avgpool2d_forward_f32_ctypes,
        avgpool2d_forward_f64_ctypes,
    )

    _NATIVE_IMPORT_OK = True
except Exception:
    _NATIVE_IMPORT_OK = False


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def _out_hw(
    H: int, W: int, k: tuple[int, int], s: tuple[int, int], p: tuple[int, int]
) -> tuple[int, int]:
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
# 1) PURE PYTHON kernels (no NumPy reductions)
# -----------------------------------------------------------------------------
def _maxpool2d_forward_pure_py(
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    argmax_idx: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> None:
    # Pure Python inner scan, no np.argmax, no np.sum
    # Matches KeyDNN argmax semantics: store h * W_pad + w (spatial index into padded plane)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w

                    best = -float("inf")
                    best_ph = 0
                    best_pw = 0

                    # scan k_h x k_w
                    for ph in range(k_h):
                        h = h0 + ph
                        for pw in range(k_w):
                            w = w0 + pw
                            v = x_pad[n, c, h, w]
                            if v > best:
                                best = float(v)
                                best_ph = ph
                                best_pw = pw

                    y[n, c, i, j] = best
                    best_h = h0 + best_ph
                    best_w = w0 + best_pw
                    argmax_idx[n, c, i, j] = best_h * W_pad + best_w


def _avgpool2d_forward_pure_py(
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> None:
    denom = float(k_h * k_w)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    acc = 0.0
                    for ph in range(k_h):
                        h = h0 + ph
                        for pw in range(k_w):
                            w = w0 + pw
                            acc += float(x_pad[n, c, h, w])
                    y[n, c, i, j] = acc / denom


# -----------------------------------------------------------------------------
# 2) NUMPY-REF kernels (Python loops + NumPy reductions inside)
# -----------------------------------------------------------------------------
def _maxpool2d_forward_numpy_ref(
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    argmax_idx: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    W_pad: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> None:
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    flat_idx = int(np.argmax(patch))
                    y[n, c, i, j] = patch.reshape(-1)[flat_idx]

                    ph = flat_idx // k_w
                    pw = flat_idx % k_w
                    h = h0 + ph
                    w_ = w0 + pw
                    argmax_idx[n, c, i, j] = h * W_pad + w_


def _avgpool2d_forward_numpy_ref(
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> None:
    denom = float(k_h * k_w)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    y[n, c, i, j] = np.sum(patch) / denom


# -----------------------------------------------------------------------------
# 3) NATIVE kernels via ctypes
# -----------------------------------------------------------------------------
def _make_native_maxpool_fwd(
    lib: "ctypes.CDLL",
    *,
    dtype: np.dtype,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    x_pad: np.ndarray,
    y: np.ndarray,
    argmax_idx: np.ndarray,
) -> Callable[[], None]:
    if dtype == np.float32:
        return lambda: maxpool2d_forward_f32_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            y=y,
            argmax_idx=argmax_idx,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )
    if dtype == np.float64:
        return lambda: maxpool2d_forward_f64_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            y=y,
            argmax_idx=argmax_idx,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )
    raise TypeError("native benchmark supports only float32/float64")


def _make_native_avgpool_fwd(
    lib: "ctypes.CDLL",
    *,
    dtype: np.dtype,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    x_pad: np.ndarray,
    y: np.ndarray,
) -> Callable[[], None]:
    if dtype == np.float32:
        return lambda: avgpool2d_forward_f32_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            y=y,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )
    if dtype == np.float64:
        return lambda: avgpool2d_forward_f64_ctypes(  # type: ignore[name-defined]
            lib,
            x_pad=x_pad,
            y=y,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )
    raise TypeError("native benchmark supports only float32/float64")


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    N: int
    C: int
    H: int
    W: int
    k: int
    s: int
    p: int


def bench_case(
    case: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    do_sanity_check: bool,
) -> None:
    rng = np.random.default_rng(0)

    k_h, k_w = _pair(case.k)
    s_h, s_w = _pair(case.s)
    p_h, p_w = _pair(case.p)

    N, C, H, W = case.N, case.C, case.H, case.W
    H_out, W_out = _out_hw(H, W, (k_h, k_w), (s_h, s_w), (p_h, p_w))

    if H_out <= 0 or W_out <= 0:
        print(f"[skip] invalid output shape for {case}")
        return

    x = rng.standard_normal((N, C, H, W)).astype(dtype, copy=False)

    # Pre-pad outside timing
    x_pad_max = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    x_pad_avg = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    H_pad, W_pad = x_pad_max.shape[2], x_pad_max.shape[3]

    # Preallocate outputs
    y_max = np.empty((N, C, H_out, W_out), dtype=dtype)
    idx_max = np.empty((N, C, H_out, W_out), dtype=np.int64)

    y_avg = np.empty((N, C, H_out, W_out), dtype=dtype)

    # ----- Prepare callables -----
    def max_pure_py() -> None:
        _maxpool2d_forward_pure_py(
            x_pad=x_pad_max,
            y=y_max,
            argmax_idx=idx_max,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            H_pad=H_pad,
            W_pad=W_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

    def max_numpy_ref() -> None:
        _maxpool2d_forward_numpy_ref(
            x_pad=x_pad_max,
            y=y_max,
            argmax_idx=idx_max,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            W_pad=W_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

    def avg_pure_py() -> None:
        _avgpool2d_forward_pure_py(
            x_pad=x_pad_avg,
            y=y_avg,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

    def avg_numpy_ref() -> None:
        _avgpool2d_forward_numpy_ref(
            x_pad=x_pad_avg,
            y=y_avg,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

    native_lib = None
    max_native: Optional[Callable[[], None]] = None
    avg_native: Optional[Callable[[], None]] = None

    if _NATIVE_IMPORT_OK:
        try:
            native_lib = load_keydnn_native()  # type: ignore[name-defined]
            max_native = _make_native_maxpool_fwd(
                native_lib,
                dtype=dtype,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
                x_pad=x_pad_max,
                y=y_max,
                argmax_idx=idx_max,
            )
            avg_native = _make_native_avgpool_fwd(
                native_lib,
                dtype=dtype,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
                x_pad=x_pad_avg,
                y=y_avg,
            )
        except Exception as e:
            print(f"[warn] native library unavailable for this run: {e}")
            max_native = None
            avg_native = None

    # ----- Optional sanity check (not timed) -----
    if do_sanity_check:
        # Check that numpy_ref and native match (pure_py may be slower and uses float best comparisons)
        max_numpy_ref()
        y_ref = y_max.copy()
        idx_ref = idx_max.copy()

        avg_numpy_ref()
        y_avg_ref = y_avg.copy()

        if max_native is not None:
            max_native()
            np.testing.assert_allclose(y_max, y_ref, rtol=0, atol=0)
            np.testing.assert_array_equal(idx_max, idx_ref)

        if avg_native is not None:
            avg_native()
            np.testing.assert_allclose(y_avg, y_avg_ref, rtol=0, atol=0)

    # ----- Time it -----
    t_max_pure = _time_one(max_pure_py, warmup=warmup, repeats=repeats)
    t_max_np = _time_one(max_numpy_ref, warmup=warmup, repeats=repeats)

    t_avg_pure = _time_one(avg_pure_py, warmup=warmup, repeats=repeats)
    t_avg_np = _time_one(avg_numpy_ref, warmup=warmup, repeats=repeats)

    t_max_native = None
    t_avg_native = None
    if max_native is not None:
        t_max_native = _time_one(max_native, warmup=warmup, repeats=repeats)
    if avg_native is not None:
        t_avg_native = _time_one(avg_native, warmup=warmup, repeats=repeats)

    print("\n" + "=" * 98)
    print(
        f"{case.name}: N={N} C={C} H={H} W={W}  k={case.k} s={case.s} p={case.p}  dtype={dtype.__name__} "
        f"(warmup={warmup}, repeats={repeats})"
    )
    print("-" * 98)

    def row(op: str, a_label: str, a: float, b_label: str, b: float) -> None:
        print(
            f"{op:<14}  {a_label}={_fmt_seconds(a):>10}  {b_label}={_fmt_seconds(b):>10}  "
            f"speedup={_speedup(a, b):>7.2f}x"
        )

    # MaxPool
    max_pure_med = _median(t_max_pure)
    max_np_med = _median(t_max_np)
    row("maxpool fwd", "pure_py", max_pure_med, "numpy_ref", max_np_med)

    if t_max_native is not None:
        max_native_med = _median(t_max_native)
        row("maxpool fwd", "pure_py", max_pure_med, "native", max_native_med)
        row("maxpool fwd", "numpy_ref", max_np_med, "native", max_native_med)
    else:
        print("maxpool fwd   native=SKIPPED (library unavailable)")

    # AvgPool
    avg_pure_med = _median(t_avg_pure)
    avg_np_med = _median(t_avg_np)
    row("avgpool fwd", "pure_py", avg_pure_med, "numpy_ref", avg_np_med)

    if t_avg_native is not None:
        avg_native_med = _median(t_avg_native)
        row("avgpool fwd", "pure_py", avg_pure_med, "native", avg_native_med)
        row("avgpool fwd", "numpy_ref", avg_np_med, "native", avg_native_med)
    else:
        print("avgpool fwd   native=SKIPPED (library unavailable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--presets", action="store_true", help="Run a preset suite.")
    ap.add_argument("--big", action="store_true", help="Use larger preset shapes.")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Run correctness checks (not timed) between numpy_ref and native.",
    )
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.presets:
        if args.big:
            cases = [
                Case("big-128", 8, 32, 128, 128, 2, 2, 0),
                Case("big-256", 8, 32, 256, 256, 2, 2, 0),
                Case("big-384", 4, 64, 384, 384, 2, 2, 0),
                Case("big-512", 2, 64, 512, 512, 2, 2, 0),
            ]
        else:
            cases = [
                Case("mnist-ish", 1, 8, 28, 28, 2, 2, 0),
                Case("small", 8, 16, 32, 32, 2, 2, 0),
                Case("mid", 8, 32, 64, 64, 2, 2, 0),
                Case("bigger", 16, 64, 56, 56, 2, 2, 0),
            ]
        for c in cases:
            bench_case(
                c,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                do_sanity_check=args.sanity,
            )
    else:
        c = Case("single", args.N, args.C, args.H, args.W, args.k, args.s, args.p)
        bench_case(
            c,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            do_sanity_check=args.sanity,
        )


if __name__ == "__main__":
    main()
