"""
scripts/bench_pool2d_native_vs_numpy.py

Benchmark script (NOT a unit test) to compare KeyDNN Pool2D performance between:
1) NumPy reference (Python loops)
2) Native C++ kernels via ctypes (float32/float64)

It measures forward/backward for:
- maxpool2d_forward_cpu / maxpool2d_backward_cpu
- avgpool2d_forward_cpu / avgpool2d_backward_cpu

How it works
------------
- "native" mode: calls the public ops normally (they should dispatch to ctypes
  for float32/float64 when the shared library is available).
- "numpy" mode: forces native load to fail via monkeypatch so ops fall back to
  the Python loop/reference implementation.

Usage examples
--------------
# Default: benchmark a few shapes
python scripts/bench_pool2d_native_vs_numpy.py

# Benchmark a single shape
python scripts/bench_pool2d_native_vs_numpy.py --N 8 --C 16 --H 64 --W 64 --k 2 --s 2 --p 0

# More repeats (more stable)
python scripts/bench_pool2d_native_vs_numpy.py --repeats 30 --warmup 5

Notes
-----
- Run with CPU scaling disabled / consistent power mode if possible.
- Close other apps and run multiple times for stable results.
- This benchmark includes Python overhead for calling the op; it’s meant to
  measure end-to-end speedup at the Python API level.
"""

from __future__ import annotations

import os
import sys

# Ensure repo_root/src is importable when running this file directly:
# repo_root/
#   src/keydnn/...
#   scripts/bench_pool2d_native_vs_numpy.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import statistics
import time
import warnings
from contextlib import contextmanager
from typing import Callable

import numpy as np

from keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
)


@contextmanager
def _force_numpy_fallback():
    """
    Force Pool2D ops to fall back to NumPy reference implementations by making
    native library loading fail (simulated OSError).

    This is intentionally not using unittest.mock to keep it as a standalone
    script.
    """
    import importlib

    # Patch both modules that might be used depending on how pool2d_cpu imports them
    # (safe to patch both).
    targets = [
        "src.keydnn.infrastructure.native.python.maxpool2d_ctypes",
        "src.keydnn.infrastructure.native.python.avgpool2d_ctypes",
    ]

    originals = []
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            originals.append((None, None, None))
            continue

        if hasattr(mod, "load_keydnn_native"):
            originals.append((mod, "load_keydnn_native", mod.load_keydnn_native))

            def _raise(*args, **kwargs):
                raise OSError("forced numpy fallback (benchmark)")

            mod.load_keydnn_native = _raise  # type: ignore[attr-defined]
        else:
            originals.append((mod, None, None))

    try:
        yield
    finally:
        for mod, attr, orig in originals:
            if mod is not None and attr and orig is not None:
                setattr(mod, attr, orig)


def _time_one(fn: Callable[[], None], *, warmup: int, repeats: int) -> list[float]:
    # Warmup
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _fmt_seconds(x: float) -> str:
    if x < 1e-6:
        return f"{x*1e9:.2f} ns"
    if x < 1e-3:
        return f"{x*1e6:.2f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def _summarize(label: str, samples: list[float]) -> dict[str, float]:
    return {
        "label": label,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _print_row(name: str, numpy_s: float, native_s: float) -> None:
    speedup = (numpy_s / native_s) if native_s > 0 else float("inf")
    print(
        f"{name:<22}  "
        f"numpy(median)={_fmt_seconds(numpy_s):>10}  "
        f"native(median)={_fmt_seconds(native_s):>10}  "
        f"speedup={speedup:>7.2f}x"
    )


def bench_one(
    *,
    N: int,
    C: int,
    H: int,
    W: int,
    k: int,
    s: int,
    p: int,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, C, H, W)).astype(dtype, copy=False)

    # Use warnings as errors? No; we silence them to avoid overhead skew.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # -------- MaxPool forward --------
    def run_max_fwd() -> None:
        y, idx = maxpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
        # touch outputs to discourage extreme dead-code elimination / caching illusions
        _ = float(np.sum(y)) + float(np.sum(idx))

    with _force_numpy_fallback():
        t_numpy = _time_one(run_max_fwd, warmup=warmup, repeats=repeats)
    t_native = _time_one(run_max_fwd, warmup=warmup, repeats=repeats)

    # -------- MaxPool backward --------
    # Prepare forward once to get idx and output shape.
    y0, idx0 = maxpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
    grad_out = np.ones_like(y0)

    def run_max_bwd() -> None:
        gx = maxpool2d_backward_cpu(
            grad_out, idx0, x_shape=x.shape, kernel_size=k, stride=s, padding=p
        )
        _ = float(np.sum(gx))

    with _force_numpy_fallback():
        tb_numpy = _time_one(run_max_bwd, warmup=warmup, repeats=repeats)
    tb_native = _time_one(run_max_bwd, warmup=warmup, repeats=repeats)

    # -------- AvgPool forward --------
    def run_avg_fwd() -> None:
        y = avgpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
        _ = float(np.sum(y))

    with _force_numpy_fallback():
        ta_numpy = _time_one(run_avg_fwd, warmup=warmup, repeats=repeats)
    ta_native = _time_one(run_avg_fwd, warmup=warmup, repeats=repeats)

    # -------- AvgPool backward --------
    y1 = avgpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)
    grad_out2 = np.ones_like(y1)

    def run_avg_bwd() -> None:
        gx = avgpool2d_backward_cpu(
            grad_out2, x_shape=x.shape, kernel_size=k, stride=s, padding=p
        )
        _ = float(np.sum(gx))

    with _force_numpy_fallback():
        tab_numpy = _time_one(run_avg_bwd, warmup=warmup, repeats=repeats)
    tab_native = _time_one(run_avg_bwd, warmup=warmup, repeats=repeats)

    # Print results (median is usually the most stable)
    print("\n" + "=" * 90)
    print(
        f"Shape: N={N} C={C} H={H} W={W}  k={k} s={s} p={p}  dtype={dtype}  "
        f"(warmup={warmup}, repeats={repeats})"
    )
    print("-" * 90)
    _print_row(
        "maxpool2d_forward", statistics.median(t_numpy), statistics.median(t_native)
    )
    _print_row(
        "maxpool2d_backward", statistics.median(tb_numpy), statistics.median(tb_native)
    )
    _print_row(
        "avgpool2d_forward", statistics.median(ta_numpy), statistics.median(ta_native)
    )
    _print_row(
        "avgpool2d_backward",
        statistics.median(tab_numpy),
        statistics.median(tab_native),
    )

    # Optional extra stats
    print("-" * 90)
    for name, arr in [
        ("max_fwd_numpy", t_numpy),
        ("max_fwd_native", t_native),
        ("max_bwd_numpy", tb_numpy),
        ("max_bwd_native", tb_native),
        ("avg_fwd_numpy", ta_numpy),
        ("avg_fwd_native", ta_native),
        ("avg_bwd_numpy", tab_numpy),
        ("avg_bwd_native", tab_native),
    ]:
        s = _summarize(name, arr)
        print(
            f"{s['label']:<14} mean={_fmt_seconds(s['mean']):>10} "
            f"median={_fmt_seconds(s['median']):>10} "
            f"min={_fmt_seconds(s['min']):>10} "
            f"max={_fmt_seconds(s['max']):>10}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument(
        "--dtypes",
        nargs="*",
        default=["float32", "float64"],
        choices=["float32", "float64", "float16"],
        help="Dtypes to benchmark. float16 will always use NumPy reference path.",
    )
    ap.add_argument(
        "--presets",
        action="store_true",
        help="Benchmark a small set of preset shapes instead of a single shape.",
    )
    args = ap.parse_args()

    dtypes = [getattr(np, dt) for dt in args.dtypes]

    if args.presets:
        # A few common-ish shapes for quick comparison.
        presets = [
            (8, 32, 128, 128),
            (8, 64, 128, 128),
            (8, 64, 256, 256),
            (4, 64, 384, 384),
            # (2, 64, 512, 512),
        ]

        for dtype in dtypes:
            for N, C, H, W in presets:
                bench_one(
                    N=N,
                    C=C,
                    H=H,
                    W=W,
                    k=args.k,
                    s=args.s,
                    p=args.p,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
    else:
        for dtype in dtypes:
            bench_one(
                N=args.N,
                C=args.C,
                H=args.H,
                W=args.W,
                k=args.k,
                s=args.s,
                p=args.p,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
            )


if __name__ == "__main__":
    main()
