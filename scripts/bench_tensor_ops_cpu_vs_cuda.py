# scripts/bench_tensor_ops_cpu_vs_cuda.py
"""
Microbench: Tensor arithmetic / comparison / unary ops (CPU vs CUDA).

What it measures
----------------
- Per-op latency for a selected set of elementwise ops on CPU and CUDA.
- Uses warmup iterations (not recorded), then repeats with median/p95.
- For CUDA, optionally synchronizes after each iteration so timings are accurate.

Notes
-----
- This benchmark intentionally includes the Python boundary overhead (wrapper, alloc,
  Tensor construction) because that's often the dominating cost for small tensors.
- Inputs are allocated once and reused. Outputs are created each iteration (as your
  current CUDA ext ops allocate new device buffers).

Example
-------
python -O scripts/bench_tensor_ops_cpu_vs_cuda.py --ops add sub mul div gt neg exp \
    --shape 256 32 --dtype float32 --warmup 50 --repeats 200 --sync_each_iter
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ----------------------------
# Project imports (KeyDNN)
# ----------------------------
def _import_keydnn():
    # Adjust these imports if your paths differ.
    from keydnn.infrastructure.tensor._tensor import Tensor  # type: ignore
    from keydnn.domain.device._device import Device  # type: ignore

    return Tensor, Device


def _cuda_available() -> bool:
    try:
        # Any known-good loader in your tree is fine.
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _get_cuda_lib():
    # Prefer Tensor._get_cuda_lib() if present to ensure a shared handle.
    Tensor, _Device = _import_keydnn()
    if hasattr(Tensor, "_get_cuda_lib"):
        return Tensor._get_cuda_lib()
    from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
        load_keydnn_cuda_native,
    )

    return load_keydnn_cuda_native()


def _cuda_sync(lib) -> None:
    # If you already have a Tensor._cuda_sync(), you can call that instead.
    try:
        from keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes import (  # type: ignore
            cuda_synchronize,
        )

        cuda_synchronize(lib)
    except Exception:
        # As a fallback, do nothing (bench will still run, but timings may be optimistic).
        pass


def _h2d(lib, dst_dev: int, host_arr: np.ndarray) -> None:
    # Use compat alias (keyword-only + sync)
    from keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc  # type: ignore

    mc.memcpy_htod(
        lib,
        dst_dev=int(dst_dev),
        src_host=host_arr,
        nbytes=int(host_arr.nbytes),
        sync=True,
    )


def _d2h(lib, src_dev: int, host_arr: np.ndarray) -> None:
    from keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc  # type: ignore

    mc.memcpy_dtoh(
        lib,
        dst_host=host_arr,
        src_dev=int(src_dev),
        nbytes=int(host_arr.nbytes),
        sync=True,
    )


def _make_cpu_tensor_from_numpy(arr: np.ndarray):
    Tensor, Device = _import_keydnn()
    t = Tensor(
        shape=arr.shape, device=Device("cpu"), requires_grad=False, dtype=arr.dtype
    )
    # Many of your CPU tensors use copy_from_numpy; keep it consistent with the framework.
    t.copy_from_numpy(arr)
    return t


def _make_cuda_tensor_from_numpy(arr: np.ndarray, device_str: str = "cuda:0"):
    Tensor, Device = _import_keydnn()
    lib = _get_cuda_lib()

    t = Tensor(
        shape=arr.shape, device=Device(device_str), requires_grad=False, dtype=arr.dtype
    )
    # Ensure device allocation exists
    if hasattr(t, "_ensure_cuda_alloc"):
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    else:
        raise RuntimeError(
            "Tensor lacks _ensure_cuda_alloc; cannot build CUDA tensors for bench."
        )

    # Copy host -> device
    _h2d(lib, int(t.data), arr)
    return t


def _to_numpy_cpu(t) -> np.ndarray:
    # Your framework likely has to_numpy / numpy conversion helpers.
    if hasattr(t, "to_numpy"):
        return t.to_numpy()
    if hasattr(t, "_to_numpy"):
        return t._to_numpy()
    raise RuntimeError("Tensor has no to_numpy() method")


def _to_numpy_cuda(t) -> np.ndarray:
    lib = _get_cuda_lib()
    arr = np.empty(tuple(t.shape), dtype=np.dtype(t.dtype))
    _d2h(lib, int(t.data), arr)
    return arr


# ----------------------------
# Stats helpers
# ----------------------------
def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _p95(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(math.ceil(0.95 * len(ys))) - 1
    k = max(0, min(k, len(ys) - 1))
    return ys[k]


def _fmt_ms(sec: float) -> str:
    return f"{sec * 1e3:8.3f} ms"


def _fmt_us(sec: float) -> str:
    return f"{sec * 1e6:8.1f} µs"


@dataclass
class OpResult:
    name: str
    cpu_med: float
    cpu_p95: float
    gpu_med: float
    gpu_p95: float


# ----------------------------
# Bench core
# ----------------------------
def _time_op(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    sync_each_iter: bool,
    is_cuda: bool,
) -> List[float]:
    lib = _get_cuda_lib() if is_cuda else None

    # warmup
    for _ in range(warmup):
        fn()
        if is_cuda and sync_each_iter:
            _cuda_sync(lib)

    # measured
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if is_cuda and sync_each_iter:
            _cuda_sync(lib)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def _build_ops(a, b, alpha: float) -> Dict[str, Callable[[], object]]:
    # Use Python operators so we measure your Tensor mixins + ext dispatch.
    # Each call returns a new Tensor (or equivalent).
    ops: Dict[str, Callable[[], object]] = {
        "add": lambda: a + b,
        "sub": lambda: a - b,
        "mul": lambda: a * b,
        "div": lambda: a / b,
        "gt": lambda: (a > b),
        "neg": lambda: (-a),
    }

    # Optional unary exp if your Tensor supports it
    if hasattr(a, "exp") and callable(getattr(a, "exp")):
        ops["exp"] = lambda: a.exp()
    else:
        # Many frameworks implement exp as a function: Tensor.exp(a) or keydnn.exp(a)
        try:
            from keydnn.infrastructure._function import exp as fn_exp  # type: ignore

            ops["exp"] = lambda: fn_exp(a)
        except Exception:
            pass

    # Optional scalar fast-paths (if implemented as magic methods or helper methods)
    # This tries to benchmark (a + alpha) and (a * alpha) etc.
    try:
        ops["add_scalar"] = lambda: a + alpha
        ops["sub_scalar"] = lambda: a - alpha
        ops["mul_scalar"] = lambda: a * alpha
        ops["div_scalar"] = lambda: a / alpha
    except Exception:
        pass

    return ops


def _sanity_check(
    cpu_out: np.ndarray, gpu_out: np.ndarray, name: str, dtype: np.dtype
) -> None:
    if name == "gt":
        # gt outputs may be float32 by design on CUDA in your kernel ABI.
        # Compare as float32.
        cpu_cmp = cpu_out.astype(np.float32, copy=False)
        gpu_cmp = gpu_out.astype(np.float32, copy=False)
        if not np.allclose(cpu_cmp, gpu_cmp, rtol=0, atol=0):
            raise AssertionError(f"[sanity] {name} mismatch (gt)")
        return

    # For float ops, tolerate tiny diffs.
    atol = 1e-6 if dtype == np.float32 else 1e-12
    rtol = 1e-5 if dtype == np.float32 else 1e-10
    if not np.allclose(cpu_out, gpu_out, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(cpu_out - gpu_out)))
        raise AssertionError(f"[sanity] {name} mismatch: max_abs={max_abs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[512, 512],
        help="Tensor shape, e.g. --shape 512 512",
    )
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument(
        "--sync_each_iter",
        action="store_true",
        help="cudaDeviceSynchronize after each iter",
    )
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument(
        "--ops",
        nargs="*",
        default=[
            "add",
            "sub",
            "mul",
            "div",
            "gt",
            "neg",
            "exp",
            "add_scalar",
            "mul_scalar",
        ],
    )
    ap.add_argument("--alpha", type=float, default=0.125)
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Run correctness check (CPU vs CUDA) once per op",
    )
    args = ap.parse_args()

    shape = tuple(int(x) for x in args.shape)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    alpha = float(args.alpha)

    print("=" * 96)
    print(
        f"Tensor ops CPU vs CUDA bench | shape={shape} dtype={dtype} warmup={args.warmup} repeats={args.repeats} sync_each_iter={args.sync_each_iter}"
    )
    print("=" * 96)

    # Input data
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal(size=shape).astype(dtype)) * 0.25
    b_np = (rng.standard_normal(size=shape).astype(dtype)) * 0.25

    # CPU tensors
    a_cpu = _make_cpu_tensor_from_numpy(a_np)
    b_cpu = _make_cpu_tensor_from_numpy(b_np)

    # CUDA tensors
    have_cuda = _cuda_available()
    if not have_cuda:
        print("[WARN] CUDA not available; running CPU-only.")
    else:
        a_gpu = _make_cuda_tensor_from_numpy(a_np, device_str=args.cuda_device)
        b_gpu = _make_cuda_tensor_from_numpy(b_np, device_str=args.cuda_device)

    # Build op maps
    cpu_ops = _build_ops(a_cpu, b_cpu, alpha)
    gpu_ops = _build_ops(a_gpu, b_gpu, alpha) if have_cuda else {}

    selected = [
        op for op in args.ops if op in cpu_ops and (not have_cuda or op in gpu_ops)
    ]
    if not selected:
        raise SystemExit(
            "No valid ops selected for this build. Try --ops add sub mul div gt neg"
        )

    results: List[OpResult] = []

    # Run bench
    for name in selected:
        # CPU
        cpu_times = _time_op(
            lambda: cpu_ops[name](),
            warmup=args.warmup,
            repeats=args.repeats,
            sync_each_iter=False,
            is_cuda=False,
        )
        cpu_med, cpu_p95 = _median(cpu_times), _p95(cpu_times)

        # CUDA
        if have_cuda:
            gpu_times = _time_op(
                lambda: gpu_ops[name](),
                warmup=args.warmup,
                repeats=args.repeats,
                sync_each_iter=args.sync_each_iter,
                is_cuda=True,
            )
            gpu_med, gpu_p95 = _median(gpu_times), _p95(gpu_times)
        else:
            gpu_med, gpu_p95 = float("nan"), float("nan")

        # Optional correctness check: run op once on both and compare
        if args.sanity and have_cuda:
            out_cpu = cpu_ops[name]()
            out_gpu = gpu_ops[name]()
            cpu_arr = _to_numpy_cpu(out_cpu)
            gpu_arr = _to_numpy_cuda(out_gpu)
            _sanity_check(cpu_arr, gpu_arr, name=name, dtype=np.dtype(dtype))

        results.append(
            OpResult(
                name=name,
                cpu_med=cpu_med,
                cpu_p95=cpu_p95,
                gpu_med=gpu_med,
                gpu_p95=gpu_p95,
            )
        )

    # Print table
    print("\nResults (median / p95):")
    print("-" * 96)
    hdr = f"{'op':12s} | {'cpu_med':>12s} {'cpu_p95':>12s} | {'gpu_med':>12s} {'gpu_p95':>12s} | {'speedup':>8s}"
    print(hdr)
    print("-" * 96)

    for r in results:
        if have_cuda and (r.gpu_med > 0):
            speedup = r.cpu_med / r.gpu_med
            speedup_s = f"{speedup:7.2f}x"
        else:
            speedup_s = "   n/a"

        # choose ms vs us formatting based on magnitude
        def fmt(sec: float) -> str:
            if sec < 1e-3:
                return _fmt_us(sec)
            return _fmt_ms(sec)

        line = (
            f"{r.name:12s} | {fmt(r.cpu_med):>12s} {fmt(r.cpu_p95):>12s} | "
            f"{fmt(r.gpu_med):>12s} {fmt(r.gpu_p95):>12s} | {speedup_s:>8s}"
        )
        print(line)

    print("-" * 96)
    if args.sanity and have_cuda:
        print("Sanity: PASS (all selected ops)")

    # One extra hint: if user didn’t sync, remind (but don’t block)
    if have_cuda and not args.sync_each_iter:
        print(
            "\nNote: CUDA timings may be optimistic without --sync_each_iter (kernels are async)."
        )


if __name__ == "__main__":
    main()
