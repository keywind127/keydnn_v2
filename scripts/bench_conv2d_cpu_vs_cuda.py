"""
scripts/bench_conv2d_cpu_vs_cuda.py

CPU vs CUDA Conv2D microbenchmark (NOT a unit test) for KeyDNN.

Benchmarks forward-only computation for:
- conv2d_forward_{cpu,cuda}_tensor(x, w, b)

Timing policy
-------------
- Excludes HtoD/DtoH transfers for *inputs/params* (performed once per case).
- Times only CPU forward or CUDA forward.
- Uses warmup iterations before timed repeats.
- Measures *forward only* (no backward).

IMPORTANT (current CUDA boundary design)
---------------------------------------
If your `conv2d_cuda_ext` implementation still performs D2H/H2D internally
(because the ops wrapper is NumPy-facing), then the CUDA timings include
those host copies. This script still gives you an apples-to-apples "what your
current API costs" comparison, but it will understate the true kernel-only speedup.

Usage
-----
python scripts/bench_conv2d_cpu_vs_cuda.py --presets --dtype float32 --sanity
python scripts/bench_conv2d_cpu_vs_cuda.py --N 32 --Cin 64 --Cout 128 --H 56 --W 56 --Kh 3 --Kw 3 --stride 1 --padding 1 --dtype float32
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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

from keydnn.infrastructure.ops.conv2d_cpu import _pair as _pair_cpu
from keydnn.infrastructure.ops.conv2d_cpu_ext import conv2d_forward_cpu_tensor
from keydnn.infrastructure.ops.conv2d_cuda_ext import conv2d_forward_cuda_tensor

# Optional CUDA imports (for device init)
try:
    from keydnn.infrastructure.ops.pool2d_cuda import _load_cuda_lib, cuda_set_device

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
    dt = np.dtype(dtype)
    if dt == np.float32:
        return "float32"
    if dt == np.float64:
        return "float64"
    return str(dt)


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


def _make_tensor_from_numpy_cpu(
    x: np.ndarray, *, requires_grad: bool = False
) -> Tensor:
    dev = Device("cpu")
    x_c = np.ascontiguousarray(x)
    if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
        return Tensor.from_numpy(x_c, device=dev, requires_grad=requires_grad)  # type: ignore[call-arg]
    t = Tensor(shape=x_c.shape, device=dev, requires_grad=requires_grad, ctx=None, dtype=np.dtype(x_c.dtype))  # type: ignore[call-arg]
    t.copy_from_numpy(x_c)
    return t


def _make_tensor_from_numpy_cuda(
    x: np.ndarray,
    *,
    cuda_device: Device,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a CUDA Tensor and copy host -> device using Tensor APIs already present in your repo.
    Prefer Tensor.from_numpy(..., device=cuda). Fallback: allocate + cudaMemcpyHtoD.
    """
    x_c = np.ascontiguousarray(x)

    if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
        return Tensor.from_numpy(x_c, device=cuda_device, requires_grad=requires_grad)  # type: ignore[call-arg]

    t = Tensor(
        shape=x_c.shape,
        device=cuda_device,
        requires_grad=requires_grad,
        ctx=None,
        dtype=np.dtype(x_c.dtype),
    )
    if hasattr(t, "_ensure_cuda_alloc") and callable(getattr(t, "_ensure_cuda_alloc")):
        t._ensure_cuda_alloc(dtype=np.dtype(x_c.dtype))  # type: ignore[attr-defined]

    dst = int(getattr(t, "data", 0))
    if dst == 0 and int(x_c.nbytes) != 0:
        raise RuntimeError("CUDA tensor allocation failed: t.data == 0")

    # Import cudaMemcpyHtoD from any known ctypes module path
    def _import_cudaMemcpyHtoD():
        for mod_path in (
            "keydnn.infrastructure.native_cuda.python.ops.memcpy_ctypes",
            "keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
        ):
            try:
                m = __import__(mod_path, fromlist=["cudaMemcpyHtoD", "memcpy_htod"])
                if hasattr(m, "memcpy_htod"):
                    return ("wrapper", m.memcpy_htod)
                if hasattr(m, "cudaMemcpyHtoD"):
                    return ("raw", m.cudaMemcpyHtoD)
            except Exception:
                continue
        raise ImportError(
            "Could not import cudaMemcpyHtoD/memcpy_htod from known native_cuda ctypes modules."
        )

    kind, fn = _import_cudaMemcpyHtoD()

    if hasattr(t, "_get_cuda_lib") and callable(getattr(t, "_get_cuda_lib")):
        lib = t._get_cuda_lib()
    else:
        lib = Tensor._get_cuda_lib()

    if int(x_c.nbytes) > 0:
        if kind == "wrapper":
            fn(lib, dst_dev=int(dst), src_host=x_c, nbytes=int(x_c.nbytes), sync=True)  # type: ignore[misc]
        else:
            fn(lib, int(dst), x_c, int(x_c.nbytes))  # type: ignore[misc]

    return t


@dataclass(frozen=True)
class Case:
    name: str
    N: int
    Cin: int
    Cout: int
    H: int
    W: int
    Kh: int
    Kw: int
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    bias: bool


def _make_case_arrays(
    rng: np.random.Generator, c: Case, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = rng.standard_normal((c.N, c.Cin, c.H, c.W)).astype(dtype, copy=False)
    w = rng.standard_normal((c.Cout, c.Cin, c.Kh, c.Kw)).astype(dtype, copy=False)
    b = rng.standard_normal((c.Cout,)).astype(dtype, copy=False) if c.bias else None
    return x, w, b


def bench_case(
    c: Case,
    *,
    dtype: np.dtype,
    warmup: int,
    repeats: int,
    sanity: bool,
    seed: int,
    device_index: int,
) -> None:
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)

    x_np, w_np, b_np = _make_case_arrays(rng, c, dtype)

    # -------------------------
    # CPU setup (outside timing)
    # -------------------------
    x_cpu = _make_tensor_from_numpy_cpu(x_np, requires_grad=False)
    w_cpu = _make_tensor_from_numpy_cpu(w_np, requires_grad=False)
    b_cpu = (
        None if b_np is None else _make_tensor_from_numpy_cpu(b_np, requires_grad=False)
    )

    # -------------------------
    # CUDA setup (outside timing)
    # -------------------------
    if not _CUDA_IMPORT_OK:
        raise RuntimeError(
            "CUDA wrappers not importable. Build/install CUDA native DLL + python wrappers first."
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device_index))
    cuda_dev = _get_cuda_device(device_index)

    x_cuda = _make_tensor_from_numpy_cuda(
        x_np, cuda_device=cuda_dev, requires_grad=False
    )
    w_cuda = _make_tensor_from_numpy_cuda(
        w_np, cuda_device=cuda_dev, requires_grad=False
    )
    b_cuda = (
        None
        if b_np is None
        else _make_tensor_from_numpy_cuda(
            b_np, cuda_device=cuda_dev, requires_grad=False
        )
    )

    stride = tuple(int(v) for v in c.stride)
    padding = tuple(int(v) for v in c.padding)

    # -------------------------
    # Sanity check (not timed)
    # -------------------------
    if sanity:
        y_cpu_t = conv2d_forward_cpu_tensor(
            x_cpu,
            w_cpu,
            b_cpu,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
        )
        y_cuda_t = conv2d_forward_cuda_tensor(
            x_cuda,
            w_cuda,
            b_cuda,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
            device_index=int(device_index),
            sync=True,
        )

        y_cpu = y_cpu_t.to_numpy()
        y_cuda = y_cuda_t.to_numpy()

        if dtype == np.float64:
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-10, atol=1e-10)
        else:
            np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-4, atol=1e-4)

    # -------------------------
    # Timed regions (forward only)
    # -------------------------
    def cpu_fwd() -> None:
        _ = conv2d_forward_cpu_tensor(
            x_cpu,
            w_cpu,
            b_cpu,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
        )

    def cuda_fwd() -> None:
        _ = conv2d_forward_cuda_tensor(
            x_cuda,
            w_cuda,
            b_cuda,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
            device_index=int(device_index),
            sync=True,
        )

    t_cpu = _time_one(cpu_fwd, warmup=warmup, repeats=repeats)
    t_cuda = _time_one(cuda_fwd, warmup=warmup, repeats=repeats)

    cpu_med = _median(t_cpu)
    cuda_med = _median(t_cuda)

    print(
        f"{c.name}: "
        f"N={c.N} Cin={c.Cin} Cout={c.Cout} HxW={c.H}x{c.W} K={c.Kh}x{c.Kw} "
        f"s={stride} p={padding} bias={c.bias} dtype={_dtype_str(dtype)} | "
        f"cpu={_fmt_seconds(cpu_med):>10}  cuda={_fmt_seconds(cuda_med):>10}  "
        f"speedup={_speedup(cpu_med, cuda_med):>7.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Cin", type=int, default=32)
    ap.add_argument("--Cout", type=int, default=64)
    ap.add_argument("--H", type=int, default=56)
    ap.add_argument("--W", type=int, default=56)
    ap.add_argument("--Kh", type=int, default=3)
    ap.add_argument("--Kw", type=int, default=3)

    ap.add_argument(
        "--stride", type=int, nargs="+", default=[1], help="stride (1) or (h w)"
    )
    ap.add_argument(
        "--padding", type=int, nargs="+", default=[1], help="padding (1) or (h w)"
    )

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

    stride = _pair_cpu(
        tuple(args.stride) if len(args.stride) > 1 else int(args.stride[0])
    )
    padding = _pair_cpu(
        tuple(args.padding) if len(args.padding) > 1 else int(args.padding[0])
    )

    if args.presets:
        if args.big:
            cases = [
                Case("resnet-ish-56", 32, 64, 128, 56, 56, 3, 3, (1, 1), (1, 1), True),
                Case("resnet-ish-28", 64, 128, 256, 28, 28, 3, 3, (1, 1), (1, 1), True),
                Case("wide-1x1", 64, 256, 256, 56, 56, 1, 1, (1, 1), (0, 0), True),
            ]
        else:
            cases = [
                Case("small-28", 16, 16, 32, 28, 28, 3, 3, (1, 1), (1, 1), True),
                Case("mid-56", 8, 32, 64, 56, 56, 3, 3, (1, 1), (1, 1), True),
                Case("stride2", 8, 32, 64, 56, 56, 3, 3, (2, 2), (1, 1), True),
                Case("no-bias", 8, 32, 64, 56, 56, 3, 3, (1, 1), (1, 1), False),
            ]

        print("\n" + "=" * 120)
        print(
            f"Conv2D CPU vs CUDA benchmark | dtype={_dtype_str(dtype)} "
            f"(warmup={args.warmup}, repeats={args.repeats}, device={args.device}, sanity={args.sanity})"
        )
        print("=" * 120)

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
        c = Case(
            "single",
            int(args.N),
            int(args.Cin),
            int(args.Cout),
            int(args.H),
            int(args.W),
            int(args.Kh),
            int(args.Kw),
            tuple(int(x) for x in stride),
            tuple(int(x) for x in padding),
            bool(args.bias),
        )

        print("\n" + "=" * 120)
        print(
            f"Conv2D CPU vs CUDA benchmark | dtype={_dtype_str(dtype)} "
            f"(warmup={args.warmup}, repeats={args.repeats}, device={args.device}, sanity={args.sanity})"
        )
        print("=" * 120)

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
