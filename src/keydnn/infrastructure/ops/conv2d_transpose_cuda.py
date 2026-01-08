"""
infrastructure/ops/conv2d_transpose_cuda.py

Ops-layer CUDA ConvTranspose2D (transpose convolution) wrapper for KeyDNN.

This module provides a NumPy-friendly interface for running KeyDNN's CUDA
ConvTranspose2D kernels via `ctypes`. It mirrors the CPU conv2d_transpose ops
API (validate inputs, compute output shapes, return NumPy outputs) while
delegating compute to native CUDA kernels.

Native contract
---------------
- Input/Output layout:
  - x, y, grad_out, grad_x: NCHW  (N, C, H, W)
  - w, grad_w:             IOHW  (C_in, C_out, K_h, K_w)
- Dtypes: float32, float64 only.
- Output size is computed on Python side:
    H_out = (H_in - 1) * s_h - 2 * pad_h + K_h + out_pad_h
    W_out = (W_in - 1) * s_w - 2 * pad_w + K_w + out_pad_w

Notes
-----
- Current CUDA kernel interface does NOT take output_padding directly; it is
  handled by choosing H_out/W_out on the caller side.
- grad_b is computed on CPU: grad_b = sum(grad_out) over (N, H_out, W_out)
- This wrapper allocates/free temporary device buffers and performs H2D/D2H
  copies using existing CUDA memory helpers.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional, Tuple

import numpy as np

from ..native_cuda.python.avgpool2d_ctypes import (
    cuda_malloc,
    cuda_free,
    cudaMemcpyHtoD,
    cudaMemcpyDtoH,
    cuda_synchronize,
    cuda_set_device,
)

from ..native_cuda.python.ops.conv2d_transpose_ctypes import (
    conv2d_transpose_forward_cuda as _conv2d_t_forward_ctypes,
    conv2d_transpose_backward_cuda as _conv2d_t_backward_ctypes,
)


def _is_cdll(obj: object) -> bool:
    return isinstance(obj, ctypes.CDLL)


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _dtype_itemsize(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(
            f"conv2d_transpose_cuda supports float32/float64 only, got {dt}"
        )
    return int(dt.itemsize)


def _probe_dev_range(lib: ctypes.CDLL, base_dev: int, nbytes_required: int) -> None:
    """
    Best-effort device range probe (optional).

    Mirrors conv2d_cuda.py behavior: if `keydnn_cuda_memcpy_d2h` exists, probe the
    last byte in the requested range to catch undersized/invalid allocations.
    """
    if nbytes_required <= 0:
        return
    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        return

    fn = lib.keydnn_cuda_memcpy_d2h
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    fn.restype = ctypes.c_int

    tmp = (ctypes.c_ubyte * 1)()
    last_addr = int(base_dev) + int(nbytes_required) - 1
    st = int(
        fn(
            ctypes.cast(tmp, ctypes.c_void_p),
            ctypes.c_uint64(last_addr),
            ctypes.c_size_t(1),
        )
    )
    if st != 0:
        raise RuntimeError(
            f"device buffer too small or invalid: probe failed at 0x{last_addr:x} (status={st})"
        )


def conv2d_transpose_forward_cuda(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Run CUDA ConvTranspose2D forward using NumPy inputs (NCHW / IOHW).

    Expected usage
    --------------
    y = conv2d_transpose_forward_cuda(
        lib,
        x=x,
        w=w,
        b=b,                    # optional
        stride=1,               # int or (int,int)
        padding=0,              # int or (int,int)
        output_padding=0,       # int or (int,int)
        device_index=0,         # optional (alias: device)
        sync=True,              # optional
        dtype=None,             # optional, inferred from x
    )
    """
    if not args:
        raise TypeError(
            "conv2d_transpose_forward_cuda expected at least a lib argument"
        )

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "conv2d_transpose_forward_cuda expected ctypes.CDLL as first arg "
                "(or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]
    if rest:
        raise TypeError(
            "conv2d_transpose_forward_cuda does not accept positional args beyond lib"
        )

    x = kwargs.pop("x", None)
    w = kwargs.pop("w", None)
    b = kwargs.pop("b", None)

    stride = kwargs.pop("stride", 1)
    padding = kwargs.pop("padding", 0)
    output_padding = kwargs.pop("output_padding", 0)

    dtype = kwargs.pop("dtype", None)
    sync = kwargs.pop("sync", True)

    device_index = kwargs.pop("device_index", None)
    if device_index is None:
        device_index = kwargs.pop("device", None)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"conv2d_transpose_forward_cuda got unexpected kwargs: {extra}")

    if x is None or w is None:
        raise TypeError("conv2d_transpose_forward_cuda requires x and w")
    if not isinstance(x, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("x and w must be numpy arrays")

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    if s_h <= 0 or s_w <= 0:
        raise ValueError(f"stride must be positive, got stride=({s_h},{s_w})")
    if op_h < 0 or op_w < 0:
        raise ValueError(f"output_padding must be non-negative, got ({op_h},{op_w})")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"output_padding must be < stride per dim, got output_padding=({op_h},{op_w}), "
            f"stride=({s_h},{s_w})"
        )

    if dtype is None:
        dtype = x.dtype
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"conv2d_transpose_forward_cuda supports float32/float64 only, got {dtype}"
        )

    x = x.astype(dtype, copy=False)
    w = w.astype(dtype, copy=False)

    b_arr: Optional[np.ndarray]
    if b is None:
        b_arr = None
    else:
        if not isinstance(b, np.ndarray):
            raise TypeError("b must be a numpy array or None")
        b_arr = b.astype(dtype, copy=False)

    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {x.shape}")
    if w.ndim != 4:
        raise ValueError(f"w must be 4D IOHW, got shape {w.shape}")

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")
    if b_arr is not None:
        if b_arr.ndim != 1 or b_arr.shape[0] != C_out:
            raise ValueError(
                f"bias shape mismatch: expected ({C_out},), got {b_arr.shape}"
            )

    # Output sizing (caller-side; native kernel just bounds-checks)
    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if H_out < 0 or W_out < 0:
        raise ValueError(f"invalid output size: H_out={H_out}, W_out={W_out}")

    y = np.empty((N, C_out, H_out, W_out), dtype=dtype)

    # Set device (optional)
    if device_index is not None:
        cuda_set_device(lib, int(device_index))

    # Allocate device buffers
    nbytes_x = int(x.nbytes)
    nbytes_w = int(w.nbytes)
    nbytes_b = int(0 if b_arr is None else b_arr.nbytes)
    nbytes_y = int(y.nbytes)

    x_dev = int(cuda_malloc(lib, nbytes_x if nbytes_x > 0 else 1))
    w_dev = int(cuda_malloc(lib, nbytes_w if nbytes_w > 0 else 1))
    y_dev = int(cuda_malloc(lib, nbytes_y if nbytes_y > 0 else 1))
    b_dev = 0

    try:
        if nbytes_x > 0:
            cudaMemcpyHtoD(lib, x_dev, x, nbytes_x)
        if nbytes_w > 0:
            cudaMemcpyHtoD(lib, w_dev, w, nbytes_w)

        if b_arr is not None:
            b_dev = int(cuda_malloc(lib, nbytes_b if nbytes_b > 0 else 1))
            if nbytes_b > 0:
                cudaMemcpyHtoD(lib, b_dev, b_arr, nbytes_b)

        # Optional probes
        itemsize = _dtype_itemsize(dtype)
        _probe_dev_range(lib, x_dev, N * C_in * max(H_in, 0) * max(W_in, 0) * itemsize)
        _probe_dev_range(lib, w_dev, C_in * C_out * K_h * K_w * itemsize)
        _probe_dev_range(
            lib, y_dev, N * C_out * max(H_out, 0) * max(W_out, 0) * itemsize
        )
        if b_arr is not None:
            _probe_dev_range(lib, b_dev, C_out * itemsize)

        _conv2d_t_forward_ctypes(
            lib,
            x_dev=x_dev,
            w_dev=w_dev,
            b_dev=(b_dev if b_arr is not None else None),
            y_dev=y_dev,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=p_h,
            pad_w=p_w,
            dtype=dtype,
        )

        if sync:
            cuda_synchronize(lib)

        if nbytes_y > 0:
            cudaMemcpyDtoH(lib, y, y_dev, nbytes_y)
        return y

    finally:
        cuda_free(lib, x_dev)
        cuda_free(lib, w_dev)
        cuda_free(lib, y_dev)
        if b_dev:
            cuda_free(lib, b_dev)


def conv2d_transpose_backward_cuda(
    *args: Any, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Run CUDA ConvTranspose2D backward and return (grad_x, grad_w, grad_b).

    Expected usage
    --------------
    grad_x, grad_w, grad_b = conv2d_transpose_backward_cuda(
        lib,
        x=x,
        w=w,
        b=b,                    # optional (controls whether grad_b is returned)
        grad_out=grad_out,
        stride=1,
        padding=0,
        output_padding=0,
        device_index=0,         # optional (alias: device)
        sync=True,              # optional
        dtype=None,             # optional, inferred from x
    )
    """
    if not args:
        raise TypeError(
            "conv2d_transpose_backward_cuda expected at least a lib argument"
        )

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "conv2d_transpose_backward_cuda expected ctypes.CDLL as first arg "
                "(or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]
    if rest:
        raise TypeError(
            "conv2d_transpose_backward_cuda does not accept positional args beyond lib"
        )

    x = kwargs.pop("x", None)
    w = kwargs.pop("w", None)
    b = kwargs.pop("b", None)
    grad_out = kwargs.pop("grad_out", None)

    stride = kwargs.pop("stride", 1)
    padding = kwargs.pop("padding", 0)
    output_padding = kwargs.pop("output_padding", 0)

    dtype = kwargs.pop("dtype", None)
    sync = kwargs.pop("sync", True)

    device_index = kwargs.pop("device_index", None)
    if device_index is None:
        device_index = kwargs.pop("device", None)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(
            f"conv2d_transpose_backward_cuda got unexpected kwargs: {extra}"
        )

    if x is None or w is None or grad_out is None:
        raise TypeError("conv2d_transpose_backward_cuda requires x, w, grad_out")
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(w, np.ndarray)
        or not isinstance(grad_out, np.ndarray)
    ):
        raise TypeError("x, w, grad_out must be numpy arrays")

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    if s_h <= 0 or s_w <= 0:
        raise ValueError(f"stride must be positive, got stride=({s_h},{s_w})")
    if op_h < 0 or op_w < 0:
        raise ValueError(f"output_padding must be non-negative, got ({op_h},{op_w})")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"output_padding must be < stride per dim, got output_padding=({op_h},{op_w}), "
            f"stride=({s_h},{s_w})"
        )

    if dtype is None:
        dtype = x.dtype
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"conv2d_transpose_backward_cuda supports float32/float64 only, got {dtype}"
        )

    x = x.astype(dtype, copy=False)
    w = w.astype(dtype, copy=False)
    grad_out = grad_out.astype(dtype, copy=False)

    b_arr: Optional[np.ndarray]
    if b is None:
        b_arr = None
    else:
        if not isinstance(b, np.ndarray):
            raise TypeError("b must be a numpy array or None")
        b_arr = b.astype(dtype, copy=False)

    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {x.shape}")
    if w.ndim != 4:
        raise ValueError(f"w must be 4D IOHW, got shape {w.shape}")
    if grad_out.ndim != 4:
        raise ValueError(f"grad_out must be 4D NCHW, got shape {grad_out.shape}")

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w

    if grad_out.shape != (N, C_out, H_out, W_out):
        raise ValueError(
            f"grad_out shape mismatch: expected {(N, C_out, H_out, W_out)}, got {grad_out.shape}"
        )

    # grad_b on CPU (matches CPU semantics)
    grad_b = None
    if b_arr is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(dtype, copy=False)

    grad_x = np.empty_like(x)
    grad_w = np.empty_like(w)

    # Set device (optional)
    if device_index is not None:
        cuda_set_device(lib, int(device_index))

    # Allocate device buffers
    nbytes_x = int(x.nbytes)
    nbytes_w = int(w.nbytes)
    nbytes_go = int(grad_out.nbytes)
    nbytes_gx = int(grad_x.nbytes)
    nbytes_gw = int(grad_w.nbytes)

    x_dev = int(cuda_malloc(lib, nbytes_x if nbytes_x > 0 else 1))
    w_dev = int(cuda_malloc(lib, nbytes_w if nbytes_w > 0 else 1))
    go_dev = int(cuda_malloc(lib, nbytes_go if nbytes_go > 0 else 1))
    gx_dev = int(cuda_malloc(lib, nbytes_gx if nbytes_gx > 0 else 1))
    gw_dev = int(cuda_malloc(lib, nbytes_gw if nbytes_gw > 0 else 1))

    try:
        if nbytes_x > 0:
            cudaMemcpyHtoD(lib, x_dev, x, nbytes_x)
        if nbytes_w > 0:
            cudaMemcpyHtoD(lib, w_dev, w, nbytes_w)
        if nbytes_go > 0:
            cudaMemcpyHtoD(lib, go_dev, grad_out, nbytes_go)

        itemsize = _dtype_itemsize(dtype)
        _probe_dev_range(lib, x_dev, N * C_in * max(H_in, 0) * max(W_in, 0) * itemsize)
        _probe_dev_range(lib, w_dev, C_in * C_out * K_h * K_w * itemsize)
        _probe_dev_range(
            lib, go_dev, N * C_out * max(H_out, 0) * max(W_out, 0) * itemsize
        )
        _probe_dev_range(lib, gx_dev, N * C_in * max(H_in, 0) * max(W_in, 0) * itemsize)
        _probe_dev_range(lib, gw_dev, C_in * C_out * K_h * K_w * itemsize)

        _conv2d_t_backward_ctypes(
            lib,
            x_dev=x_dev,
            w_dev=w_dev,
            grad_out_dev=go_dev,
            grad_x_dev=gx_dev,
            grad_w_dev=gw_dev,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=p_h,
            pad_w=p_w,
            dtype=dtype,
        )

        if sync:
            cuda_synchronize(lib)

        if nbytes_gx > 0:
            cudaMemcpyDtoH(lib, grad_x, gx_dev, nbytes_gx)
        if nbytes_gw > 0:
            cudaMemcpyDtoH(lib, grad_w, gw_dev, nbytes_gw)

        return grad_x, grad_w, grad_b

    finally:
        cuda_free(lib, x_dev)
        cuda_free(lib, w_dev)
        cuda_free(lib, go_dev)
        cuda_free(lib, gx_dev)
        cuda_free(lib, gw_dev)


class _Conv2dTransposeCudaAliases:
    """
    Public alias names exposed by this ops-layer module (documentation only).
    """

    conv2d_transpose_cuda = conv2d_transpose_forward_cuda
    conv2d_transpose_forward = conv2d_transpose_forward_cuda
    conv2d_transpose_backward = conv2d_transpose_backward_cuda


conv2d_transpose_cuda = conv2d_transpose_forward_cuda
conv2d_transpose_forward = conv2d_transpose_forward_cuda
conv2d_transpose_backward = conv2d_transpose_backward_cuda

__all__ = [
    "conv2d_transpose_forward_cuda",
    "conv2d_transpose_backward_cuda",
    "conv2d_transpose_cuda",
    "conv2d_transpose_forward",
    "conv2d_transpose_backward",
]
