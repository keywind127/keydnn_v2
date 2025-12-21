"""
ctypes bindings for KeyDNN native Conv2D kernels.

This module provides thin `ctypes` wrappers around the compiled KeyDNN native
shared library (CPU), exposing Conv2D forward and backward kernels for:

- float32 (C++: float)
- float64 (C++: double)

The native kernels assume:
- NCHW layout for activations (x_pad, y, grad_out, grad_x_pad)
- OIHW layout for weights (w, grad_w)
- Row-major (C-contiguous) memory

All outputs are written in-place into caller-provided NumPy buffers.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_float, c_double, POINTER
from typing import Optional

import numpy as np


from ._native_loader import (
    load_keydnn_native,
)  # dynamic load the shared library, do not remove this line


def conv2d_forward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
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
) -> None:
    """
    Execute the native Conv2D forward kernel (float32, CPU) via ctypes.

    Requirements
    ------------
    - x_pad: float32 contiguous, shape (N, C_in, H_pad, W_pad)
    - w: float32 contiguous, shape (C_out, C_in, K_h, K_w)
    - b: Optional[float32 contiguous], shape (C_out,) or None
    - y: float32 contiguous, shape (N, C_out, H_out, W_out)

    Notes
    -----
    - x_pad must already be padded by the caller.
    - Results are written in-place to y.
    """
    if x_pad.dtype != np.float32:
        raise TypeError(f"x_pad must be float32, got {x_pad.dtype}")
    if w.dtype != np.float32:
        raise TypeError(f"w must be float32, got {w.dtype}")
    if y.dtype != np.float32:
        raise TypeError(f"y must be float32, got {y.dtype}")
    if b is not None and b.dtype != np.float32:
        raise TypeError(f"b must be float32 if provided, got {b.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if b is not None and not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_forward_f32
    fn.argtypes = [
        POINTER(c_float),  # x_pad
        POINTER(c_float),  # w
        POINTER(c_float),  # b (nullable)
        POINTER(c_float),  # y
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_pad, W_pad
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    x_ptr = x_pad.ctypes.data_as(POINTER(c_float))
    w_ptr = w.ctypes.data_as(POINTER(c_float))
    y_ptr = y.ctypes.data_as(POINTER(c_float))

    if b is None:
        b_ptr = ctypes.cast(0, POINTER(c_float))
    else:
        b_ptr = b.ctypes.data_as(POINTER(c_float))

    fn(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        N,
        C_in,
        H_pad,
        W_pad,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
    )


def conv2d_forward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
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
) -> None:
    """
    Execute the native Conv2D forward kernel (float64, CPU) via ctypes.

    Requirements
    ------------
    - x_pad: float64 contiguous, shape (N, C_in, H_pad, W_pad)
    - w: float64 contiguous, shape (C_out, C_in, K_h, K_w)
    - b: Optional[float64 contiguous], shape (C_out,) or None
    - y: float64 contiguous, shape (N, C_out, H_out, W_out)
    """
    if x_pad.dtype != np.float64:
        raise TypeError(f"x_pad must be float64, got {x_pad.dtype}")
    if w.dtype != np.float64:
        raise TypeError(f"w must be float64, got {w.dtype}")
    if y.dtype != np.float64:
        raise TypeError(f"y must be float64, got {y.dtype}")
    if b is not None and b.dtype != np.float64:
        raise TypeError(f"b must be float64 if provided, got {b.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if b is not None and not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_forward_f64
    fn.argtypes = [
        POINTER(c_double),  # x_pad
        POINTER(c_double),  # w
        POINTER(c_double),  # b (nullable)
        POINTER(c_double),  # y
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_pad, W_pad
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    x_ptr = x_pad.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))

    if b is None:
        b_ptr = ctypes.cast(0, POINTER(c_double))
    else:
        b_ptr = b.ctypes.data_as(POINTER(c_double))

    fn(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        N,
        C_in,
        H_pad,
        W_pad,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
    )


def conv2d_backward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    grad_x_pad: np.ndarray,
    grad_w: np.ndarray,
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
) -> None:
    """
    Execute the native Conv2D backward kernel (float32, CPU) via ctypes.

    Requirements
    ------------
    - x_pad: float32 contiguous, shape (N, C_in, H_pad, W_pad)
    - w: float32 contiguous, shape (C_out, C_in, K_h, K_w)
    - grad_out: float32 contiguous, shape (N, C_out, H_out, W_out)
    - grad_x_pad: float32 contiguous, shape (N, C_in, H_pad, W_pad) (zero-initialized)
    - grad_w: float32 contiguous, shape (C_out, C_in, K_h, K_w) (zero-initialized)
    """
    if x_pad.dtype != np.float32:
        raise TypeError(f"x_pad must be float32, got {x_pad.dtype}")
    if w.dtype != np.float32:
        raise TypeError(f"w must be float32, got {w.dtype}")
    if grad_out.dtype != np.float32:
        raise TypeError(f"grad_out must be float32, got {grad_out.dtype}")
    if grad_x_pad.dtype != np.float32:
        raise TypeError(f"grad_x_pad must be float32, got {grad_x_pad.dtype}")
    if grad_w.dtype != np.float32:
        raise TypeError(f"grad_w must be float32, got {grad_w.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous (allocate it contiguously)")
    if not grad_w.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_w must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_backward_f32
    fn.argtypes = [
        POINTER(c_float),  # x_pad
        POINTER(c_float),  # w
        POINTER(c_float),  # grad_out
        POINTER(c_float),  # grad_x_pad
        POINTER(c_float),  # grad_w
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_pad, W_pad
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        x_pad.ctypes.data_as(POINTER(c_float)),
        w.ctypes.data_as(POINTER(c_float)),
        grad_out.ctypes.data_as(POINTER(c_float)),
        grad_x_pad.ctypes.data_as(POINTER(c_float)),
        grad_w.ctypes.data_as(POINTER(c_float)),
        N,
        C_in,
        H_pad,
        W_pad,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
    )


def conv2d_backward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    grad_x_pad: np.ndarray,
    grad_w: np.ndarray,
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
) -> None:
    """
    Execute the native Conv2D backward kernel (float64, CPU) via ctypes.

    Requirements
    ------------
    - x_pad: float64 contiguous, shape (N, C_in, H_pad, W_pad)
    - w: float64 contiguous, shape (C_out, C_in, K_h, K_w)
    - grad_out: float64 contiguous, shape (N, C_out, H_out, W_out)
    - grad_x_pad: float64 contiguous, shape (N, C_in, H_pad, W_pad) (zero-initialized)
    - grad_w: float64 contiguous, shape (C_out, C_in, K_h, K_w) (zero-initialized)
    """
    if x_pad.dtype != np.float64:
        raise TypeError(f"x_pad must be float64, got {x_pad.dtype}")
    if w.dtype != np.float64:
        raise TypeError(f"w must be float64, got {w.dtype}")
    if grad_out.dtype != np.float64:
        raise TypeError(f"grad_out must be float64, got {grad_out.dtype}")
    if grad_x_pad.dtype != np.float64:
        raise TypeError(f"grad_x_pad must be float64, got {grad_x_pad.dtype}")
    if grad_w.dtype != np.float64:
        raise TypeError(f"grad_w must be float64, got {grad_w.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous (allocate it contiguously)")
    if not grad_w.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_w must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_backward_f64
    fn.argtypes = [
        POINTER(c_double),  # x_pad
        POINTER(c_double),  # w
        POINTER(c_double),  # grad_out
        POINTER(c_double),  # grad_x_pad
        POINTER(c_double),  # grad_w
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_pad, W_pad
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        x_pad.ctypes.data_as(POINTER(c_double)),
        w.ctypes.data_as(POINTER(c_double)),
        grad_out.ctypes.data_as(POINTER(c_double)),
        grad_x_pad.ctypes.data_as(POINTER(c_double)),
        grad_w.ctypes.data_as(POINTER(c_double)),
        N,
        C_in,
        H_pad,
        W_pad,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
    )
