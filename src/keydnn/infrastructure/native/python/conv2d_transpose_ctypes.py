"""
ctypes bindings for KeyDNN native ConvTranspose2D (Conv2D Transpose) kernels.

This module provides thin `ctypes` wrappers around the compiled KeyDNN native
shared library (CPU), exposing ConvTranspose2D forward and backward kernels for:

- float32 (C++: float)
- float64 (C++: double)

Native kernel assumptions
-------------------------
- NCHW layout for activations (x, y, grad_out, grad_x)
- IOHW layout for weights (w, grad_w)  <-- NOTE: transpose conv uses (C_in, C_out, K_h, K_w)
- Row-major (C-contiguous) memory

All outputs are written in-place into caller-provided NumPy buffers.

Notes
-----
- The native forward kernel is *additive* (scatter-style): it does y[...] += ...
  Therefore, the caller should typically allocate y as zeros to get pure output.
- The native backward kernel accumulates into grad_x and grad_w. Caller should
  typically pass zero-initialized buffers.
- grad_b is not computed in native; Python side typically does grad_b = sum(grad_out).
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_float, c_int
from typing import Optional

import numpy as np

from ._native_loader import (
    load_keydnn_native,
)  # dynamic load the shared library, do not remove this line


def conv2d_transpose_forward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> None:
    """
    Execute the native ConvTranspose2D forward kernel (float32, CPU) via ctypes.

    Requirements
    ------------
    - x: float32 contiguous, shape (N, C_in, H_in, W_in)
    - w: float32 contiguous, shape (C_in, C_out, K_h, K_w)   [IOHW]
    - b: Optional[float32 contiguous], shape (C_out,) or None
    - y: float32 contiguous, shape (N, C_out, H_out, W_out)

    Notes
    -----
    - The kernel performs scatter-style accumulation: y[...] += x * w
      so `y` should usually be zero-initialized before calling.
    """
    if x.dtype != np.float32:
        raise TypeError(f"x must be float32, got {x.dtype}")
    if w.dtype != np.float32:
        raise TypeError(f"w must be float32, got {w.dtype}")
    if y.dtype != np.float32:
        raise TypeError(f"y must be float32, got {y.dtype}")
    if b is not None and b.dtype != np.float32:
        raise TypeError(f"b must be float32 if provided, got {b.dtype}")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if b is not None and not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_transpose_forward_f32
    fn.argtypes = [
        POINTER(c_float),  # x
        POINTER(c_float),  # w
        POINTER(c_float),  # b (nullable)
        POINTER(c_float),  # y
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_in, W_in
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
        c_int,
        c_int,  # pad_h, pad_w
    ]
    fn.restype = None

    x_ptr = x.ctypes.data_as(POINTER(c_float))
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
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
        pad_h,
        pad_w,
    )


def conv2d_transpose_forward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    y: np.ndarray,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> None:
    """
    Execute the native ConvTranspose2D forward kernel (float64, CPU) via ctypes.

    Requirements
    ------------
    - x: float64 contiguous, shape (N, C_in, H_in, W_in)
    - w: float64 contiguous, shape (C_in, C_out, K_h, K_w)   [IOHW]
    - b: Optional[float64 contiguous], shape (C_out,) or None
    - y: float64 contiguous, shape (N, C_out, H_out, W_out)

    Notes
    -----
    - The kernel performs scatter-style accumulation: y[...] += x * w
      so `y` should usually be zero-initialized before calling.
    """
    if x.dtype != np.float64:
        raise TypeError(f"x must be float64, got {x.dtype}")
    if w.dtype != np.float64:
        raise TypeError(f"w must be float64, got {w.dtype}")
    if y.dtype != np.float64:
        raise TypeError(f"y must be float64, got {y.dtype}")
    if b is not None and b.dtype != np.float64:
        raise TypeError(f"b must be float64 if provided, got {b.dtype}")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if b is not None and not b.flags["C_CONTIGUOUS"]:
        b = np.ascontiguousarray(b)

    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_transpose_forward_f64
    fn.argtypes = [
        POINTER(c_double),  # x
        POINTER(c_double),  # w
        POINTER(c_double),  # b (nullable)
        POINTER(c_double),  # y
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_in, W_in
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
        c_int,
        c_int,  # pad_h, pad_w
    ]
    fn.restype = None

    x_ptr = x.ctypes.data_as(POINTER(c_double))
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
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
        pad_h,
        pad_w,
    )


def conv2d_transpose_backward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    grad_x: np.ndarray,
    grad_w: np.ndarray,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> None:
    """
    Execute the native ConvTranspose2D backward kernel (float32, CPU) via ctypes.

    Requirements
    ------------
    - x: float32 contiguous, shape (N, C_in, H_in, W_in)
    - w: float32 contiguous, shape (C_in, C_out, K_h, K_w)   [IOHW]
    - grad_out: float32 contiguous, shape (N, C_out, H_out, W_out)
    - grad_x: float32 contiguous, shape (N, C_in, H_in, W_in) (zero-initialized)
    - grad_w: float32 contiguous, shape (C_in, C_out, K_h, K_w) (zero-initialized)
    """
    if x.dtype != np.float32:
        raise TypeError(f"x must be float32, got {x.dtype}")
    if w.dtype != np.float32:
        raise TypeError(f"w must be float32, got {w.dtype}")
    if grad_out.dtype != np.float32:
        raise TypeError(f"grad_out must be float32, got {grad_out.dtype}")
    if grad_x.dtype != np.float32:
        raise TypeError(f"grad_x must be float32, got {grad_x.dtype}")
    if grad_w.dtype != np.float32:
        raise TypeError(f"grad_w must be float32, got {grad_w.dtype}")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    if not grad_x.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x must be C-contiguous (allocate it contiguously)")
    if not grad_w.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_w must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_transpose_backward_f32
    fn.argtypes = [
        POINTER(c_float),  # x
        POINTER(c_float),  # w
        POINTER(c_float),  # grad_out
        POINTER(c_float),  # grad_x
        POINTER(c_float),  # grad_w
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_in, W_in
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
        c_int,
        c_int,  # pad_h, pad_w
    ]
    fn.restype = None

    fn(
        x.ctypes.data_as(POINTER(c_float)),
        w.ctypes.data_as(POINTER(c_float)),
        grad_out.ctypes.data_as(POINTER(c_float)),
        grad_x.ctypes.data_as(POINTER(c_float)),
        grad_w.ctypes.data_as(POINTER(c_float)),
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
        pad_h,
        pad_w,
    )


def conv2d_transpose_backward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    grad_x: np.ndarray,
    grad_w: np.ndarray,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> None:
    """
    Execute the native ConvTranspose2D backward kernel (float64, CPU) via ctypes.

    Requirements
    ------------
    - x: float64 contiguous, shape (N, C_in, H_in, W_in)
    - w: float64 contiguous, shape (C_in, C_out, K_h, K_w)   [IOHW]
    - grad_out: float64 contiguous, shape (N, C_out, H_out, W_out)
    - grad_x: float64 contiguous, shape (N, C_in, H_in, W_in) (zero-initialized)
    - grad_w: float64 contiguous, shape (C_in, C_out, K_h, K_w) (zero-initialized)
    """
    if x.dtype != np.float64:
        raise TypeError(f"x must be float64, got {x.dtype}")
    if w.dtype != np.float64:
        raise TypeError(f"w must be float64, got {w.dtype}")
    if grad_out.dtype != np.float64:
        raise TypeError(f"grad_out must be float64, got {grad_out.dtype}")
    if grad_x.dtype != np.float64:
        raise TypeError(f"grad_x must be float64, got {grad_x.dtype}")
    if grad_w.dtype != np.float64:
        raise TypeError(f"grad_w must be float64, got {grad_w.dtype}")

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    if not w.flags["C_CONTIGUOUS"]:
        w = np.ascontiguousarray(w)
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    if not grad_x.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x must be C-contiguous (allocate it contiguously)")
    if not grad_w.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_w must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_conv2d_transpose_backward_f64
    fn.argtypes = [
        POINTER(c_double),  # x
        POINTER(c_double),  # w
        POINTER(c_double),  # grad_out
        POINTER(c_double),  # grad_x
        POINTER(c_double),  # grad_w
        c_int,
        c_int,
        c_int,
        c_int,  # N, C_in, H_in, W_in
        c_int,
        c_int,
        c_int,  # C_out, H_out, W_out
        c_int,
        c_int,  # K_h, K_w
        c_int,
        c_int,  # s_h, s_w
        c_int,
        c_int,  # pad_h, pad_w
    ]
    fn.restype = None

    fn(
        x.ctypes.data_as(POINTER(c_double)),
        w.ctypes.data_as(POINTER(c_double)),
        grad_out.ctypes.data_as(POINTER(c_double)),
        grad_x.ctypes.data_as(POINTER(c_double)),
        grad_w.ctypes.data_as(POINTER(c_double)),
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        s_h,
        s_w,
        pad_h,
        pad_w,
    )
