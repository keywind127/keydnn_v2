"""
ctypes bindings for KeyDNN native CPU kernels.

This module provides low-level Python bindings to native C/C++ implementations
of performance-critical tensor operations via `ctypes`. It is part of the
KeyDNN infrastructure layer and is intentionally backend-specific.

Currently, this module exposes:
- A CPU-based MaxPool2D forward kernel for float32 tensors in NCHW layout.

Design notes
------------
- The native shared library is expected to live alongside this Python file.
- All exposed functions assume **contiguous NCHW memory layout**.
- Argument validation is intentionally strict to prevent undefined behavior
  when crossing the Python ↔ native boundary.
- This module is not intended for direct end-user consumption; higher-level
  APIs should wrap these functions with safer abstractions.

Platform notes
--------------
- Windows   : keydnn_native.dll
- macOS     : libkeydnn_native.dylib
- Linux     : libkeydnn_native.so
"""

from __future__ import annotations

import os
import sys
import ctypes
from ctypes import c_int, c_float, c_int64, POINTER
from ctypes import c_double
from typing import Optional

import numpy as np


def _default_lib_name() -> str:
    """
    Return the platform-specific filename of the KeyDNN native shared library.

    Returns
    -------
    str
        Name of the shared library file corresponding to the current platform.
    """
    if sys.platform.startswith("win"):
        return "keydnn_native.dll"
    if sys.platform == "darwin":
        return "libkeydnn_native.dylib"
    return "libkeydnn_native.so"


def load_keydnn_native(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """
    Load the KeyDNN native shared library using `ctypes`.

    By default, the library is loaded from the same directory as this Python
    module. A custom path may be provided to support non-standard build or
    deployment layouts.

    Parameters
    ----------
    lib_path : Optional[str]
        Absolute or relative path to the compiled shared library. If None,
        the loader resolves the library name based on the current platform
        and loads it from the directory containing this file.

    Returns
    -------
    ctypes.CDLL
        Loaded shared library handle.

    Raises
    ------
    OSError
        If the shared library cannot be found or loaded.
    """
    if lib_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(here, _default_lib_name())

    return ctypes.CDLL(lib_path)


def maxpool2d_forward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    argmax_idx: np.ndarray,
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
) -> None:
    """
    Execute the native MaxPool2D forward kernel (float32, CPU) via ctypes.

    This function dispatches to a C++ implementation that computes the forward
    pass of a 2D max pooling operation over an already-padded input tensor.
    Results are written **in-place** into the provided output buffers.

    The native kernel assumes:
    - NCHW tensor layout
    - Row-major (C-contiguous) memory
    - No shape or bounds checking beyond what is validated here

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library handle.
    x_pad : np.ndarray
        Padded input tensor of shape (N, C, H_pad, W_pad), dtype float32.
    y : np.ndarray
        Output tensor of shape (N, C, H_out, W_out), dtype float32.
    argmax_idx : np.ndarray
        Integer tensor of shape (N, C, H_out, W_out), dtype int64.
        Stores the flattened spatial index (h * W_pad + w) of the selected
        maximum within the padded input.
    N : int
        Batch size.
    C : int
        Number of channels.
    H_pad : int
        Height of the padded input.
    W_pad : int
        Width of the padded input.
    H_out : int
        Height of the output feature map.
    W_out : int
        Width of the output feature map.
    k_h : int
        Pooling kernel height.
    k_w : int
        Pooling kernel width.
    s_h : int
        Vertical stride.
    s_w : int
        Horizontal stride.

    Returns
    -------
    None
        Results are written in-place to `y` and `argmax_idx`.

    Raises
    ------
    TypeError
        If input/output arrays have incorrect dtypes.
    ValueError
        If output arrays are not C-contiguous.

    Notes
    -----
    - Padding is assumed to have been applied by the caller (typically using
      `-inf` padding to preserve max semantics).
    - This function performs no allocation and is safe to call in tight loops.
    """
    if x_pad.dtype != np.float32:
        raise TypeError(f"x_pad must be float32, got {x_pad.dtype}")
    if y.dtype != np.float32:
        raise TypeError(f"y must be float32, got {y.dtype}")
    if argmax_idx.dtype != np.int64:
        raise TypeError(f"argmax_idx must be int64, got {argmax_idx.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")
    if not argmax_idx.flags["C_CONTIGUOUS"]:
        raise ValueError("argmax_idx must be C-contiguous (allocate it contiguously)")

    # Bind native function signature
    fn = lib.keydnn_maxpool2d_forward_f32
    fn.argtypes = [
        POINTER(c_float),  # x_pad
        POINTER(c_float),  # y
        POINTER(c_int64),  # argmax_idx
        c_int,
        c_int,
        c_int,
        c_int,  # N, C, H_pad, W_pad
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    x_ptr = x_pad.ctypes.data_as(POINTER(c_float))
    y_ptr = y.ctypes.data_as(POINTER(c_float))
    idx_ptr = argmax_idx.ctypes.data_as(POINTER(c_int64))

    fn(
        x_ptr,
        y_ptr,
        idx_ptr,
        N,
        C,
        H_pad,
        W_pad,
        H_out,
        W_out,
        k_h,
        k_w,
        s_h,
        s_w,
    )


def maxpool2d_forward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
    argmax_idx: np.ndarray,
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
) -> None:
    """
    Execute the native MaxPool2D forward kernel (float64, CPU) via ctypes.

    Requirements
    ------------
    - x_pad: float64 contiguous, shape (N, C, H_pad, W_pad)
    - y: float64 contiguous, shape (N, C, H_out, W_out)
    - argmax_idx: int64 contiguous, shape (N, C, H_out, W_out)
    """
    if x_pad.dtype != np.float64:
        raise TypeError(f"x_pad must be float64, got {x_pad.dtype}")
    if y.dtype != np.float64:
        raise TypeError(f"y must be float64, got {y.dtype}")
    if argmax_idx.dtype != np.int64:
        raise TypeError(f"argmax_idx must be int64, got {argmax_idx.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous (allocate it contiguously)")
    if not argmax_idx.flags["C_CONTIGUOUS"]:
        raise ValueError("argmax_idx must be C-contiguous (allocate it contiguously)")

    fn = lib.keydnn_maxpool2d_forward_f64
    fn.argtypes = [
        POINTER(c_double),  # x_pad
        POINTER(c_double),  # y
        POINTER(c_int64),  # argmax_idx
        c_int,
        c_int,
        c_int,
        c_int,  # N, C, H_pad, W_pad
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    x_ptr = x_pad.ctypes.data_as(POINTER(c_double))
    y_ptr = y.ctypes.data_as(POINTER(c_double))
    idx_ptr = argmax_idx.ctypes.data_as(POINTER(c_int64))

    fn(
        x_ptr,
        y_ptr,
        idx_ptr,
        N,
        C,
        H_pad,
        W_pad,
        H_out,
        W_out,
        k_h,
        k_w,
        s_h,
        s_w,
    )
