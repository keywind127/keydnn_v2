from __future__ import annotations

import os
import sys
import ctypes
from ctypes import c_int, c_float, c_int64, POINTER
from typing import Optional

import numpy as np


def _default_lib_name() -> str:
    if sys.platform.startswith("win"):
        return "keydnn_native.dll"
    if sys.platform == "darwin":
        return "libkeydnn_native.dylib"
    return "libkeydnn_native.so"


def load_keydnn_native(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """
    Load the native KeyDNN shared library.

    Parameters
    ----------
    lib_path : Optional[str]
        Absolute or relative path to the compiled shared library. If None, tries
        to load from the same directory as this file.

    Returns
    -------
    ctypes.CDLL
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
    Call the C++ maxpool2d forward kernel for float32, NCHW contiguous arrays.

    Requirements
    ------------
    - x_pad: float32 contiguous, shape (N, C, H_pad, W_pad)
    - y: float32 contiguous, shape (N, C, H_out, W_out)
    - argmax_idx: int64 contiguous, shape (N, C, H_out, W_out)
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

    # Bind signature once
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
