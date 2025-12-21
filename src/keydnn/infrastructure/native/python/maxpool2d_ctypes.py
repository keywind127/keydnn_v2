"""
ctypes bindings for KeyDNN native CPU kernels.

This module provides low-level Python bindings to native C/C++ implementations
of performance-critical tensor operations via `ctypes`. It is part of the
KeyDNN infrastructure layer and is intentionally backend-specific.

Currently, this module exposes:
- A CPU-based MaxPool2D forward/backward kernels for float32/float64 tensors in NCHW layout.

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
    from pathlib import Path

    # Resolve the target DLL path (your existing logic can remain; this is just a sketch)
    if lib_path is None:
        # default path (example)
        lib_path = str(Path(__file__).resolve().parent / "keydnn_native.dll")

    dll_path = Path(lib_path).resolve()
    if not dll_path.exists():
        raise FileNotFoundError(f"Native library not found: {dll_path}")

    # ---- Critical: ensure dependent DLL search paths are registered on Windows (Py 3.8+) ----
    add_dirs = []
    if sys.platform.startswith("win"):
        # 1) directory containing keydnn_native_omp.dll / keydnn_native_noomp.dll
        add_dirs.append(str(dll_path.parent))

        # 2) MinGW bin directory (best effort)
        mingw_bin = os.environ.get("KEYDNN_MINGW_BIN", "")
        if mingw_bin:
            add_dirs.append(mingw_bin)

    # Register DLL directories (Python 3.8+)
    # Keep handles alive to preserve directory registration.
    _handles = []
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        for d in add_dirs:
            if d and Path(d).exists():
                _handles.append(os.add_dll_directory(d))

    # Now load the DLL
    try:
        return ctypes.CDLL(str(dll_path))
    except OSError as e:
        # Optional: include helpful details
        raise OSError(
            f"Failed to load native library: {dll_path}\n"
            f"Registered DLL dirs: {add_dirs}\n"
            f"Original error: {e}"
        ) from e


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


def maxpool2d_backward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    argmax_idx: np.ndarray,
    grad_x_pad: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
) -> None:
    """
    Call the native C++ MaxPool2D backward kernel (float32, NCHW) via ctypes.

    This function accumulates gradients into the padded input gradient buffer
    (`grad_x_pad`) by routing each output gradient element to the input location
    that won the max operation during the forward pass.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    grad_out : np.ndarray
        Gradient with respect to the output, shape (N, C, H_out, W_out),
        dtype float32, C-contiguous.
    argmax_idx : np.ndarray
        Argmax indices produced by the forward pass, shape (N, C, H_out, W_out),
        dtype int64, storing flattened indices into the padded spatial plane.
    grad_x_pad : np.ndarray
        Output buffer for gradients with respect to the padded input,
        shape (N, C, H_pad, W_pad), dtype float32, C-contiguous.
        Must be zero-initialized before calling.
    N, C : int
        Batch size and number of channels.
    H_out, W_out : int
        Spatial dimensions of the pooling output.
    H_pad, W_pad : int
        Spatial dimensions of the padded input.

    Returns
    -------
    None
        Results are written in-place into `grad_x_pad`.

    Notes
    -----
    - This function performs no padding removal; the caller is responsible
      for slicing `grad_x_pad` back to the original input shape.
    - Only float32 inputs are supported by this wrapper.
    - No bounds checking is performed inside the native kernel.
    """
    if grad_out.dtype != np.float32:
        raise TypeError(f"grad_out must be float32, got {grad_out.dtype}")
    if argmax_idx.dtype != np.int64:
        raise TypeError(f"argmax_idx must be int64, got {argmax_idx.dtype}")
    if grad_x_pad.dtype != np.float32:
        raise TypeError(f"grad_x_pad must be float32, got {grad_x_pad.dtype}")

    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)
    if not argmax_idx.flags["C_CONTIGUOUS"]:
        argmax_idx = np.ascontiguousarray(argmax_idx)
    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous")

    fn = lib.keydnn_maxpool2d_backward_f32
    fn.argtypes = [
        POINTER(c_float),  # grad_out
        POINTER(c_int64),  # argmax_idx
        POINTER(c_float),  # grad_x_pad
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # H_pad, W_pad
    ]
    fn.restype = None

    fn(
        grad_out.ctypes.data_as(POINTER(c_float)),
        argmax_idx.ctypes.data_as(POINTER(c_int64)),
        grad_x_pad.ctypes.data_as(POINTER(c_float)),
        N,
        C,
        H_out,
        W_out,
        H_pad,
        W_pad,
    )


def maxpool2d_backward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    argmax_idx: np.ndarray,
    grad_x_pad: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
) -> None:
    """
    Call the native C++ MaxPool2D backward kernel (float64, NCHW) via ctypes.

    This function is identical in semantics to
    `maxpool2d_backward_f32_ctypes`, but operates on double-precision
    floating-point data.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    grad_out : np.ndarray
        Gradient with respect to the output, shape (N, C, H_out, W_out),
        dtype float64, C-contiguous.
    argmax_idx : np.ndarray
        Argmax indices produced by the forward pass, shape (N, C, H_out, W_out),
        dtype int64.
    grad_x_pad : np.ndarray
        Output buffer for gradients with respect to the padded input,
        shape (N, C, H_pad, W_pad), dtype float64, C-contiguous.
        Must be zero-initialized before calling.
    N, C : int
        Batch size and number of channels.
    H_out, W_out : int
        Spatial dimensions of the pooling output.
    H_pad, W_pad : int
        Spatial dimensions of the padded input.

    Returns
    -------
    None
        Results are written in-place into `grad_x_pad`.

    Notes
    -----
    - This wrapper supports float64 only.
    - The native kernel accumulates gradients additively.
    - The caller is responsible for removing padding after the call.
    """
    if grad_out.dtype != np.float64:
        raise TypeError(f"grad_out must be float64, got {grad_out.dtype}")
    if argmax_idx.dtype != np.int64:
        raise TypeError(f"argmax_idx must be int64, got {argmax_idx.dtype}")
    if grad_x_pad.dtype != np.float64:
        raise TypeError(f"grad_x_pad must be float64, got {grad_x_pad.dtype}")

    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)
    if not argmax_idx.flags["C_CONTIGUOUS"]:
        argmax_idx = np.ascontiguousarray(argmax_idx)
    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous")

    fn = lib.keydnn_maxpool2d_backward_f64
    fn.argtypes = [
        POINTER(c_double),
        POINTER(c_int64),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    fn.restype = None

    fn(
        grad_out.ctypes.data_as(POINTER(c_double)),
        argmax_idx.ctypes.data_as(POINTER(c_int64)),
        grad_x_pad.ctypes.data_as(POINTER(c_double)),
        N,
        C,
        H_out,
        W_out,
        H_pad,
        W_pad,
    )
