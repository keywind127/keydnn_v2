"""
ctypes bindings for KeyDNN AvgPool2D native kernels.

This module exposes thin Python wrappers around KeyDNN's C++ AvgPool2D kernels
compiled into the shared library loaded by :func:`load_keydnn_native`.

Supported kernels
-----------------
- AvgPool2D forward (float32 / float64)
- AvgPool2D backward (float32 / float64)

All kernels assume NCHW layout and C-contiguous NumPy arrays.

Design notes
------------
- These wrappers only validate dtypes and contiguity, then invoke the native
  symbol with a fixed ctypes signature.
- Output buffers (e.g., ``y`` and ``grad_x_pad``) are written in-place.
- The native kernels operate on *padded* inputs/gradients. The caller (Python
  op) is responsible for applying padding and slicing padding away when
  returning gradients.

Usage
-----
Typical usage is:

1) ``lib = load_keydnn_native()``
2) Call the appropriate wrapper based on dtype (float32 / float64).

The import of ``load_keydnn_native`` is intentionally retained to ensure the
dynamic loader is available in the same package namespace.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int, c_float, c_double

import numpy as np

from .maxpool2d_ctypes import (
    load_keydnn_native,
)  # reuse your loader, don't remove, dynamic loading


class AvgPool2DNativeKernels:
    """
    Namespace for AvgPool2D native kernel symbol names.

    This class is informational (a small "registry") and does not implement any
    runtime behavior. It helps keep exported C symbol names centralized and
    avoids scattering hard-coded strings throughout the module.

    Attributes
    ----------
    FORWARD_F32 : str
        Exported symbol for float32 forward kernel.
    FORWARD_F64 : str
        Exported symbol for float64 forward kernel.
    BACKWARD_F32 : str
        Exported symbol for float32 backward kernel.
    BACKWARD_F64 : str
        Exported symbol for float64 backward kernel.
    """

    FORWARD_F32: str = "keydnn_avgpool2d_forward_f32"
    FORWARD_F64: str = "keydnn_avgpool2d_forward_f64"
    BACKWARD_F32: str = "keydnn_avgpool2d_backward_f32"
    BACKWARD_F64: str = "keydnn_avgpool2d_backward_f64"


def avgpool2d_forward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
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
    Call the native C++ AvgPool2D forward kernel (float32, NCHW) via ctypes.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    x_pad : np.ndarray
        Zero-padded input tensor, shape (N, C, H_pad, W_pad),
        dtype float32, C-contiguous.
    y : np.ndarray
        Output buffer, shape (N, C, H_out, W_out),
        dtype float32, C-contiguous.
    N, C : int
        Batch size and number of channels.
    H_pad, W_pad : int
        Spatial dimensions of the padded input.
    H_out, W_out : int
        Spatial dimensions of the output.
    k_h, k_w : int
        Pooling kernel height and width.
    s_h, s_w : int
        Pooling stride height and width.

    Returns
    -------
    None
        Results are written in-place into ``y``.

    Notes
    -----
    - The caller must apply padding to ``x`` and pass the padded tensor as
      ``x_pad``.
    - Only float32 is supported by this wrapper.
    """
    if x_pad.dtype != np.float32:
        raise TypeError(f"x_pad must be float32, got {x_pad.dtype}")
    if y.dtype != np.float32:
        raise TypeError(f"y must be float32, got {y.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous")

    fn = getattr(lib, AvgPool2DNativeKernels.FORWARD_F32)
    fn.argtypes = [
        POINTER(c_float),  # x_pad
        POINTER(c_float),  # y
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_pad, W_pad
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        x_pad.ctypes.data_as(POINTER(c_float)),
        y.ctypes.data_as(POINTER(c_float)),
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


def avgpool2d_forward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    y: np.ndarray,
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
    Call the native C++ AvgPool2D forward kernel (float64, NCHW) via ctypes.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    x_pad : np.ndarray
        Zero-padded input tensor, shape (N, C, H_pad, W_pad),
        dtype float64, C-contiguous.
    y : np.ndarray
        Output buffer, shape (N, C, H_out, W_out),
        dtype float64, C-contiguous.
    N, C : int
        Batch size and number of channels.
    H_pad, W_pad : int
        Spatial dimensions of the padded input.
    H_out, W_out : int
        Spatial dimensions of the output.
    k_h, k_w : int
        Pooling kernel height and width.
    s_h, s_w : int
        Pooling stride height and width.

    Returns
    -------
    None
        Results are written in-place into ``y``.

    Notes
    -----
    - The caller must apply padding to ``x`` and pass the padded tensor as
      ``x_pad``.
    - Only float64 is supported by this wrapper.
    """
    if x_pad.dtype != np.float64:
        raise TypeError(f"x_pad must be float64, got {x_pad.dtype}")
    if y.dtype != np.float64:
        raise TypeError(f"y must be float64, got {y.dtype}")

    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)
    if not y.flags["C_CONTIGUOUS"]:
        raise ValueError("y must be C-contiguous")

    fn = getattr(lib, AvgPool2DNativeKernels.FORWARD_F64)
    fn.argtypes = [
        POINTER(c_double),  # x_pad
        POINTER(c_double),  # y
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_pad, W_pad
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        x_pad.ctypes.data_as(POINTER(c_double)),
        y.ctypes.data_as(POINTER(c_double)),
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


def avgpool2d_backward_f32_ctypes(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    grad_x_pad: np.ndarray,
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
    """
    Call the native C++ AvgPool2D backward kernel (float32, NCHW) via ctypes.

    This kernel distributes each output gradient element uniformly across the
    corresponding pooling window in the padded input gradient buffer
    (``grad_x_pad``). The caller is responsible for slicing away padding after
    the call.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    grad_out : np.ndarray
        Gradient with respect to the output, shape (N, C, H_out, W_out),
        dtype float32, C-contiguous.
    grad_x_pad : np.ndarray
        Output buffer for gradients with respect to the padded input, shape
        (N, C, H_pad, W_pad), dtype float32, C-contiguous.
        Must be zero-initialized before calling.
    N, C : int
        Batch size and number of channels.
    H_out, W_out : int
        Spatial dimensions of the output gradient.
    H_pad, W_pad : int
        Spatial dimensions of the padded input gradient buffer.
    k_h, k_w : int
        Pooling kernel height and width.
    s_h, s_w : int
        Pooling stride height and width.

    Returns
    -------
    None
        Results are written in-place into ``grad_x_pad``.

    Notes
    -----
    - Only float32 is supported by this wrapper.
    - ``grad_x_pad`` must be C-contiguous and zero-initialized.
    """
    if grad_out.dtype != np.float32:
        raise TypeError(f"grad_out must be float32, got {grad_out.dtype}")
    if grad_x_pad.dtype != np.float32:
        raise TypeError(f"grad_x_pad must be float32, got {grad_x_pad.dtype}")

    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)
    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous")

    fn = getattr(lib, AvgPool2DNativeKernels.BACKWARD_F32)
    fn.argtypes = [
        POINTER(c_float),  # grad_out
        POINTER(c_float),  # grad_x_pad
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # H_pad, W_pad
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        grad_out.ctypes.data_as(POINTER(c_float)),
        grad_x_pad.ctypes.data_as(POINTER(c_float)),
        N,
        C,
        H_out,
        W_out,
        H_pad,
        W_pad,
        k_h,
        k_w,
        s_h,
        s_w,
    )


def avgpool2d_backward_f64_ctypes(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    grad_x_pad: np.ndarray,
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
    """
    Call the native C++ AvgPool2D backward kernel (float64, NCHW) via ctypes.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN native shared library.
    grad_out : np.ndarray
        Gradient with respect to the output, shape (N, C, H_out, W_out),
        dtype float64, C-contiguous.
    grad_x_pad : np.ndarray
        Output buffer for gradients with respect to the padded input, shape
        (N, C, H_pad, W_pad), dtype float64, C-contiguous.
        Must be zero-initialized before calling.
    N, C : int
        Batch size and number of channels.
    H_out, W_out : int
        Spatial dimensions of the output gradient.
    H_pad, W_pad : int
        Spatial dimensions of the padded input gradient buffer.
    k_h, k_w : int
        Pooling kernel height and width.
    s_h, s_w : int
        Pooling stride height and width.

    Returns
    -------
    None
        Results are written in-place into ``grad_x_pad``.

    Notes
    -----
    - Only float64 is supported by this wrapper.
    - The caller must remove padding after accumulation.
    """
    if grad_out.dtype != np.float64:
        raise TypeError(f"grad_out must be float64, got {grad_out.dtype}")
    if grad_x_pad.dtype != np.float64:
        raise TypeError(f"grad_x_pad must be float64, got {grad_x_pad.dtype}")

    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)
    if not grad_x_pad.flags["C_CONTIGUOUS"]:
        raise ValueError("grad_x_pad must be C-contiguous")

    fn = getattr(lib, AvgPool2DNativeKernels.BACKWARD_F64)
    fn.argtypes = [
        POINTER(c_double),  # grad_out
        POINTER(c_double),  # grad_x_pad
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_out, W_out
        c_int,
        c_int,  # H_pad, W_pad
        c_int,
        c_int,  # k_h, k_w
        c_int,
        c_int,  # s_h, s_w
    ]
    fn.restype = None

    fn(
        grad_out.ctypes.data_as(POINTER(c_double)),
        grad_x_pad.ctypes.data_as(POINTER(c_double)),
        N,
        C,
        H_out,
        W_out,
        H_pad,
        W_pad,
        k_h,
        k_w,
        s_h,
        s_w,
    )
