"""
ctypes bindings for KeyDNN v2 CUDA Pad2D / Crop2D utilities.

This module provides low-level device-pointer bindings to helper kernels used
by CUDA pooling ops to avoid host round-trips:

- pad2d:  construct a padded tensor (N, C, H+2p_h, W+2p_w) from (N, C, H, W)
- crop2d: extract the unpadded region (N, C, H, W) from a padded tensor

Assumptions
-----------
- The CUDA DLL exports C ABI functions:
    - keydnn_cuda_pad2d_f32 / keydnn_cuda_pad2d_f64
    - keydnn_cuda_crop2d_f32 / keydnn_cuda_crop2d_f64
- Device pointers are passed as uintptr_t handles (Python int).
- Float32/float64 only.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    c_int,
    c_float,
    c_double,
    c_int64,
    c_uint64,
    POINTER,
    c_void_p,
)
import numpy as np

# Reuse the same DLL loader / CUDA utils from your existing maxpool2d ctypes.
from .maxpool2d_ctypes import (
    DevPtr,
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_synchronize,
)


def _bind_pad2d(lib: ctypes.CDLL) -> None:
    """Bind argtypes/restype for pad2d/crop2d CUDA exports."""
    # pad f32
    lib.keydnn_cuda_pad2d_f32.argtypes = [
        POINTER(c_float),  # x
        POINTER(c_float),  # y_pad
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H, W
        c_int,
        c_int,  # p_h, p_w
        c_float,  # pad_value
    ]
    lib.keydnn_cuda_pad2d_f32.restype = c_int

    # pad f64
    lib.keydnn_cuda_pad2d_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_double,
    ]
    lib.keydnn_cuda_pad2d_f64.restype = c_int

    # crop f32
    lib.keydnn_cuda_crop2d_f32.argtypes = [
        POINTER(c_float),  # x_pad
        POINTER(c_float),  # y
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H_pad, W_pad
        c_int,
        c_int,  # p_h, p_w
        c_int,
        c_int,  # H, W
    ]
    lib.keydnn_cuda_crop2d_f32.restype = c_int

    # crop f64
    lib.keydnn_cuda_crop2d_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_crop2d_f64.restype = c_int


def _as_f32_ptr(dev_ptr: DevPtr) -> c_void_p:
    """
    Convert a device pointer handle (Python int) into a typed float* for ctypes.

    Notes
    -----
    On Windows, ctypes.cast expects a pointer-like instance (e.g., c_void_p),
    not a raw integer or c_uint64. We therefore wrap the integer address in
    c_void_p first, then cast to the desired pointer type.
    """
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_float))


def _as_f64_ptr(dev_ptr: DevPtr) -> c_void_p:
    """
    Convert a device pointer handle (Python int) into a typed double* for ctypes.
    """
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_double))


def _as_i64_ptr(dev_ptr: DevPtr) -> c_void_p:
    """
    Convert a device pointer handle (Python int) into a typed int64* for ctypes.
    """
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_int64))


def pad2d_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_pad_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    p_h: int,
    p_w: int,
    pad_value: float,
    dtype: np.dtype,
    device: int = 0,
    sync: bool = True,
) -> None:
    """
    Run CUDA pad2d on device buffers.

    Parameters
    ----------
    x_dev : DevPtr
        Input device pointer (N, C, H, W).
    y_pad_dev : DevPtr
        Output padded device pointer (N, C, H+2p_h, W+2p_w).
    pad_value : float
        Padding fill value (e.g., -inf for maxpool, 0 for avgpool).
    dtype : np.dtype
        np.float32 or np.float64.
    """
    _bind_pad2d(lib)
    cuda_set_device(lib, int(device))

    if dtype == np.float32:
        st = lib.keydnn_cuda_pad2d_f32(
            _as_f32_ptr(x_dev),
            _as_f32_ptr(y_pad_dev),
            int(N),
            int(C),
            int(H),
            int(W),
            int(p_h),
            int(p_w),
            c_float(float(pad_value)),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_pad2d_f64(
            _as_f64_ptr(x_dev),
            _as_f64_ptr(y_pad_dev),
            int(N),
            int(C),
            int(H),
            int(W),
            int(p_h),
            int(p_w),
            c_double(float(pad_value)),
        )
    else:
        raise TypeError(f"pad2d_cuda supports float32/float64 only, got {dtype}")

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_pad2d failed with status={st}")

    if sync:
        cuda_synchronize(lib)


def crop2d_cuda(
    lib: ctypes.CDLL,
    *,
    x_pad_dev: DevPtr,
    y_dev: DevPtr,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    p_h: int,
    p_w: int,
    H: int,
    W: int,
    dtype: np.dtype,
    device: int = 0,
    sync: bool = True,
) -> None:
    """
    Run CUDA crop2d on device buffers.

    Extracts y(n,c,h,w) = x_pad(n,c,h+p_h,w+p_w).
    """
    _bind_pad2d(lib)
    cuda_set_device(lib, int(device))

    if dtype == np.float32:
        st = lib.keydnn_cuda_crop2d_f32(
            _as_f32_ptr(x_pad_dev),
            _as_f32_ptr(y_dev),
            int(N),
            int(C),
            int(H_pad),
            int(W_pad),
            int(p_h),
            int(p_w),
            int(H),
            int(W),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_crop2d_f64(
            _as_f64_ptr(x_pad_dev),
            _as_f64_ptr(y_dev),
            int(N),
            int(C),
            int(H_pad),
            int(W_pad),
            int(p_h),
            int(p_w),
            int(H),
            int(W),
        )
    else:
        raise TypeError(f"crop2d_cuda supports float32/float64 only, got {dtype}")

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_crop2d failed with status={st}")

    if sync:
        cuda_synchronize(lib)
