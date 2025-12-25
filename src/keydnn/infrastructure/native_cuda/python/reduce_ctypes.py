"""
ctypes bindings for KeyDNN v2 CUDA reduction kernels (sum/mean/max).

Exports expected in the CUDA DLL:
- keydnn_cuda_sum_all_f32 / _f64
- keydnn_cuda_mean_all_f32 / _f64
- keydnn_cuda_sum_backward_fill_f32 / _f64
- keydnn_cuda_mean_backward_fill_f32 / _f64
- keydnn_cuda_max_axis2d_forward_f32 / _f64
- keydnn_cuda_max_axis2d_backward_f32 / _f64

Assumptions
-----------
- Device pointers are uintptr_t handles (Python int).
- Input tensors are contiguous.
- Max axis is implemented only for 2D tensors with axis in {0,1}.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_float, c_int, c_int64, c_void_p
import numpy as np

DevPtr = int


def _bind_reduce(lib: ctypes.CDLL) -> None:
    # sum_all
    lib.keydnn_cuda_sum_all_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int64,
    ]
    lib.keydnn_cuda_sum_all_f32.restype = c_int
    lib.keydnn_cuda_sum_all_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int64,
    ]
    lib.keydnn_cuda_sum_all_f64.restype = c_int

    # mean_all
    lib.keydnn_cuda_mean_all_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int64,
    ]
    lib.keydnn_cuda_mean_all_f32.restype = c_int
    lib.keydnn_cuda_mean_all_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int64,
    ]
    lib.keydnn_cuda_mean_all_f64.restype = c_int

    # sum backward fill
    lib.keydnn_cuda_sum_backward_fill_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int64,
    ]
    lib.keydnn_cuda_sum_backward_fill_f32.restype = c_int
    lib.keydnn_cuda_sum_backward_fill_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int64,
    ]
    lib.keydnn_cuda_sum_backward_fill_f64.restype = c_int

    # mean backward fill
    lib.keydnn_cuda_mean_backward_fill_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int64,
    ]
    lib.keydnn_cuda_mean_backward_fill_f32.restype = c_int
    lib.keydnn_cuda_mean_backward_fill_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int64,
    ]
    lib.keydnn_cuda_mean_backward_fill_f64.restype = c_int

    # max axis2d forward/backward (idx is int64)
    lib.keydnn_cuda_max_axis2d_forward_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_int64),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_max_axis2d_forward_f32.restype = c_int
    lib.keydnn_cuda_max_axis2d_forward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int64),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_max_axis2d_forward_f64.restype = c_int

    lib.keydnn_cuda_max_axis2d_backward_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_int64),
        POINTER(c_float),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_max_axis2d_backward_f32.restype = c_int
    lib.keydnn_cuda_max_axis2d_backward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_int64),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_max_axis2d_backward_f64.restype = c_int


def _as_ptr_float(dev_ptr: DevPtr):
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_float))


def _as_ptr_double(dev_ptr: DevPtr):
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_double))


def _as_ptr_int64(dev_ptr: DevPtr):
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_int64))


def sum_all_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,  # device scalar pointer
    numel: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if dtype == np.float32:
        st = lib.keydnn_cuda_sum_all_f32(
            _as_ptr_float(x_dev), _as_ptr_float(y_dev), int(numel)
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_sum_all_f64(
            _as_ptr_double(x_dev), _as_ptr_double(y_dev), int(numel)
        )
    else:
        raise TypeError(f"sum_all_cuda supports float32/float64 only, got {dtype}")
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_sum_all failed with status={st}")


def mean_all_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    numel: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if dtype == np.float32:
        st = lib.keydnn_cuda_mean_all_f32(
            _as_ptr_float(x_dev), _as_ptr_float(y_dev), int(numel)
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_mean_all_f64(
            _as_ptr_double(x_dev), _as_ptr_double(y_dev), int(numel)
        )
    else:
        raise TypeError(f"mean_all_cuda supports float32/float64 only, got {dtype}")
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_mean_all failed with status={st}")


def sum_backward_fill_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,  # device scalar
    grad_x_dev: DevPtr,  # device array (numel)
    numel: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if dtype == np.float32:
        st = lib.keydnn_cuda_sum_backward_fill_f32(
            _as_ptr_float(grad_out_dev), _as_ptr_float(grad_x_dev), int(numel)
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_sum_backward_fill_f64(
            _as_ptr_double(grad_out_dev), _as_ptr_double(grad_x_dev), int(numel)
        )
    else:
        raise TypeError(
            f"sum_backward_fill_cuda supports float32/float64 only, got {dtype}"
        )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_sum_backward_fill failed with status={st}")


def mean_backward_fill_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,
    grad_x_dev: DevPtr,
    numel: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if dtype == np.float32:
        st = lib.keydnn_cuda_mean_backward_fill_f32(
            _as_ptr_float(grad_out_dev), _as_ptr_float(grad_x_dev), int(numel)
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_mean_backward_fill_f64(
            _as_ptr_double(grad_out_dev), _as_ptr_double(grad_x_dev), int(numel)
        )
    else:
        raise TypeError(
            f"mean_backward_fill_cuda supports float32/float64 only, got {dtype}"
        )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_mean_backward_fill failed with status={st}")


def max_axis2d_forward_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    idx_dev: DevPtr,  # int64 device buffer
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for max_axis2d_forward_cuda")

    if dtype == np.float32:
        st = lib.keydnn_cuda_max_axis2d_forward_f32(
            _as_ptr_float(x_dev),
            _as_ptr_float(y_dev),
            _as_ptr_int64(idx_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_max_axis2d_forward_f64(
            _as_ptr_double(x_dev),
            _as_ptr_double(y_dev),
            _as_ptr_int64(idx_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    else:
        raise TypeError(
            f"max_axis2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_max_axis2d_forward failed with status={st}")


def max_axis2d_backward_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,
    idx_dev: DevPtr,
    grad_x_dev: DevPtr,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for max_axis2d_backward_cuda")

    if dtype == np.float32:
        st = lib.keydnn_cuda_max_axis2d_backward_f32(
            _as_ptr_float(grad_out_dev),
            _as_ptr_int64(idx_dev),
            _as_ptr_float(grad_x_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_max_axis2d_backward_f64(
            _as_ptr_double(grad_out_dev),
            _as_ptr_int64(idx_dev),
            _as_ptr_double(grad_x_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    else:
        raise TypeError(
            f"max_axis2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_max_axis2d_backward failed with status={st}")
