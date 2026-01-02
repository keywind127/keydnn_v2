"""
ctypes bindings for KeyDNN v2 CUDA reduction kernels (sum/mean/max).

Exports expected in the CUDA DLL:
- keydnn_cuda_sum_all_f32 / _f64
- keydnn_cuda_mean_all_f32 / _f64
- keydnn_cuda_sum_backward_fill_f32 / _f64
- keydnn_cuda_mean_backward_fill_f32 / _f64
- keydnn_cuda_max_axis2d_forward_f32 / _f64
- keydnn_cuda_max_axis2d_backward_f32 / _f64
- keydnn_cuda_sum_axis2d_forward_f32 / _f64
- keydnn_cuda_sum_axis2d_backward_f32 / _f64
- keydnn_cuda_sum_to_shape_f32 / _f64

Assumptions
-----------
- Device pointers are uintptr_t handles (Python int).
- Input tensors are contiguous.
- Max axis is implemented only for 2D tensors with axis in {0,1}.
- Sum axis is implemented only for 2D tensors with axis in {0,1}.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_double, c_float, c_int, c_int64, c_void_p
import numpy as np

DevPtr = int


def _bind_reduce(lib: ctypes.CDLL) -> None:
    # sum_all
    lib.keydnn_cuda_sum_all_f32.argtypes = [POINTER(c_float), POINTER(c_float), c_int64]
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

    # sum axis2d forward/backward
    lib.keydnn_cuda_sum_axis2d_forward_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_sum_axis2d_forward_f32.restype = c_int
    lib.keydnn_cuda_sum_axis2d_forward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_sum_axis2d_forward_f64.restype = c_int

    lib.keydnn_cuda_sum_axis2d_backward_f32.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_sum_axis2d_backward_f32.restype = c_int
    lib.keydnn_cuda_sum_axis2d_backward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_sum_axis2d_backward_f64.restype = c_int
    # sum_to_shape (general unbroadcast reduction)
    lib.keydnn_cuda_sum_to_shape_f32.argtypes = [
        POINTER(c_float),  # x
        POINTER(c_float),  # y
        POINTER(c_int64),  # in_shape (host)
        POINTER(c_int64),  # out_shape (host)
        c_int,  # ndim
    ]
    lib.keydnn_cuda_sum_to_shape_f32.restype = c_int

    lib.keydnn_cuda_sum_to_shape_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int64),
        POINTER(c_int64),
        c_int,
    ]
    lib.keydnn_cuda_sum_to_shape_f64.restype = c_int


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


def sum_axis2d_forward_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for sum_axis2d_forward_cuda")

    if dtype == np.float32:
        st = lib.keydnn_cuda_sum_axis2d_forward_f32(
            _as_ptr_float(x_dev),
            _as_ptr_float(y_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_sum_axis2d_forward_f64(
            _as_ptr_double(x_dev),
            _as_ptr_double(y_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    else:
        raise TypeError(
            f"sum_axis2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_sum_axis2d_forward failed with status={st}")


def sum_axis2d_backward_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,
    grad_x_dev: DevPtr,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
) -> None:
    _bind_reduce(lib)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for sum_axis2d_backward_cuda")

    if dtype == np.float32:
        st = lib.keydnn_cuda_sum_axis2d_backward_f32(
            _as_ptr_float(grad_out_dev),
            _as_ptr_float(grad_x_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_sum_axis2d_backward_f64(
            _as_ptr_double(grad_out_dev),
            _as_ptr_double(grad_x_dev),
            int(rows),
            int(cols),
            int(axis),
        )
    else:
        raise TypeError(
            f"sum_axis2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_sum_axis2d_backward failed with status={st}")


def _as_host_int64_array(x: np.ndarray) -> np.ndarray:
    """
    Ensure a host-side int64 contiguous array for shape metadata.

    We pass shape arrays by pointer into the DLL. The native code copies them
    immediately, but we still keep the array alive for the duration of the call.
    """
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1:
        raise TypeError(f"shape arrays must be 1D, got shape={x.shape}")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def sum_to_shape_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    in_shape: tuple[int, ...] | np.ndarray,
    out_shape: tuple[int, ...] | np.ndarray,
    dtype: np.dtype,
) -> None:
    """
    Reduce-sum `x` into `y` according to broadcast-compatible `out_shape`.

    This is the CUDA primitive used for "unbroadcast" in backward passes:
    it sums over dimensions where out_shape[d] == 1 while in_shape[d] > 1.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA native DLL.
    x_dev : DevPtr
        Device pointer to input buffer.
    y_dev : DevPtr
        Device pointer to output buffer (caller allocates sized for out_shape).
    in_shape : tuple[int,...] | np.ndarray
        Shape of input tensor (rank must match out_shape rank).
    out_shape : tuple[int,...] | np.ndarray
        Target shape (same rank). For each d: out_shape[d] == in_shape[d] or 1.
    dtype : np.dtype
        float32 or float64.

    Raises
    ------
    TypeError
        If dtype unsupported or shapes are invalid.
    ValueError
        If ranks mismatch or ndim is invalid.
    RuntimeError
        If native kernel returns non-zero status.
    """
    _bind_reduce(lib)
    dtype = np.dtype(dtype)

    in_arr = _as_host_int64_array(np.array(in_shape, dtype=np.int64))
    out_arr = _as_host_int64_array(np.array(out_shape, dtype=np.int64))

    if in_arr.size != out_arr.size:
        raise ValueError(
            f"in_shape and out_shape must have same rank, got {in_arr.size} vs {out_arr.size}"
        )

    ndim = int(in_arr.size)
    if ndim <= 0:
        raise ValueError(f"ndim must be positive, got {ndim}")

    in_ptr = in_arr.ctypes.data_as(POINTER(c_int64))
    out_ptr = out_arr.ctypes.data_as(POINTER(c_int64))

    if dtype == np.float32:
        st = lib.keydnn_cuda_sum_to_shape_f32(
            _as_ptr_float(x_dev),
            _as_ptr_float(y_dev),
            in_ptr,
            out_ptr,
            int(ndim),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_sum_to_shape_f64(
            _as_ptr_double(x_dev),
            _as_ptr_double(y_dev),
            in_ptr,
            out_ptr,
            int(ndim),
        )
    else:
        raise TypeError(f"sum_to_shape_cuda supports float32/float64 only, got {dtype}")

    if st != 0:
        raise RuntimeError(f"keydnn_cuda_sum_to_shape failed with status={st}")
