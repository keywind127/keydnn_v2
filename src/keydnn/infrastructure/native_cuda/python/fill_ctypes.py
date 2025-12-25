"""
ctypes bindings for KeyDNN v2 CUDA fill kernels.

Exports:
- keydnn_cuda_fill_f32
- keydnn_cuda_fill_f64

Intended usage:
- Tensor.zeros (CUDA): cuda_malloc + cuda_memset(..., 0, nbytes)
- Tensor.ones  (CUDA): cuda_malloc + cuda_fill_* (value=1.0)
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_int64, c_void_p, c_float, c_double
from pathlib import Path

import numpy as np

from .maxpool2d_ctypes import (
    load_keydnn_cuda_native,  # you already use this loader elsewhere
    cuda_malloc,
    cuda_free,
    cuda_memset,
    cuda_synchronize,
)

DevPtr = int


class CudaFillLib:
    """
    Thin binding layer for CUDA fill exports.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        self._bound = False

    def _bind(self) -> None:
        if self._bound:
            return

        lib = self.lib

        # f32: (void* y, int64 numel, float value) -> int
        lib.keydnn_cuda_fill_f32.argtypes = [c_void_p, c_int64, c_float]
        lib.keydnn_cuda_fill_f32.restype = c_int

        # f64: (void* y, int64 numel, double value) -> int
        lib.keydnn_cuda_fill_f64.argtypes = [c_void_p, c_int64, c_double]
        lib.keydnn_cuda_fill_f64.restype = c_int

        self._bound = True

    @staticmethod
    def _as_dev_ptr(dev_ptr: DevPtr) -> c_void_p:
        return c_void_p(int(dev_ptr))

    def fill(self, *, y_dev: DevPtr, numel: int, value: float, dtype: np.dtype) -> None:
        """
        Fill a device buffer with a scalar value.

        Parameters
        ----------
        y_dev : DevPtr
            Device pointer.
        numel : int
            Number of elements.
        value : float
            Scalar fill value.
        dtype : np.dtype
            np.float32 or np.float64.
        """
        self._bind()

        dt = np.dtype(dtype)
        if dt == np.float32:
            st = self.lib.keydnn_cuda_fill_f32(
                self._as_dev_ptr(y_dev), c_int64(int(numel)), c_float(float(value))
            )
        elif dt == np.float64:
            st = self.lib.keydnn_cuda_fill_f64(
                self._as_dev_ptr(y_dev), c_int64(int(numel)), c_double(float(value))
            )
        else:
            raise TypeError(f"Unsupported dtype for CUDA fill: {dt}")

        if st != 0:
            raise RuntimeError(f"keydnn_cuda_fill failed with status={st}")


# ---------------------------------------------------------------------
# Singleton pattern (same as your other wrappers)
# ---------------------------------------------------------------------

_fill_singleton: CudaFillLib | None = None


def _get_fill(lib: ctypes.CDLL) -> CudaFillLib:
    global _fill_singleton
    if _fill_singleton is None or _fill_singleton.lib is not lib:
        _fill_singleton = CudaFillLib(lib)
    return _fill_singleton


def cuda_fill(
    lib: ctypes.CDLL, *, y_dev: DevPtr, numel: int, value: float, dtype: np.dtype
) -> None:
    """
    Module-level convenience wrapper for CUDA fill.
    """
    _get_fill(lib).fill(y_dev=y_dev, numel=numel, value=value, dtype=dtype)


# ---------------------------------------------------------------------
# Convenience helpers for Tensor.zeros / Tensor.ones (optional)
# ---------------------------------------------------------------------


def cuda_zeros_like_allocation(
    lib: ctypes.CDLL, *, shape: tuple[int, ...], dtype: np.dtype = np.float32
) -> DevPtr:
    """
    Allocate a device buffer and set it to zeros (byte-wise memset).

    Returns DevPtr. Caller frees.
    """
    dt = np.dtype(dtype)
    numel = int(np.prod(shape, dtype=np.int64))
    nbytes = int(numel) * int(dt.itemsize)
    y_dev = cuda_malloc(lib, nbytes)
    cuda_memset(lib, y_dev, 0, nbytes)
    cuda_synchronize(lib)
    return y_dev


def cuda_ones_like_allocation(
    lib: ctypes.CDLL, *, shape: tuple[int, ...], dtype: np.dtype = np.float32
) -> DevPtr:
    """
    Allocate a device buffer and fill it with ones (typed kernel fill).

    Returns DevPtr. Caller frees.
    """
    dt = np.dtype(dtype)
    numel = int(np.prod(shape, dtype=np.int64))
    nbytes = int(numel) * int(dt.itemsize)
    y_dev = cuda_malloc(lib, nbytes)
    cuda_fill(lib, y_dev=y_dev, numel=numel, value=1.0, dtype=dt)
    cuda_synchronize(lib)
    return y_dev
