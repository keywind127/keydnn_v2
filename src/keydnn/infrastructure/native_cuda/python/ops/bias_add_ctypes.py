"""
ctypes bindings for KeyDNN v2 CUDA bias-add kernels.

This module provides a small, low-level binding layer for adding a 1D bias
vector `b` to a 2D matrix-like buffer of shape (batch, out_features) on CUDA.

Two entry points are exposed:
- `bias_add_cuda`: out-of-place bias add, `y = x + b`
- `bias_add_inplace_cuda`: in-place bias add, `y += b`

The functions are intentionally thin wrappers:
- Device pointers are passed as Python ints (uintptr_t) and converted to
  `ctypes.c_void_p` for the native call.
- Dtype dispatch selects the float32 or float64 exported symbol.
- Argument type binding is performed once per loaded library via `_bind`.

Expected native exports
-----------------------
- keydnn_cuda_bias_add_f32 / keydnn_cuda_bias_add_f64
- keydnn_cuda_bias_add_inplace_f32 / keydnn_cuda_bias_add_inplace_f64

Notes
-----
- This module does not allocate memory and does not synchronize; callers are
  responsible for ensuring correct device selection and synchronization
  semantics at a higher level.
- Shapes are provided as `(batch, out_features)` integers; this wrapper assumes
  the native kernel uses them to map the bias vector across rows.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_int64, c_void_p
from typing import Any

import numpy as np


def _bind(lib: Any) -> None:
    """
    Bind argtypes/restype for bias-add CUDA exports (idempotent).

    Parameters
    ----------
    lib : Any
        Loaded native CUDA library handle (typically a `ctypes.CDLL`).

    Notes
    -----
    A private flag (`_keydnn_bias_add_bound`) is stored on the `lib` object to
    ensure binding happens at most once per library instance.
    """
    if getattr(lib, "_keydnn_bias_add_bound", False):
        return

    # y = x + b
    lib.keydnn_cuda_bias_add_f32.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_int64,
        c_int64,
    ]
    lib.keydnn_cuda_bias_add_f32.restype = c_int

    lib.keydnn_cuda_bias_add_f64.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_int64,
        c_int64,
    ]
    lib.keydnn_cuda_bias_add_f64.restype = c_int

    # y += b
    lib.keydnn_cuda_bias_add_inplace_f32.argtypes = [
        c_void_p,
        c_void_p,
        c_int64,
        c_int64,
    ]
    lib.keydnn_cuda_bias_add_inplace_f32.restype = c_int

    lib.keydnn_cuda_bias_add_inplace_f64.argtypes = [
        c_void_p,
        c_void_p,
        c_int64,
        c_int64,
    ]
    lib.keydnn_cuda_bias_add_inplace_f64.restype = c_int

    lib._keydnn_bias_add_bound = True


def bias_add_cuda(
    lib: Any,
    *,
    x_dev: int,
    b_dev: int,
    y_dev: int,
    batch: int,
    out_features: int,
    dtype: np.dtype,
) -> None:
    """
    Compute out-of-place bias add on CUDA: `y = x + b`.

    Parameters
    ----------
    lib : Any
        Loaded native CUDA library handle (typically a `ctypes.CDLL`).
    x_dev : int
        Device pointer to the input buffer `x`, interpreted as a contiguous
        array of length `batch * out_features`.
    b_dev : int
        Device pointer to the bias vector `b`, interpreted as length
        `out_features`.
    y_dev : int
        Device pointer to the output buffer `y`, interpreted as a contiguous
        array of length `batch * out_features`.
    batch : int
        Number of rows (batch dimension).
    out_features : int
        Number of columns (feature dimension); also the length of `b`.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Raises
    ------
    TypeError
        If `dtype` is not float32/float64.
    RuntimeError
        If the native kernel returns a non-zero status.

    Expected native signatures
    --------------------------
    - keydnn_cuda_bias_add_f32(void* x, void* b, void* y, int64 batch, int64 out_features)
    - keydnn_cuda_bias_add_f64(void* x, void* b, void* y, int64 batch, int64 out_features)
    """
    _bind(lib)
    dt = np.dtype(dtype)

    if dt == np.float32:
        st = lib.keydnn_cuda_bias_add_f32(
            c_void_p(int(x_dev)),
            c_void_p(int(b_dev)),
            c_void_p(int(y_dev)),
            c_int64(int(batch)),
            c_int64(int(out_features)),
        )
    elif dt == np.float64:
        st = lib.keydnn_cuda_bias_add_f64(
            c_void_p(int(x_dev)),
            c_void_p(int(b_dev)),
            c_void_p(int(y_dev)),
            c_int64(int(batch)),
            c_int64(int(out_features)),
        )
    else:
        raise TypeError(f"bias_add_cuda supports float32/float64 only, got {dt}")

    if int(st) != 0:
        raise RuntimeError(f"keydnn_cuda_bias_add failed with status={int(st)}")


def bias_add_inplace_cuda(
    lib: Any,
    *,
    y_dev: int,
    b_dev: int,
    batch: int,
    out_features: int,
    dtype: np.dtype,
) -> None:
    """
    Compute in-place bias add on CUDA: `y += b`.

    Parameters
    ----------
    lib : Any
        Loaded native CUDA library handle (typically a `ctypes.CDLL`).
    y_dev : int
        Device pointer to the in-place buffer `y`, interpreted as a contiguous
        array of length `batch * out_features`.
    b_dev : int
        Device pointer to the bias vector `b`, interpreted as length
        `out_features`.
    batch : int
        Number of rows (batch dimension).
    out_features : int
        Number of columns (feature dimension); also the length of `b`.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Raises
    ------
    TypeError
        If `dtype` is not float32/float64.
    RuntimeError
        If the native kernel returns a non-zero status.

    Expected native signatures
    --------------------------
    - keydnn_cuda_bias_add_inplace_f32(void* y, void* b, int64 batch, int64 out_features)
    - keydnn_cuda_bias_add_inplace_f64(void* y, void* b, int64 batch, int64 out_features)
    """
    _bind(lib)
    dt = np.dtype(dtype)

    if dt == np.float32:
        st = lib.keydnn_cuda_bias_add_inplace_f32(
            c_void_p(int(y_dev)),
            c_void_p(int(b_dev)),
            c_int64(int(batch)),
            c_int64(int(out_features)),
        )
    elif dt == np.float64:
        st = lib.keydnn_cuda_bias_add_inplace_f64(
            c_void_p(int(y_dev)),
            c_void_p(int(b_dev)),
            c_int64(int(batch)),
            c_int64(int(out_features)),
        )
    else:
        raise TypeError(
            f"bias_add_inplace_cuda supports float32/float64 only, got {dt}"
        )

    if int(st) != 0:
        raise RuntimeError(f"keydnn_cuda_bias_add_inplace failed with status={int(st)}")


__all__ = ["bias_add_cuda", "bias_add_inplace_cuda"]
