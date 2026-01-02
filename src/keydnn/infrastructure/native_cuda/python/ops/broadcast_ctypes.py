"""
infrastructure/native_cuda/python/ops/broadcast_ctypes.py

CUDA broadcast (broadcast_to) ctypes bindings for KeyDNN.

This module provides a thin, Windows-specific boundary layer that:
- locates exported symbols in the KeyDNN CUDA native DLL via `GetProcAddress`,
- marshals Python/NumPy metadata (input/output shapes) into C-compatible buffers,
- invokes the corresponding CUDA broadcast kernel through a typed `ctypes` function pointer.

The primary entrypoint is `broadcast_to_cuda`, which applies NumPy-style broadcasting
semantics in CUDA for supported floating-point dtypes (currently fp32/fp64). Shape
information is passed as int64 arrays.

Notes
-----
- This file intentionally contains minimal policy/validation and focuses on correct
  ABI marshalling.
- The native side is expected to interpret `in_shape` and `out_shape` and implement
  broadcasting accordingly.
"""

from __future__ import annotations
import ctypes
from ctypes import wintypes
import numpy as np

from .._native_loader import load_keydnn_cuda_native

_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    """
    Resolve an exported function symbol from a loaded native DLL.

    Parameters
    ----------
    lib:
        A loaded `ctypes.CDLL` instance (KeyDNN CUDA native library).
    sym:
        The ASCII name of the exported function symbol to resolve.

    Returns
    -------
    int
        The resolved function address as an integer.

    Raises
    ------
    RuntimeError
        If the symbol is missing (resolved address is null).

    Notes
    -----
    Uses Windows `kernel32.GetProcAddress` against `lib._handle`.
    """
    addr = _GetProcAddress(lib._handle, sym.encode("ascii"))
    v = int(ctypes.cast(addr, ctypes.c_void_p).value or 0)
    if v == 0:
        raise RuntimeError(f"Native DLL missing symbol: {sym}")
    return v


def _as_i64_arr(shape: tuple[int, ...]) -> np.ndarray:
    """
    Convert a Python shape tuple into a 1D NumPy int64 array.

    Parameters
    ----------
    shape:
        Shape tuple (e.g., `(N, C, H, W)`) to be passed to the native layer.

    Returns
    -------
    numpy.ndarray
        1D array of dtype int64 containing the shape dimensions.

    Raises
    ------
    ValueError
        If the produced array is not 1-dimensional (defensive check).
    """
    a = np.asarray(shape, dtype=np.int64)
    if a.ndim != 1:
        raise ValueError("shape must be 1D")
    return a


def broadcast_to_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    """
    Perform a CUDA broadcast-to operation using the KeyDNN native DLL.

    This function wraps native CUDA kernels that broadcast an input tensor with
    shape `in_shape` into an output tensor with shape `out_shape`, writing into
    pre-allocated device memory.

    Parameters
    ----------
    lib:
        Loaded KeyDNN CUDA native library handle (e.g., from `load_keydnn_cuda_native()`).
    x_dev:
        Device pointer (address) to the input tensor buffer.
    y_dev:
        Device pointer (address) to the output tensor buffer.
    in_shape:
        Input tensor shape. Must be broadcast-compatible with `out_shape`.
    out_shape:
        Output tensor shape after broadcasting.
    dtype:
        NumPy dtype of the tensor elements. Supported: `np.float32`, `np.float64`.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `dtype` is unsupported.
    RuntimeError
        If the native call returns a non-zero status code or if the required
        symbol is missing from the DLL.

    Native ABI
    ----------
    The resolved symbol is expected to match the following signature:

        int fn(
            void* x_dev,
            void* y_dev,
            int64_t* in_shape,
            int in_ndim,
            int64_t* out_shape,
            int out_ndim
        )

    Where `x_dev` and `y_dev` are device pointers and shapes are int64 arrays.
    """
    dtype = np.dtype(dtype)
    in_a = _as_i64_arr(in_shape)
    out_a = _as_i64_arr(out_shape)

    if dtype == np.float32:
        sym = "keydnn_cuda_broadcast_to_f32"
    elif dtype == np.float64:
        sym = "keydnn_cuda_broadcast_to_f64"
    else:
        raise TypeError(f"broadcast_to_cuda: unsupported dtype {dtype}")

    fn_addr = _get_proc_addr(lib, sym)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    )(fn_addr)

    st = fn(
        ctypes.c_void_p(x_dev),
        ctypes.c_void_p(y_dev),
        in_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(in_a.size),
        out_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(out_a.size),
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")
