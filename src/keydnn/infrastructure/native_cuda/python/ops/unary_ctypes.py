"""
ctypes bindings for elementwise CUDA arithmetic kernels.

This module provides low-level ctypes wrappers around KeyDNN's native CUDA
elementwise math kernels (e.g., exp, multiply, scalar multiply), dynamically
resolving exported symbols from the compiled CUDA shared library at runtime.

Design notes
------------
- This module is Windows-specific and relies on `GetProcAddress` to resolve
  kernel entry points from the loaded CUDA DLL.
- All functions operate directly on raw CUDA device pointers (ints).
- No memory allocation, shape inference, or broadcasting is performed here.
  Those responsibilities belong to higher-level Tensor or ops layers.
- Error handling is strict: missing symbols or non-zero kernel return codes
  raise RuntimeError immediately.

Supported dtypes
----------------
- np.float32
- np.float64

This module is intended to be used internally by KeyDNN's CUDA ops layer and
should not be called directly by end users.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import numpy as np

from ..avgpool2d_ctypes import load_keydnn_cuda_native

_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    """
    Resolve the address of an exported symbol from a native CUDA DLL.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    sym : str
        Name of the exported symbol to resolve.

    Returns
    -------
    int
        Integer address of the resolved function pointer.

    Raises
    ------
    RuntimeError
        If the symbol cannot be found in the DLL.
    """
    addr = _GetProcAddress(lib._handle, sym.encode("ascii"))
    v = int(ctypes.cast(addr, ctypes.c_void_p).value or 0)
    if v == 0:
        raise RuntimeError(f"Native DLL missing symbol: {sym}")
    return v


def exp_cuda(lib, *, x_dev: int, y_dev: int, numel: int, dtype: np.dtype) -> None:
    """
    Elementwise exponential on CUDA: y = exp(x).

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    x_dev : int
        Device pointer to input tensor.
    y_dev : int
        Device pointer to output tensor.
    numel : int
        Number of elements.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    TypeError
        If dtype is not supported.
    RuntimeError
        If the CUDA kernel returns a non-zero status.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_exp_f32"
        arg_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_exp_f64"
        arg_t = ctypes.c_double
    else:
        raise TypeError(f"exp_cuda supports float32/float64 only, got {dtype}")

    addr = _get_proc_addr(lib, sym)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(arg_t), ctypes.POINTER(arg_t), ctypes.c_int
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(numel)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def mul_cuda(
    lib, *, a_dev: int, b_dev: int, y_dev: int, numel: int, dtype: np.dtype
) -> None:
    """
    Elementwise multiply on CUDA: y = a * b.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    a_dev : int
        Device pointer to first input tensor.
    b_dev : int
        Device pointer to second input tensor.
    y_dev : int
        Device pointer to output tensor.
    numel : int
        Number of elements.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    TypeError
        If dtype is not supported.
    RuntimeError
        If the CUDA kernel returns a non-zero status.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_mul_f32"
        arg_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_mul_f64"
        arg_t = ctypes.c_double
    else:
        raise TypeError(f"mul_cuda supports float32/float64 only, got {dtype}")

    addr = _get_proc_addr(lib, sym)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(numel)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def mul_scalar_cuda(
    lib, *, a_dev: int, alpha: float, y_dev: int, numel: int, dtype: np.dtype
) -> None:
    """
    Elementwise scalar multiply on CUDA: y = a * alpha.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    a_dev : int
        Device pointer to input tensor.
    alpha : float
        Scalar multiplier.
    y_dev : int
        Device pointer to output tensor.
    numel : int
        Number of elements.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    TypeError
        If dtype is not supported.
    RuntimeError
        If the CUDA kernel returns a non-zero status.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_mul_scalar_f32"
        arg_t = ctypes.c_float
        scalar_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_mul_scalar_f64"
        arg_t = ctypes.c_double
        scalar_t = ctypes.c_double
    else:
        raise TypeError(f"mul_scalar_cuda supports float32/float64 only, got {dtype}")

    addr = _get_proc_addr(lib, sym)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        scalar_t,
        ctypes.POINTER(arg_t),
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            scalar_t(alpha),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(numel)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def mul_inplace_cuda(
    lib, *, a_dev: int, b_dev: int, numel: int, dtype: np.dtype
) -> None:
    """
    Elementwise in-place multiply on CUDA: a *= b.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    a_dev : int
        Device pointer to LHS tensor (modified in-place).
    b_dev : int
        Device pointer to RHS tensor (read-only).
    numel : int
        Number of elements.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    TypeError
        If dtype is not supported.
    RuntimeError
        If the CUDA kernel returns a non-zero status.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_mul_inplace_f32"
        arg_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_mul_inplace_f64"
        arg_t = ctypes.c_double
    else:
        raise TypeError(f"mul_inplace_cuda supports float32/float64 only, got {dtype}")

    addr = _get_proc_addr(lib, sym)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(numel)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def mul_scalar_inplace_cuda(
    lib, *, a_dev: int, alpha: float, numel: int, dtype: np.dtype
) -> None:
    """
    Elementwise scalar in-place multiply on CUDA: a *= alpha.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded CUDA shared library.
    a_dev : int
        Device pointer to tensor (modified in-place).
    alpha : float
        Scalar multiplier.
    numel : int
        Number of elements.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    TypeError
        If dtype is not supported.
    RuntimeError
        If the CUDA kernel returns a non-zero status.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_mul_scalar_inplace_f32"
        arg_t = ctypes.c_float
        scalar_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_mul_scalar_inplace_f64"
        arg_t = ctypes.c_double
        scalar_t = ctypes.c_double
    else:
        raise TypeError(
            f"mul_scalar_inplace_cuda supports float32/float64 only, got {dtype}"
        )

    addr = _get_proc_addr(lib, sym)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        scalar_t,
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            scalar_t(alpha),
            ctypes.c_int(int(numel)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")
