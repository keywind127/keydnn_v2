"""
Dynamic ctypes bindings for KeyDNN v2 CUDA elementwise kernels (Windows).

This module provides low-level Python wrappers for a small set of CUDA
elementwise operations implemented in the KeyDNN native DLL. It uses a
Windows-specific `GetProcAddress` lookup pattern (matching the project's
`matmul_ctypes` style) to:

- resolve exported function addresses by symbol name at runtime
- build `ctypes.CFUNCTYPE` callables pointing to those addresses
- cache `(dll_handle, symbol) -> callable` to avoid repeated CFUNCTYPE creation

Supported operations
--------------------
Unary:
- neg: elementwise negation

Binary:
- add: elementwise addition
- sub: elementwise subtraction
- div: elementwise division

Compare:
- gt: elementwise greater-than (a > b), writing float32 outputs

Dtype conventions
-----------------
- Most kernels are dtype-specialized by suffix:
    *_f32 for float32 inputs/outputs
    *_f64 for float64 inputs/outputs
- `gt_cuda` always writes float32 outputs even when inputs are float64.
- All functions take device pointers as Python ints (`uintptr_t`) and an element
  count `n` (int64 on the native side).

Notes
-----
- These wrappers do not validate pointer validity, contiguity, or shapes; they
  assume the caller has already prepared correct device buffers.
- On non-zero native status codes, wrappers raise `RuntimeError` with the symbol
  name and status.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Tuple

import numpy as np

from ..avgpool2d_ctypes import load_keydnn_cuda_native

# Windows-only GetProcAddress pattern (matches your matmul_ctypes style)
_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    """
    Resolve a symbol address from a loaded Windows DLL.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded DLL handle. The underlying HMODULE is read from `lib._handle`.
    sym : str
        Exported symbol name to resolve (ASCII).

    Returns
    -------
    int
        The resolved function address as an integer.

    Raises
    ------
    RuntimeError
        If the symbol is not found (address resolves to 0).
    """
    addr = _GetProcAddress(lib._handle, sym.encode("ascii"))
    v = int(ctypes.cast(addr, ctypes.c_void_p).value or 0)
    if v == 0:
        raise RuntimeError(f"Native DLL missing symbol: {sym}")
    return v


# Cache (addr -> callable) so we don't rebuild CFUNCTYPE each call.
_FN_CACHE: dict[Tuple[int, str], ctypes._CFuncPtr] = {}


def _get_fn(lib: ctypes.CDLL, sym: str, fntype) -> ctypes._CFuncPtr:
    """
    Get (and cache) a ctypes callable for a resolved native function symbol.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native DLL handle.
    sym : str
        Exported symbol name.
    fntype : ctypes.CFUNCTYPE
        A CFUNCTYPE factory describing the native function signature.

    Returns
    -------
    ctypes._CFuncPtr
        Callable function pointer wrapper.

    Notes
    -----
    The cache key is `(int(lib._handle), sym)` so the same symbol name loaded
    from different DLL handles does not collide.
    """
    key = (int(lib._handle), sym)
    fn = _FN_CACHE.get(key)
    if fn is None:
        addr = _get_proc_addr(lib, sym)
        fn = fntype(addr)
        _FN_CACHE[key] = fn
    return fn


def _select_sym_and_ctype(sym_base: str, dtype: np.dtype) -> tuple[str, type]:
    """
    Map a base symbol name and dtype to a concrete symbol suffix and element ctype.

    Parameters
    ----------
    sym_base : str
        Base exported symbol name, e.g. "keydnn_cuda_add", "keydnn_cuda_neg".
        The wrapper appends a suffix based on dtype.
    dtype : np.dtype
        Input dtype. Only float32 and float64 are supported.

    Returns
    -------
    (str, type)
        - symbol: full exported symbol name (e.g. f"{sym_base}_f32")
        - element_ctype: ctypes scalar type for input/output elements

    Raises
    ------
    TypeError
        If `dtype` is not float32 or float64.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return f"{sym_base}_f32", ctypes.c_float
    if dtype == np.float64:
        return f"{sym_base}_f64", ctypes.c_double
    raise TypeError(f"{sym_base} supports float32/float64 only, got {dtype}")


# ----------------------------
# Unary: neg
# ----------------------------
def neg_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Compute elementwise negation on CUDA: `y[i] = -x[i]`.

    Parameters
    ----------
    lib : ctypes.CDLL | Any
        Loaded native DLL handle. If falsy/None, this function will load the
        default KeyDNN CUDA DLL via `load_keydnn_cuda_native()`.
    x_dev : int
        Device pointer (uintptr_t) to the input buffer `x`.
    y_dev : int
        Device pointer (uintptr_t) to the output buffer `y`.
    n : int
        Number of elements to process.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Calls
    -----
    - keydnn_cuda_neg_f32(const float* x, float* y, int64 n)
    - keydnn_cuda_neg_f64(const double* x, double* y, int64 n)

    Raises
    ------
    TypeError
        If dtype is unsupported.
    RuntimeError
        If the native function returns non-zero status or the symbol is missing.
    """
    lib = lib or load_keydnn_cuda_native()
    sym, arg_t = _select_sym_and_ctype("keydnn_cuda_neg", dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # x
        ctypes.POINTER(arg_t),  # y
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


# ----------------------------
# Binary: add/sub/div
# ----------------------------
def add_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Compute elementwise addition on CUDA: `y[i] = a[i] + b[i]`.

    Parameters
    ----------
    lib : ctypes.CDLL | Any
        Loaded native DLL handle. If falsy/None, this function will load the
        default KeyDNN CUDA DLL via `load_keydnn_cuda_native()`.
    a_dev : int
        Device pointer (uintptr_t) to input buffer `a`.
    b_dev : int
        Device pointer (uintptr_t) to input buffer `b`.
    y_dev : int
        Device pointer (uintptr_t) to output buffer `y`.
    n : int
        Number of elements to process.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Calls
    -----
    - keydnn_cuda_add_f32(const float* a, const float* b, float* y, int64 n)
    - keydnn_cuda_add_f64(const double* a, const double* b, double* y, int64 n)

    Raises
    ------
    TypeError
        If dtype is unsupported.
    RuntimeError
        If the native function returns non-zero status or the symbol is missing.
    """
    lib = lib or load_keydnn_cuda_native()
    sym, arg_t = _select_sym_and_ctype("keydnn_cuda_add", dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # a
        ctypes.POINTER(arg_t),  # b
        ctypes.POINTER(arg_t),  # y
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def sub_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Compute elementwise subtraction on CUDA: `y[i] = a[i] - b[i]`.

    Parameters
    ----------
    lib : ctypes.CDLL | Any
        Loaded native DLL handle. If falsy/None, this function will load the
        default KeyDNN CUDA DLL via `load_keydnn_cuda_native()`.
    a_dev : int
        Device pointer (uintptr_t) to input buffer `a`.
    b_dev : int
        Device pointer (uintptr_t) to input buffer `b`.
    y_dev : int
        Device pointer (uintptr_t) to output buffer `y`.
    n : int
        Number of elements to process.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Calls
    -----
    - keydnn_cuda_sub_f32(const float* a, const float* b, float* y, int64 n)
    - keydnn_cuda_sub_f64(const double* a, const double* b, double* y, int64 n)

    Raises
    ------
    TypeError
        If dtype is unsupported.
    RuntimeError
        If the native function returns non-zero status or the symbol is missing.
    """
    lib = lib or load_keydnn_cuda_native()
    sym, arg_t = _select_sym_and_ctype("keydnn_cuda_sub", dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # a
        ctypes.POINTER(arg_t),  # b
        ctypes.POINTER(arg_t),  # y
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def div_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Compute elementwise division on CUDA: `y[i] = a[i] / b[i]`.

    Parameters
    ----------
    lib : ctypes.CDLL | Any
        Loaded native DLL handle. If falsy/None, this function will load the
        default KeyDNN CUDA DLL via `load_keydnn_cuda_native()`.
    a_dev : int
        Device pointer (uintptr_t) to input buffer `a`.
    b_dev : int
        Device pointer (uintptr_t) to input buffer `b`.
    y_dev : int
        Device pointer (uintptr_t) to output buffer `y`.
    n : int
        Number of elements to process.
    dtype : np.dtype
        Input/output dtype. Must be float32 or float64.

    Calls
    -----
    - keydnn_cuda_div_f32(const float* a, const float* b, float* y, int64 n)
    - keydnn_cuda_div_f64(const double* a, const double* b, double* y, int64 n)

    Raises
    ------
    TypeError
        If dtype is unsupported.
    RuntimeError
        If the native function returns non-zero status or the symbol is missing.
    """
    lib = lib or load_keydnn_cuda_native()
    sym, arg_t = _select_sym_and_ctype("keydnn_cuda_div", dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # a
        ctypes.POINTER(arg_t),  # b
        ctypes.POINTER(arg_t),  # y
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


# ----------------------------
# Compare: gt (output is float32 always)
# ----------------------------
def gt_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Compute elementwise greater-than on CUDA: `y[i] = float(a[i] > b[i])`.

    Parameters
    ----------
    lib : ctypes.CDLL | Any
        Loaded native DLL handle. If falsy/None, this function will load the
        default KeyDNN CUDA DLL via `load_keydnn_cuda_native()`.
    a_dev : int
        Device pointer (uintptr_t) to input buffer `a`.
    b_dev : int
        Device pointer (uintptr_t) to input buffer `b`.
    y_dev : int
        Device pointer (uintptr_t) to output buffer `y`.

        Important: `y_dev` must point to **float32** device memory of length `n`,
        even when `dtype` is float64.
    n : int
        Number of elements to process.
    dtype : np.dtype
        Input dtype for `a` and `b`. Must be float32 or float64.

    Calls
    -----
    - keydnn_cuda_gt_f32(const float* a, const float* b, float* y, int64 n)
    - keydnn_cuda_gt_f64(const double* a, const double* b, float* y, int64 n)

    Raises
    ------
    TypeError
        If dtype is unsupported.
    RuntimeError
        If the native function returns non-zero status or the symbol is missing.
    """
    lib = lib or load_keydnn_cuda_native()
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        sym = "keydnn_cuda_gt_f32"
        a_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_gt_f64"
        a_t = ctypes.c_double
    else:
        raise TypeError(f"gt_cuda supports float32/float64 only, got {dtype}")

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(a_t),  # a
        ctypes.POINTER(a_t),  # b
        ctypes.POINTER(ctypes.c_float),  # y (float32)
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(a_t)),
            ctypes.cast(ctypes.c_void_p(int(b_dev)), ctypes.POINTER(a_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")
