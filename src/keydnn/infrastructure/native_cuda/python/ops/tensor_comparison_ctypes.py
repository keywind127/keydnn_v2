"""
Dynamic ctypes bindings for KeyDNN v2 CUDA comparison kernels (Windows).

This module provides low-level Python wrappers for CUDA elementwise comparison
operations implemented in the KeyDNN native DLL. It follows the project's
Windows `GetProcAddress` lookup pattern to:

- resolve exported function addresses by symbol name at runtime
- build `ctypes.CFUNCTYPE` callables pointing to those addresses
- cache `(dll_handle, symbol) -> callable` to avoid repeated CFUNCTYPE creation

Supported operations
--------------------
Elementwise (same shape):
- gt: a >  b
- ge: a >= b
- lt: a <  b
- le: a <= b
- eq: a == b
- ne: a != b

Scalar (same shape vs scalar):
- gt_scalar: a >  alpha
- ge_scalar: a >= alpha
- lt_scalar: a <  alpha
- le_scalar: a <= alpha
- eq_scalar: a == alpha
- ne_scalar: a != alpha

Dtype conventions
-----------------
- Inputs support float32 and float64:
    *_f32 for float32 inputs
    *_f64 for float64 inputs
- All comparison ops write float32 outputs (mask) with values {1.0, 0.0}
  even when inputs are float64.

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


def _select_cmp_sym_and_in_ctype(sym_base: str, dtype: np.dtype) -> tuple[str, type]:
    """
    Map a base comparison symbol name and dtype to a concrete symbol and input ctype.

    Parameters
    ----------
    sym_base : str
        Base exported symbol name, e.g. "keydnn_cuda_ge", "keydnn_cuda_eq_scalar".
        The wrapper appends a suffix based on dtype.
    dtype : np.dtype
        Input dtype. Only float32 and float64 are supported.

    Returns
    -------
    (str, type)
        - symbol: full exported symbol name (e.g. f"{sym_base}_f32")
        - input_ctype: ctypes scalar type for input elements (float/double)

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


def _call_cmp_binary(
    lib,
    *,
    sym_base: str,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Call an elementwise binary comparison kernel producing float32 mask output.

    Native signature (by dtype):
      - *_f32(const float* a, const float* b, float* y, int64 n)
      - *_f64(const double* a, const double* b, float* y, int64 n)

    Parameters
    ----------
    sym_base : str
        Base symbol (without _f32/_f64), e.g. "keydnn_cuda_ge".
    a_dev, b_dev : int
        Device pointers to inputs (uintptr_t).
    y_dev : int
        Device pointer to output float32 buffer (uintptr_t).
    n : int
        Number of elements.
    dtype : np.dtype
        Input dtype (float32/float64).
    """
    lib = lib or load_keydnn_cuda_native()
    sym, a_t = _select_cmp_sym_and_in_ctype(sym_base, dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(a_t),  # a
        ctypes.POINTER(a_t),  # b
        ctypes.POINTER(ctypes.c_float),  # y (float32 mask)
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


def _call_cmp_scalar(
    lib,
    *,
    sym_base: str,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Call an elementwise scalar comparison kernel producing float32 mask output.

    Native signature (by dtype):
      - *_f32(const float* a, float alpha, float* y, int64 n)
      - *_f64(const double* a, double alpha, float* y, int64 n)

    Parameters
    ----------
    sym_base : str
        Base symbol (without _f32/_f64), e.g. "keydnn_cuda_ge_scalar".
    a_dev : int
        Device pointer to input (uintptr_t).
    alpha : float
        Python scalar; cast to float/double depending on dtype.
    y_dev : int
        Device pointer to output float32 buffer (uintptr_t).
    n : int
        Number of elements.
    dtype : np.dtype
        Input dtype (float32/float64).
    """
    lib = lib or load_keydnn_cuda_native()
    sym, a_t = _select_cmp_sym_and_in_ctype(sym_base, dtype)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(a_t),  # a
        a_t,  # alpha
        ctypes.POINTER(ctypes.c_float),  # y (float32 mask)
        ctypes.c_int64,  # n
    )
    fn = _get_fn(lib, sym, FN)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(a_dev)), ctypes.POINTER(a_t)),
            a_t(alpha),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(int(n)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


# ============================================================
# Public API: elementwise (binary) comparisons
# ============================================================


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
    Elementwise greater-than: y[i] = float(a[i] > b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_gt_f32(const float* a, const float* b, float* y, int64 n)
      - keydnn_cuda_gt_f64(const double* a, const double* b, float* y, int64 n)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_gt",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def ge_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Elementwise greater-or-equal: y[i] = float(a[i] >= b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_ge_f32(...)
      - keydnn_cuda_ge_f64(...)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_ge",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def lt_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Elementwise less-than: y[i] = float(a[i] < b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_lt_f32(...)
      - keydnn_cuda_lt_f64(...)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_lt",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def le_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Elementwise less-or-equal: y[i] = float(a[i] <= b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_le_f32(...)
      - keydnn_cuda_le_f64(...)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_le",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def eq_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Elementwise equality: y[i] = float(a[i] == b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_eq_f32(...)
      - keydnn_cuda_eq_f64(...)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_eq",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def ne_cuda(
    lib,
    *,
    a_dev: int,
    b_dev: int,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Elementwise not-equal: y[i] = float(a[i] != b[i]) (float32 mask output).
    Calls:
      - keydnn_cuda_ne_f32(...)
      - keydnn_cuda_ne_f64(...)
    """
    _call_cmp_binary(
        lib,
        sym_base="keydnn_cuda_ne",
        a_dev=a_dev,
        b_dev=b_dev,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


# ============================================================
# Public API: scalar comparisons
# ============================================================


def gt_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar greater-than: y[i] = float(a[i] > alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_gt_scalar_f32(...)
      - keydnn_cuda_gt_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_gt_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def ge_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar greater-or-equal: y[i] = float(a[i] >= alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_ge_scalar_f32(...)
      - keydnn_cuda_ge_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_ge_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def lt_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar less-than: y[i] = float(a[i] < alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_lt_scalar_f32(...)
      - keydnn_cuda_lt_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_lt_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def le_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar less-or-equal: y[i] = float(a[i] <= alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_le_scalar_f32(...)
      - keydnn_cuda_le_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_le_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def eq_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar equality: y[i] = float(a[i] == alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_eq_scalar_f32(...)
      - keydnn_cuda_eq_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_eq_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )


def ne_scalar_cuda(
    lib,
    *,
    a_dev: int,
    alpha: float,
    y_dev: int,
    n: int,
    dtype: np.dtype,
) -> None:
    """
    Scalar not-equal: y[i] = float(a[i] != alpha) (float32 mask output).
    Calls:
      - keydnn_cuda_ne_scalar_f32(...)
      - keydnn_cuda_ne_scalar_f64(...)
    """
    _call_cmp_scalar(
        lib,
        sym_base="keydnn_cuda_ne_scalar",
        a_dev=a_dev,
        alpha=alpha,
        y_dev=y_dev,
        n=n,
        dtype=dtype,
    )
