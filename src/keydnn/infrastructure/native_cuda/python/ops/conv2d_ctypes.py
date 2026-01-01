"""
infrastructure/native_cuda/python/ops/conv2d_ctypes.py

ctypes bindings for KeyDNN's CUDA Conv2D kernels on Windows.

This module provides thin wrappers around exported functions in the KeyDNN CUDA
native DLL (e.g., `keydnn_cuda_conv2d_forward_f32/f64`). It resolves symbols via
Win32 `GetProcAddress`, builds appropriately-typed `ctypes` call signatures, and
invokes the native kernels using raw device pointers (integers).

Design goals
------------
- Keep Python overhead low: no data marshaling beyond pointer casts and scalar args.
- Be explicit about memory layout: NCHW for activations and OIHW for weights.
- Fail fast: any missing symbol or non-zero native status raises an exception.

Notes
-----
- This module assumes the caller manages CUDA device selection and memory
  allocation. Pointers are passed as integer device addresses.
- The forward wrapper supports an optional bias pointer (pass None for no bias).
- The backward wrapper computes gradients for x_pad and w; grad_b is expected to
  be computed in Python (e.g., reduction over grad_out).

Platform
--------
- Windows only (uses `ctypes.windll.kernel32.GetProcAddress`).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import numpy as np

from ..avgpool2d_ctypes import load_keydnn_cuda_native  # consistent with your tree

_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    """
    Resolve a symbol address from a loaded CUDA native DLL.

    This is a small Win32 helper that wraps `GetProcAddress` and normalizes the
    returned value into a Python `int` suitable for constructing a `ctypes`
    callable via `ctypes.CFUNCTYPE`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    sym : str
        Exported symbol name (ASCII).

    Returns
    -------
    int
        Function pointer address.

    Raises
    ------
    RuntimeError
        If the symbol does not exist in the DLL.
    """
    addr = _GetProcAddress(lib._handle, sym.encode("ascii"))
    v = int(ctypes.cast(addr, ctypes.c_void_p).value or 0)
    if v == 0:
        raise RuntimeError(f"Native DLL missing symbol: {sym}")
    return v


def _dtype_to_sym_and_arg(dtype: np.dtype, *, base: str) -> tuple[str, type]:
    """
    Resolve the exported symbol suffix and ctypes scalar type for a NumPy dtype.

    The KeyDNN CUDA DLL exports separate entry points per floating dtype
    (currently float32 and float64). This helper maps a requested dtype to:

    - the full exported symbol name (e.g., f"{base}_f32")
    - the corresponding ctypes scalar type (e.g., ctypes.c_float)

    Parameters
    ----------
    dtype : np.dtype
        Requested floating dtype.
    base : str
        Base symbol name without dtype suffix (e.g., "keydnn_cuda_conv2d_forward").

    Returns
    -------
    tuple[str, type]
        (exported_symbol_name, ctypes_scalar_type)

    Raises
    ------
    TypeError
        If `dtype` is not float32 or float64.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return f"{base}_f32", ctypes.c_float
    if dtype == np.float64:
        return f"{base}_f64", ctypes.c_double
    raise TypeError(f"conv2d CUDA supports float32/float64 only, got {dtype}")


def conv2d_forward_cuda(
    lib,
    *,
    x_pad_dev: int,
    w_dev: int,
    b_dev: int | None,
    y_dev: int,
    N: int,
    C_in: int,
    H_pad: int,
    W_pad: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    dtype: np.dtype,
) -> None:
    """
    Run the CUDA Conv2D forward kernel via the KeyDNN native DLL.

    This function is a thin `ctypes` wrapper that:
    1) selects the correct exported symbol based on `dtype`,
    2) constructs a typed callable with the expected signature, and
    3) invokes the kernel using raw device pointers.

    Native entry points
    -------------------
    - keydnn_cuda_conv2d_forward_f32
    - keydnn_cuda_conv2d_forward_f64

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    x_pad_dev : int
        Device pointer to x_pad, shape (N, C_in, H_pad, W_pad), NCHW.
    w_dev : int
        Device pointer to w, shape (C_out, C_in, K_h, K_w), OIHW.
    b_dev : int | None
        Device pointer to bias, shape (C_out,). May be None or 0 for "no bias".
    y_dev : int
        Device pointer to output y, shape (N, C_out, H_out, W_out), NCHW.
    N, C_in, H_pad, W_pad : int
        Padded input dimensions.
    C_out, H_out, W_out : int
        Output dimensions.
    K_h, K_w : int
        Kernel (filter) spatial dimensions.
    s_h, s_w : int
        Stride along height/width.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    RuntimeError
        If the exported symbol is missing, or the native kernel returns a
        non-zero status code.
    TypeError
        If `dtype` is not float32 or float64.

    Notes
    -----
    - Assumes `x_pad_dev` points to already-padded input; padding is handled by
      the caller.
    - The bias pointer is optional. Passing None results in a null pointer being
      passed to the native kernel.
    - Returns None on success.
    """
    sym, arg_t = _dtype_to_sym_and_arg(dtype, base="keydnn_cuda_conv2d_forward")
    addr = _get_proc_addr(lib, sym)

    # int fn(const T* x_pad, const T* w, const T* b, T* y,
    #        int N, int C_in, int H_pad, int W_pad, int C_out, int H_out, int W_out,
    #        int K_h, int K_w, int s_h, int s_w)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    fn = FN(addr)

    b_ptr_int = 0 if (b_dev is None) else int(b_dev)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_pad_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(w_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(b_ptr_int), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(N)),
            ctypes.c_int(int(C_in)),
            ctypes.c_int(int(H_pad)),
            ctypes.c_int(int(W_pad)),
            ctypes.c_int(int(C_out)),
            ctypes.c_int(int(H_out)),
            ctypes.c_int(int(W_out)),
            ctypes.c_int(int(K_h)),
            ctypes.c_int(int(K_w)),
            ctypes.c_int(int(s_h)),
            ctypes.c_int(int(s_w)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def conv2d_backward_cuda(
    lib,
    *,
    x_pad_dev: int,
    w_dev: int,
    grad_out_dev: int,
    grad_x_pad_dev: int,
    grad_w_dev: int,
    N: int,
    C_in: int,
    H_pad: int,
    W_pad: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    dtype: np.dtype,
) -> None:
    """
    Run the CUDA Conv2D backward kernel (grad_x_pad and grad_w) via the native DLL.

    This function wraps the KeyDNN CUDA backward kernel and computes gradients:
    - grad_x_pad: gradient w.r.t. padded input
    - grad_w:     gradient w.r.t. weights

    Native entry points
    -------------------
    - keydnn_cuda_conv2d_backward_f32
    - keydnn_cuda_conv2d_backward_f64

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    x_pad_dev : int
        Device pointer to x_pad, shape (N, C_in, H_pad, W_pad), NCHW.
    w_dev : int
        Device pointer to w, shape (C_out, C_in, K_h, K_w), OIHW.
    grad_out_dev : int
        Device pointer to grad_out, shape (N, C_out, H_out, W_out), NCHW.
    grad_x_pad_dev : int
        Device pointer to grad_x_pad, shape (N, C_in, H_pad, W_pad), NCHW.
        Must be zero-initialized by caller (accumulated in-place).
    grad_w_dev : int
        Device pointer to grad_w, shape (C_out, C_in, K_h, K_w), OIHW.
        Must be zero-initialized by caller (accumulated in-place).
    N, C_in, H_pad, W_pad : int
        Padded input dimensions.
    C_out, H_out, W_out : int
        Output dimensions / grad_out dimensions.
    K_h, K_w : int
        Kernel (filter) spatial dimensions.
    s_h, s_w : int
        Stride along height/width.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    RuntimeError
        If the exported symbol is missing, or the native kernel returns a
        non-zero status code.
    TypeError
        If `dtype` is not float32 or float64.

    Notes
    -----
    - This wrapper does not compute grad_b. A common approach is to compute it
      in Python by reducing `grad_out` across (N, H_out, W_out).
    - The output gradient buffers are expected to be pre-zeroed because the
      native implementation accumulates into them.
    - Returns None on success.
    """
    sym, arg_t = _dtype_to_sym_and_arg(dtype, base="keydnn_cuda_conv2d_backward")
    addr = _get_proc_addr(lib, sym)

    # int fn(const T* x_pad, const T* w, const T* grad_out,
    #        T* grad_x_pad, T* grad_w,
    #        int N, int C_in, int H_pad, int W_pad, int C_out, int H_out, int W_out,
    #        int K_h, int K_w, int s_h, int s_w)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_pad_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(w_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_out_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_x_pad_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_w_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(N)),
            ctypes.c_int(int(C_in)),
            ctypes.c_int(int(H_pad)),
            ctypes.c_int(int(W_pad)),
            ctypes.c_int(int(C_out)),
            ctypes.c_int(int(H_out)),
            ctypes.c_int(int(W_out)),
            ctypes.c_int(int(K_h)),
            ctypes.c_int(int(K_w)),
            ctypes.c_int(int(s_h)),
            ctypes.c_int(int(s_w)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


class _Conv2dCudaExports:
    """
    Symbol name and dtype conventions for KeyDNN Conv2D CUDA exports.

    This class is purely organizational/documentational and is not used at
    runtime. It captures the naming convention used by the native DLL and the
    supported dtype set to make the wrapper surface easier to understand.

    Attributes
    ----------
    forward_base : str
        Base name of the forward kernel symbol (dtype suffix appended).
    backward_base : str
        Base name of the backward kernel symbol (dtype suffix appended).
    supported_dtypes : tuple[np.dtype, ...]
        Dtypes supported by the wrappers and expected to be exported by the DLL.
    """

    forward_base: str = "keydnn_cuda_conv2d_forward"
    backward_base: str = "keydnn_cuda_conv2d_backward"
    supported_dtypes: tuple[np.dtype, ...] = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )


# Optional convenience helpers (mirrors how you often use load_keydnn_cuda_native)


def load_and_conv2d_forward_cuda(**kwargs) -> None:
    """
    Load the KeyDNN CUDA native library and run `conv2d_forward_cuda`.

    This convenience wrapper is useful for call sites that do not already cache
    the native library handle. For performance-sensitive paths, prefer calling
    `load_keydnn_cuda_native()` once and reusing the returned `ctypes.CDLL`.

    Parameters
    ----------
    **kwargs
        Forwarded directly to `conv2d_forward_cuda` (excluding `lib`).

    Returns
    -------
    None
    """
    lib = load_keydnn_cuda_native()
    conv2d_forward_cuda(lib, **kwargs)


def load_and_conv2d_backward_cuda(**kwargs) -> None:
    """
    Load the KeyDNN CUDA native library and run `conv2d_backward_cuda`.

    This convenience wrapper is useful for call sites that do not already cache
    the native library handle. For performance-sensitive paths, prefer calling
    `load_keydnn_cuda_native()` once and reusing the returned `ctypes.CDLL`.

    Parameters
    ----------
    **kwargs
        Forwarded directly to `conv2d_backward_cuda` (excluding `lib`).

    Returns
    -------
    None
    """
    lib = load_keydnn_cuda_native()
    conv2d_backward_cuda(lib, **kwargs)
