"""
infrastructure/native_cuda/python/ops/conv2d_transpose_ctypes.py

ctypes bindings for KeyDNN's CUDA ConvTranspose2D (Conv2D Transpose) kernels on Windows.

This module provides thin wrappers around exported functions in the KeyDNN CUDA
native DLL:

- keydnn_cuda_conv2d_transpose_forward_f32 / _f64
- keydnn_cuda_conv2d_transpose_backward_f32 / _f64

It resolves symbols via Win32 `GetProcAddress`, constructs typed `ctypes`
callables, and invokes the native kernels using raw CUDA device pointers
represented as Python integers.

Design goals
------------
- Minimal Python overhead: no marshaling beyond pointer casts and scalar args.
- Explicit layouts:
  - activations: NCHW (x, y, grad_out, grad_x)
  - weights:     IOHW (w, grad_w) for transpose conv: (C_in, C_out, K_h, K_w)
- Fail fast: missing symbol or non-zero native status raises.

Notes
-----
- All pointers passed to these wrappers are device pointers (ints).
- The CUDA forward kernel (as implemented in the provided .cu) *writes* y (not +=).
- The CUDA backward kernels *write* grad_x and grad_w (not +=).
- grad_b is not computed natively; Python side typically computes
  grad_b = sum(grad_out) over (N, H_out, W_out).

Platform
--------
- Windows only (uses `ctypes.windll.kernel32.GetProcAddress`).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import numpy as np

from ..avgpool2d_ctypes import load_keydnn_cuda_native  # keep tree-consistent

_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    """
    Resolve an exported symbol address from a loaded CUDA native DLL.

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
    Map NumPy dtype to (exported symbol name, ctypes scalar type).

    Supported
    ---------
    - np.float32 -> suffix _f32 -> ctypes.c_float
    - np.float64 -> suffix _f64 -> ctypes.c_double
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return f"{base}_f32", ctypes.c_float
    if dtype == np.float64:
        return f"{base}_f64", ctypes.c_double
    raise TypeError(f"conv2d_transpose CUDA supports float32/float64 only, got {dtype}")


def conv2d_transpose_forward_cuda(
    lib,
    *,
    x_dev: int,
    w_dev: int,
    b_dev: int | None,
    y_dev: int,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
    dtype: np.dtype,
) -> None:
    """
    Run the CUDA ConvTranspose2D forward kernel via the KeyDNN native DLL.

    Native entry points
    -------------------
    - keydnn_cuda_conv2d_transpose_forward_f32
    - keydnn_cuda_conv2d_transpose_forward_f64

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    x_dev : int
        Device pointer to x, shape (N, C_in, H_in, W_in), NCHW.
    w_dev : int
        Device pointer to w, shape (C_in, C_out, K_h, K_w), IOHW.
    b_dev : int | None
        Device pointer to bias, shape (C_out,), or None for no bias.
    y_dev : int
        Device pointer to y, shape (N, C_out, H_out, W_out), NCHW.
    N, C_in, H_in, W_in : int
        Input dimensions.
    C_out, H_out, W_out : int
        Output dimensions.
    K_h, K_w : int
        Kernel spatial dimensions.
    s_h, s_w : int
        Stride.
    pad_h, pad_w : int
        Transpose-conv padding (cropping offset) used by the native kernel.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    RuntimeError
        If the exported symbol is missing, or the native kernel returns non-zero.
    TypeError
        If dtype is unsupported.

    Notes
    -----
    - The provided CUDA implementation is gather-style and writes y (not +=).
      Allocate y normally; no need to zero for accumulation semantics.
    """
    sym, arg_t = _dtype_to_sym_and_arg(
        dtype, base="keydnn_cuda_conv2d_transpose_forward"
    )
    addr = _get_proc_addr(lib, sym)

    # int fn(const T* x, const T* w, const T* b, T* y,
    #        int N, int C_in, int H_in, int W_in,
    #        int C_out, int H_out, int W_out,
    #        int K_h, int K_w, int s_h, int s_w, int pad_h, int pad_w)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # x
        ctypes.POINTER(arg_t),  # w
        ctypes.POINTER(arg_t),  # b (nullable)
        ctypes.POINTER(arg_t),  # y
        ctypes.c_int,  # N
        ctypes.c_int,  # C_in
        ctypes.c_int,  # H_in
        ctypes.c_int,  # W_in
        ctypes.c_int,  # C_out
        ctypes.c_int,  # H_out
        ctypes.c_int,  # W_out
        ctypes.c_int,  # K_h
        ctypes.c_int,  # K_w
        ctypes.c_int,  # s_h
        ctypes.c_int,  # s_w
        ctypes.c_int,  # pad_h
        ctypes.c_int,  # pad_w
    )
    fn = FN(addr)

    b_ptr_int = 0 if (b_dev is None) else int(b_dev)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(w_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(b_ptr_int), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(N)),
            ctypes.c_int(int(C_in)),
            ctypes.c_int(int(H_in)),
            ctypes.c_int(int(W_in)),
            ctypes.c_int(int(C_out)),
            ctypes.c_int(int(H_out)),
            ctypes.c_int(int(W_out)),
            ctypes.c_int(int(K_h)),
            ctypes.c_int(int(K_w)),
            ctypes.c_int(int(s_h)),
            ctypes.c_int(int(s_w)),
            ctypes.c_int(int(pad_h)),
            ctypes.c_int(int(pad_w)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


def conv2d_transpose_backward_cuda(
    lib,
    *,
    x_dev: int,
    w_dev: int,
    grad_out_dev: int,
    grad_x_dev: int,
    grad_w_dev: int,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
    dtype: np.dtype,
) -> None:
    """
    Run the CUDA ConvTranspose2D backward kernel via the KeyDNN native DLL.

    Computes:
    - grad_x: gradient w.r.t x, shape (N, C_in, H_in, W_in)
    - grad_w: gradient w.r.t w, shape (C_in, C_out, K_h, K_w)

    Native entry points
    -------------------
    - keydnn_cuda_conv2d_transpose_backward_f32
    - keydnn_cuda_conv2d_transpose_backward_f64

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    x_dev : int
        Device pointer to x, shape (N, C_in, H_in, W_in), NCHW.
    w_dev : int
        Device pointer to w, shape (C_in, C_out, K_h, K_w), IOHW.
    grad_out_dev : int
        Device pointer to grad_out, shape (N, C_out, H_out, W_out), NCHW.
    grad_x_dev : int
        Device pointer to grad_x, shape (N, C_in, H_in, W_in), NCHW.
    grad_w_dev : int
        Device pointer to grad_w, shape (C_in, C_out, K_h, K_w), IOHW.
    N, C_in, H_in, W_in : int
        Input dimensions.
    C_out, H_out, W_out : int
        Output / grad_out dimensions.
    K_h, K_w : int
        Kernel spatial dimensions.
    s_h, s_w : int
        Stride.
    pad_h, pad_w : int
        Transpose-conv padding (cropping offset) used by the native kernel.
    dtype : np.dtype
        np.float32 or np.float64.

    Raises
    ------
    RuntimeError
        If the exported symbol is missing, or the native kernel returns non-zero.
    TypeError
        If dtype is unsupported.

    Notes
    -----
    - The provided CUDA backward implementation is gather-style and writes
      grad_x and grad_w (not +=). No need to pre-zero unless your caller expects
      accumulation semantics across multiple ops.
    - grad_b is not computed here.
    """
    sym, arg_t = _dtype_to_sym_and_arg(
        dtype, base="keydnn_cuda_conv2d_transpose_backward"
    )
    addr = _get_proc_addr(lib, sym)

    # int fn(const T* x, const T* w, const T* grad_out,
    #        T* grad_x, T* grad_w,
    #        int N, int C_in, int H_in, int W_in,
    #        int C_out, int H_out, int W_out,
    #        int K_h, int K_w, int s_h, int s_w, int pad_h, int pad_w)
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),  # x
        ctypes.POINTER(arg_t),  # w
        ctypes.POINTER(arg_t),  # grad_out
        ctypes.POINTER(arg_t),  # grad_x
        ctypes.POINTER(arg_t),  # grad_w
        ctypes.c_int,  # N
        ctypes.c_int,  # C_in
        ctypes.c_int,  # H_in
        ctypes.c_int,  # W_in
        ctypes.c_int,  # C_out
        ctypes.c_int,  # H_out
        ctypes.c_int,  # W_out
        ctypes.c_int,  # K_h
        ctypes.c_int,  # K_w
        ctypes.c_int,  # s_h
        ctypes.c_int,  # s_w
        ctypes.c_int,  # pad_h
        ctypes.c_int,  # pad_w
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(w_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_out_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(grad_w_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(N)),
            ctypes.c_int(int(C_in)),
            ctypes.c_int(int(H_in)),
            ctypes.c_int(int(W_in)),
            ctypes.c_int(int(C_out)),
            ctypes.c_int(int(H_out)),
            ctypes.c_int(int(W_out)),
            ctypes.c_int(int(K_h)),
            ctypes.c_int(int(K_w)),
            ctypes.c_int(int(s_h)),
            ctypes.c_int(int(s_w)),
            ctypes.c_int(int(pad_h)),
            ctypes.c_int(int(pad_w)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")


class _Conv2dTransposeCudaExports:
    """
    Symbol naming convention and supported dtypes for ConvTranspose2D CUDA exports.
    """

    forward_base: str = "keydnn_cuda_conv2d_transpose_forward"
    backward_base: str = "keydnn_cuda_conv2d_transpose_backward"
    supported_dtypes: tuple[np.dtype, ...] = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )


def load_and_conv2d_transpose_forward_cuda(**kwargs) -> None:
    """
    Load the KeyDNN CUDA native library and run `conv2d_transpose_forward_cuda`.

    For perf-sensitive call sites, prefer caching `load_keydnn_cuda_native()`.
    """
    lib = load_keydnn_cuda_native()
    conv2d_transpose_forward_cuda(lib, **kwargs)


def load_and_conv2d_transpose_backward_cuda(**kwargs) -> None:
    """
    Load the KeyDNN CUDA native library and run `conv2d_transpose_backward_cuda`.

    For perf-sensitive call sites, prefer caching `load_keydnn_cuda_native()`.
    """
    lib = load_keydnn_cuda_native()
    conv2d_transpose_backward_cuda(lib, **kwargs)
