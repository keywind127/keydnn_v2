"""
CPU-based naive ConvTranspose2D (transpose convolution) kernels for KeyDNN.

This module provides reference implementations of 2D transpose convolution
forward and backward passes using NumPy on the CPU. These kernels are written
with explicit Python loops to prioritize correctness and pedagogical clarity.

Design goals
------------
- Serve as a correctness baseline for ConvTranspose2D operations
- Enable unit testing of higher-level abstractions (ConvTranspose2dFn, module)
- Provide a fast-path to the native C++ CPU kernels via ctypes (float32/float64)

Non-goals
---------
- High performance (no im2col, GEMM, or vectorization)
- GPU or CUDA support
- Advanced features such as dilation, groups

Tensor layout
-------------
All tensors follow NCHW for activations:

- x: (N, C_in, H_in, W_in)
- y: (N, C_out, H_out, W_out)

Weight layout for transpose-conv is IOHW:

- w: (C_in, C_out, K_h, K_w)
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    """
    Normalize an integer or pair into a 2-tuple.
    """
    return v if isinstance(v, tuple) else (v, v)


def conv2d_transpose_forward_cpu(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    output_padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    """
    Compute the forward pass of a 2D transpose convolution (CPU, NumPy).

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C_in, H_in, W_in).
    w : np.ndarray
        Transposed convolution weights of shape (C_in, C_out, K_h, K_w) (IOHW).
    b : Optional[np.ndarray]
        Optional bias of shape (C_out,), or None.
    stride : int or tuple[int, int]
        Stride used for upsampling.
    padding : int or tuple[int, int]
        Padding (subtracted from the scattered output indices).
    output_padding : int or tuple[int, int], default 0
        Additional size added to one side of each spatial dimension in the output.

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C_out, H_out, W_out), where:

        H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h
        W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w

    Raises
    ------
    ValueError
        If channel dimensions are inconsistent or output sizes are invalid.

    Notes
    -----
    - Scatter-style accumulation:
        y[n, co, hi*s_h + kh - p_h, wi*s_w + kw - p_w] += x[n, ci, hi, wi] * w[ci, co, kh, kw]
    - Bias is added after accumulation: y[n, co] += b[co]
    """
    import warnings

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")

    if b is not None:
        if b.ndim != 1 or b.shape[0] != C_out:
            raise ValueError(f"bias shape mismatch: expected ({C_out},), got {b.shape}")

    if s_h <= 0 or s_w <= 0:
        raise ValueError(f"stride must be positive, got stride=({s_h},{s_w})")
    if op_h < 0 or op_w < 0:
        raise ValueError(f"output_padding must be non-negative, got ({op_h},{op_w})")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"output_padding must be < stride per dim, got output_padding=({op_h},{op_w}), stride=({s_h},{s_w})"
        )

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"invalid output size: H_out={H_out}, W_out={W_out}")

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    def _transpose_loops(
        x_: np.ndarray,
        w_: np.ndarray,
        b_: Optional[np.ndarray],
        y_: np.ndarray,
        N_: int,
        C_in_: int,
        H_in_: int,
        W_in_: int,
        C_out_: int,
        H_out_: int,
        W_out_: int,
        K_h_: int,
        K_w_: int,
        s_h_: int,
        s_w_: int,
        p_h_: int,
        p_w_: int,
    ) -> None:
        for n in range(N_):
            for ci in range(C_in_):
                for hi in range(H_in_):
                    base_oh = hi * s_h_ - p_h_
                    for wi in range(W_in_):
                        base_ow = wi * s_w_ - p_w_
                        xv = x_[n, ci, hi, wi]
                        for co in range(C_out_):
                            for kh in range(K_h_):
                                oh = base_oh + kh
                                if oh < 0 or oh >= H_out_:
                                    continue
                                for kw in range(K_w_):
                                    ow = base_ow + kw
                                    if ow < 0 or ow >= W_out_:
                                        continue
                                    y_[n, co, oh, ow] += xv * w_[ci, co, kh, kw]
            if b_ is not None:
                for co in range(C_out_):
                    y_[n, co, :, :] += b_[co]

    # Fast path: call the native C++ kernel via ctypes (float32/float64 only).
    # If native kernel is unavailable or dtype is unsupported, fall back to
    # the original NumPy reference loop for correctness and dtype preservation.
    try:
        if (
            op_h == 0
            and op_w == 0
            and x.dtype in (np.float32, np.float64)
            and w.dtype == x.dtype
            and y.dtype == x.dtype
        ):
            from ..native.python.conv2d_transpose_ctypes import (
                load_keydnn_native,
                conv2d_transpose_forward_f32_ctypes,
                conv2d_transpose_forward_f64_ctypes,
            )

            lib = load_keydnn_native()

            if x.dtype == np.float32:
                b32 = (
                    None
                    if b is None
                    else (
                        b if b.dtype == np.float32 else b.astype(np.float32, copy=False)
                    )
                )
                w32 = w if w.dtype == np.float32 else w.astype(np.float32, copy=False)

                # Kernel is additive (scatter) -> ensure y is zero-initialized (it is).
                conv2d_transpose_forward_f32_ctypes(
                    lib,
                    x=x,
                    w=w32,
                    b=b32,
                    y=y,
                    N=N,
                    C_in=C_in,
                    H_in=H_in,
                    W_in=W_in,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                    pad_h=p_h,
                    pad_w=p_w,
                )
                return y

            if x.dtype == np.float64:
                b64 = (
                    None
                    if b is None
                    else (
                        b if b.dtype == np.float64 else b.astype(np.float64, copy=False)
                    )
                )
                w64 = w if w.dtype == np.float64 else w.astype(np.float64, copy=False)

                conv2d_transpose_forward_f64_ctypes(
                    lib,
                    x=x,
                    w=w64,
                    b=b64,
                    y=y,
                    N=N,
                    C_in=C_in,
                    H_in=H_in,
                    W_in=W_in,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                    pad_h=p_h,
                    pad_w=p_w,
                )
                return y
    except OSError as e:
        warnings.warn(
            "KeyDNN native conv2d_transpose library could not be loaded; "
            "falling back to NumPy reference implementation. "
            f"Reason: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reference path (original Python loop) for non-float32/float64 or when native is unavailable.
    _transpose_loops(
        x_=x,
        w_=w,
        b_=b,
        y_=y,
        N_=N,
        C_in_=C_in,
        H_in_=H_in,
        W_in_=W_in,
        C_out_=C_out,
        H_out_=H_out,
        W_out_=W_out,
        K_h_=K_h,
        K_w_=K_w,
        s_h_=s_h,
        s_w_=s_w,
        p_h_=p_h,
        p_w_=p_w,
    )

    return y


def conv2d_transpose_backward_cpu(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    grad_out: np.ndarray,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    output_padding: int | Tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute the backward pass of a 2D transpose convolution (CPU, NumPy).

    Parameters
    ----------
    x : np.ndarray
        Original input tensor of shape (N, C_in, H_in, W_in).
    w : np.ndarray
        Weights of shape (C_in, C_out, K_h, K_w) (IOHW).
    b : Optional[np.ndarray]
        Bias tensor of shape (C_out,), or None if no bias was used.
    grad_out : np.ndarray
        Gradient w.r.t. output, shape (N, C_out, H_out, W_out).
    stride : int or tuple[int, int]
        Stride used in the forward pass.
    padding : int or tuple[int, int]
        Padding used in the forward pass.
    output_padding : int or tuple[int, int], default 0
        Output padding used in the forward pass.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        - grad_x : (N, C_in, H_in, W_in)
        - grad_w : (C_in, C_out, K_h, K_w)
        - grad_b : (C_out,) or None

    Notes
    -----
    - grad_b = sum(grad_out) over (N, H_out, W_out) if b is not None.
    - Native fast-path (ctypes) is used for float32/float64 only and currently
      supports output_padding == 0. Otherwise falls back to Python loops.
    """
    import warnings

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    if s_h <= 0 or s_w <= 0:
        raise ValueError(f"stride must be positive, got stride=({s_h},{s_w})")
    if op_h < 0 or op_w < 0:
        raise ValueError(f"output_padding must be non-negative, got ({op_h},{op_w})")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError(
            f"output_padding must be < stride per dim, got output_padding=({op_h},{op_w}), stride=({s_h},{s_w})"
        )

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if grad_out.shape != (N, C_out, H_out, W_out):
        raise ValueError(
            f"grad_out shape mismatch: expected {(N, C_out, H_out, W_out)}, got {grad_out.shape}"
        )

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    def _transpose_backward_loops(
        x_: np.ndarray,
        w_: np.ndarray,
        grad_out_: np.ndarray,
        grad_x_: np.ndarray,
        grad_w_: np.ndarray,
        N_: int,
        C_in_: int,
        H_in_: int,
        W_in_: int,
        C_out_: int,
        H_out_: int,
        W_out_: int,
        K_h_: int,
        K_w_: int,
        s_h_: int,
        s_w_: int,
        p_h_: int,
        p_w_: int,
    ) -> None:
        for n in range(N_):
            for ci in range(C_in_):
                for hi in range(H_in_):
                    base_oh = hi * s_h_ - p_h_
                    for wi in range(W_in_):
                        base_ow = wi * s_w_ - p_w_
                        xv = x_[n, ci, hi, wi]

                        for co in range(C_out_):
                            for kh in range(K_h_):
                                oh = base_oh + kh
                                if oh < 0 or oh >= H_out_:
                                    continue
                                for kw in range(K_w_):
                                    ow = base_ow + kw
                                    if ow < 0 or ow >= W_out_:
                                        continue
                                    go = grad_out_[n, co, oh, ow]
                                    grad_x_[n, ci, hi, wi] += go * w_[ci, co, kh, kw]
                                    grad_w_[ci, co, kh, kw] += xv * go

    # Fast path: call the native C++ kernel via ctypes (float32/float64 only).
    # Requires output_padding == 0 to match current native signature.
    try:
        if (
            op_h == 0
            and op_w == 0
            and x.dtype in (np.float32, np.float64)
            and w.dtype == x.dtype
            and grad_out.dtype == x.dtype
            and grad_x.dtype == x.dtype
            and grad_w.dtype == x.dtype
        ):
            from ..native.python.conv2d_transpose_ctypes import (
                load_keydnn_native,
                conv2d_transpose_backward_f32_ctypes,
                conv2d_transpose_backward_f64_ctypes,
            )

            lib = load_keydnn_native()

            if x.dtype == np.float32:
                conv2d_transpose_backward_f32_ctypes(
                    lib,
                    x=x,
                    w=w,
                    grad_out=grad_out,
                    grad_x=grad_x,
                    grad_w=grad_w,
                    N=N,
                    C_in=C_in,
                    H_in=H_in,
                    W_in=W_in,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                    pad_h=p_h,
                    pad_w=p_w,
                )
                return grad_x, grad_w, grad_b

            if x.dtype == np.float64:
                conv2d_transpose_backward_f64_ctypes(
                    lib,
                    x=x,
                    w=w,
                    grad_out=grad_out,
                    grad_x=grad_x,
                    grad_w=grad_w,
                    N=N,
                    C_in=C_in,
                    H_in=H_in,
                    W_in=W_in,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                    pad_h=p_h,
                    pad_w=p_w,
                )
                return grad_x, grad_w, grad_b
    except OSError as e:
        warnings.warn(
            "KeyDNN native conv2d_transpose library could not be loaded; "
            "falling back to NumPy reference implementation. "
            f"Reason: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reference path (original Python loop) for non-float32/float64 or when native is unavailable.
    _transpose_backward_loops(
        x_=x,
        w_=w,
        grad_out_=grad_out,
        grad_x_=grad_x,
        grad_w_=grad_w,
        N_=N,
        C_in_=C_in,
        H_in_=H_in,
        W_in_=W_in,
        C_out_=C_out,
        H_out_=H_out,
        W_out_=W_out,
        K_h_=K_h,
        K_w_=K_w,
        s_h_=s_h,
        s_w_=s_w,
        p_h_=p_h,
        p_w_=p_w,
    )

    return grad_x, grad_w, grad_b
