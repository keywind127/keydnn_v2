"""
CPU-based naive Conv2D kernels for KeyDNN.

This module provides reference implementations of 2D convolution forward and
backward passes using NumPy on the CPU. These kernels are intentionally written
in a clear, explicit manner (nested Python loops) to prioritize correctness and
pedagogical clarity over performance.

Design goals
------------
- Serve as a correctness baseline for Conv2D operations
- Enable unit testing of higher-level abstractions (Conv2dFn, Conv2d module)
- Remain backend-agnostic with respect to autograd and Parameter handling

Non-goals
---------
- High performance (no im2col, GEMM, or vectorization)
- GPU or CUDA support
- Advanced features such as dilation, groups, or asymmetric padding

Tensor layout
-------------
All tensors follow the NCHW layout:

- N: batch size
- C: channels
- H: height
- W: width
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    """
    Normalize an integer or pair into a 2-tuple.

    Parameters
    ----------
    v : int or tuple[int, int]
        A scalar value or a 2D pair.

    Returns
    -------
    tuple[int, int]
        A normalized (height, width) pair.

    Notes
    -----
    This helper is commonly used for convolution hyperparameters such as
    `stride` and `padding`, which may be specified as either a single integer
    or a tuple.
    """
    return v if isinstance(v, tuple) else (v, v)


def conv2d_forward_cpu(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
) -> np.ndarray:
    """
    Compute the forward pass of a 2D convolution (CPU, NumPy).

    This function performs a naive 2D convolution using explicit loops over
    batch, channels, and spatial dimensions.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C_in, H, W).
    w : np.ndarray
        Convolution kernel weights of shape (C_out, C_in, K_h, K_w).
    b : Optional[np.ndarray]
        Optional bias of shape (C_out,). If None, no bias is added.
    stride : int or tuple[int, int]
        Convolution stride. If an integer is provided, the same stride is used
        for both spatial dimensions.
    padding : int or tuple[int, int]
        Zero-padding applied to the input tensor. If an integer is provided,
        symmetric padding is applied to both height and width.

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C_out, H_out, W_out), where:

        - H_out = floor((H + 2 * padding_h - K_h) / stride_h) + 1
        - W_out = floor((W + 2 * padding_w - K_w) / stride_w) + 1

    Raises
    ------
    ValueError
        If the number of input channels in `x` does not match the kernel.

    Notes
    -----
    - This implementation assumes NCHW layout.
    - All computations are performed in NumPy using float32.
    - This kernel is intended for correctness and testing, not performance.
    """
    import warnings

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")

    if b is not None:
        if b.ndim != 1 or b.shape[0] != C_out:
            raise ValueError(f"bias shape mismatch: expected ({C_out},), got {b.shape}")

    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    def _conv2d_loops(
        x_pad_: np.ndarray,
        w_: np.ndarray,
        b_: Optional[np.ndarray],
        y_: np.ndarray,
        N_: int,
        C_out_: int,
        H_out_: int,
        W_out_: int,
        K_h_: int,
        K_w_: int,
        s_h_: int,
        s_w_: int,
    ) -> None:
        for n in range(N_):
            for co in range(C_out_):
                for i in range(H_out_):
                    h0 = i * s_h_
                    for j in range(W_out_):
                        w0 = j * s_w_
                        patch = x_pad_[n, :, h0 : h0 + K_h_, w0 : w0 + K_w_]
                        y_[n, co, i, j] = np.sum(patch * w_[co])
                if b_ is not None:
                    y_[n, co, :, :] += b_[co]

    # Fast path: call the native C++ kernel via ctypes (float32/float64 only).
    # If native kernel is unavailable or dtype is unsupported, fall back to
    # the original NumPy reference loop for correctness and dtype preservation.
    try:
        if (
            x_pad.dtype in (np.float32, np.float64)
            and w.dtype == x_pad.dtype
            and y.dtype == x_pad.dtype
        ):
            from ..native.python.conv2d_ctypes import (
                load_keydnn_native,
                conv2d_forward_f32_ctypes,
                conv2d_forward_f64_ctypes,
            )

            lib = load_keydnn_native()

            if x_pad.dtype == np.float32:
                b32 = (
                    None
                    if b is None
                    else (
                        b if b.dtype == np.float32 else b.astype(np.float32, copy=False)
                    )
                )
                w32 = w if w.dtype == np.float32 else w.astype(np.float32, copy=False)

                conv2d_forward_f32_ctypes(
                    lib,
                    x_pad=x_pad,
                    w=w32,
                    b=b32,
                    y=y,
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                )
                return y

            if x_pad.dtype == np.float64:
                b64 = (
                    None
                    if b is None
                    else (
                        b if b.dtype == np.float64 else b.astype(np.float64, copy=False)
                    )
                )
                w64 = w if w.dtype == np.float64 else w.astype(np.float64, copy=False)

                conv2d_forward_f64_ctypes(
                    lib,
                    x_pad=x_pad,
                    w=w64,
                    b=b64,
                    y=y,
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                )
                return y
    except OSError as e:
        warnings.warn(
            "KeyDNN native conv2d library could not be loaded; "
            "falling back to NumPy reference implementation. "
            f"Reason: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reference path (original Python loop) for non-float32/float64 or when native is unavailable.
    _conv2d_loops(
        x_pad_=x_pad,
        w_=w,
        b_=b,
        y_=y,
        N_=N,
        C_out_=C_out,
        H_out_=H_out,
        W_out_=W_out,
        K_h_=K_h,
        K_w_=K_w,
        s_h_=s_h,
        s_w_=s_w,
    )

    return y


def conv2d_backward_cpu(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    grad_out: np.ndarray,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute the backward pass of a 2D convolution (CPU, NumPy).

    Given the gradient with respect to the output, this function computes
    gradients with respect to the input, weights, and optional bias.

    Parameters
    ----------
    x : np.ndarray
        Original input tensor of shape (N, C_in, H, W).
    w : np.ndarray
        Convolution kernel weights of shape (C_out, C_in, K_h, K_w).
    b : Optional[np.ndarray]
        Bias tensor of shape (C_out,), or None if no bias was used.
    grad_out : np.ndarray
        Gradient with respect to the output tensor, of shape
        (N, C_out, H_out, W_out).
    stride : int or tuple[int, int]
        Convolution stride used in the forward pass.
    padding : int or tuple[int, int]
        Padding used in the forward pass.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        A tuple containing:

        - grad_x : np.ndarray
            Gradient with respect to the input, shape (N, C_in, H, W).
        - grad_w : np.ndarray
            Gradient with respect to the weights, shape (C_out, C_in, K_h, K_w).
        - grad_b : Optional[np.ndarray]
            Gradient with respect to the bias, shape (C_out,), or None if
            no bias was provided.

    Raises
    ------
    ValueError
        If the number of input channels does not match the kernel.

    Notes
    -----
    - Padding is applied to the input gradient and removed before returning.
    - Bias gradients are computed by summing over batch and spatial dimensions.
    - This implementation mirrors the forward kernel for correctness and clarity.
    """
    import warnings

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    _, _, H_out, W_out = grad_out.shape

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    grad_x_pad = np.zeros_like(x_pad)
    grad_w = np.zeros_like(w)

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    def _conv2d_backward_loops(
        x_pad_: np.ndarray,
        w_: np.ndarray,
        grad_out_: np.ndarray,
        grad_x_pad_: np.ndarray,
        grad_w_: np.ndarray,
        N_: int,
        C_out_: int,
        H_out_: int,
        W_out_: int,
        K_h_: int,
        K_w_: int,
        s_h_: int,
        s_w_: int,
    ) -> None:
        for n in range(N_):
            for co in range(C_out_):
                for i in range(H_out_):
                    h0 = i * s_h_
                    for j in range(W_out_):
                        w0 = j * s_w_
                        go = grad_out_[n, co, i, j]
                        grad_w_[co] += go * x_pad_[n, :, h0 : h0 + K_h_, w0 : w0 + K_w_]
                        grad_x_pad_[n, :, h0 : h0 + K_h_, w0 : w0 + K_w_] += go * w_[co]

    # Fast path: call the native C++ kernel via ctypes (float32/float64 only).
    # If native kernel is unavailable or dtype is unsupported, fall back to
    # the original NumPy reference loop for correctness and dtype preservation.
    try:
        if (
            x_pad.dtype in (np.float32, np.float64)
            and w.dtype == x_pad.dtype
            and grad_out.dtype == x_pad.dtype
            and grad_x_pad.dtype == x_pad.dtype
            and grad_w.dtype == x_pad.dtype
        ):
            from ..native.python.conv2d_ctypes import (
                load_keydnn_native,
                conv2d_backward_f32_ctypes,
                conv2d_backward_f64_ctypes,
            )

            lib = load_keydnn_native()

            if x_pad.dtype == np.float32:
                conv2d_backward_f32_ctypes(
                    lib,
                    x_pad=x_pad,
                    w=w,
                    grad_out=grad_out,
                    grad_x_pad=grad_x_pad,
                    grad_w=grad_w,
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                )
                grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]
                return grad_x, grad_w, grad_b

            if x_pad.dtype == np.float64:
                conv2d_backward_f64_ctypes(
                    lib,
                    x_pad=x_pad,
                    w=w,
                    grad_out=grad_out,
                    grad_x_pad=grad_x_pad,
                    grad_w=grad_w,
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    H_out=H_out,
                    W_out=W_out,
                    K_h=K_h,
                    K_w=K_w,
                    s_h=s_h,
                    s_w=s_w,
                )
                grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]
                return grad_x, grad_w, grad_b
    except OSError as e:
        warnings.warn(
            "KeyDNN native conv2d library could not be loaded; "
            "falling back to NumPy reference implementation. "
            f"Reason: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reference path (original Python loop) for non-float32/float64 or when native is unavailable.
    _conv2d_backward_loops(
        x_pad_=x_pad,
        w_=w,
        grad_out_=grad_out,
        grad_x_pad_=grad_x_pad,
        grad_w_=grad_w,
        N_=N,
        C_out_=C_out,
        H_out_=H_out,
        W_out_=W_out,
        K_h_=K_h,
        K_w_=K_w,
        s_h_=s_h,
        s_w_=s_w,
    )

    grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]
    return grad_x, grad_w, grad_b
