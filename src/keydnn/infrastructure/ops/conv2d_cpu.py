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
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")

    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, :, h0 : h0 + K_h, w0 : w0 + K_w]
                    y[n, co, i, j] = np.sum(patch * w[co])
            if b is not None:
                y[n, co, :, :] += b[co]

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

    grad_x_pad = np.zeros_like(x_pad)
    grad_w = np.zeros_like(w)

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    go = grad_out[n, co, i, j]
                    grad_w[co] += go * x_pad[n, :, h0 : h0 + K_h, w0 : w0 + K_w]
                    grad_x_pad[n, :, h0 : h0 + K_h, w0 : w0 + K_w] += go * w[co]

    grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]
    return grad_x, grad_w, grad_b
