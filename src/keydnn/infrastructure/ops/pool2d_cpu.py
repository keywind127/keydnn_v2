"""
CPU reference implementations for 2D pooling operations (NumPy backend).

This module provides **naive, readable, and correct** NumPy-based
implementations of common 2D pooling operations for tensors in **NCHW**
layout. These functions serve as:

- A correctness reference for optimized backends (e.g., CUDA)
- The numerical ground truth for unit tests
- The backend kernels used by autograd `Function` implementations

Implemented pooling variants
-----------------------------
- MaxPool2D (forward + backward)
- AveragePool2D (forward + backward)
- GlobalAveragePool2D (forward + backward)

Design notes
------------
- These implementations favor clarity over performance.
- Padding semantics are explicit and carefully chosen:
  - MaxPool uses `-inf` padding so padded values never win.
  - AvgPool uses zero padding and averages over the full kernel area.
- All operations assume **NCHW** layout.
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
        Scalar or pair value.

    Returns
    -------
    tuple[int, int]
        Normalized (v, v) if scalar, otherwise the original tuple.
    """
    return v if isinstance(v, tuple) else (v, v)


def _out_hw(
    H: int, W: int, k: Tuple[int, int], s: Tuple[int, int], p: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Compute output spatial dimensions for a 2D pooling operation.

    Parameters
    ----------
    H, W : int
        Input height and width.
    k : tuple[int, int]
        Kernel size (k_h, k_w).
    s : tuple[int, int]
        Stride (s_h, s_w).
    p : tuple[int, int]
        Padding (p_h, p_w).

    Returns
    -------
    tuple[int, int]
        Output height and width (H_out, W_out).
    """
    k_h, k_w = k
    s_h, s_w = s
    p_h, p_w = p
    H_out = (H + 2 * p_h - k_h) // s_h + 1
    W_out = (W + 2 * p_w - k_w) // s_w + 1
    return H_out, W_out


def maxpool2d_forward_cpu(
    x: np.ndarray,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive MaxPool2D forward pass (CPU, NumPy) for NCHW tensors.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C, H, W).
    kernel_size : int or tuple[int, int]
        Pooling window size.
    stride : int or tuple[int, int] or None, optional
        Pooling stride. If None, defaults to `kernel_size`.
    padding : int or tuple[int, int], optional
        Zero-padding applied to spatial dimensions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        y :
            Output tensor of shape (N, C, H_out, W_out).
        argmax_idx :
            Integer array of shape (N, C, H_out, W_out) storing the flattened
            index into the padded input where the maximum was selected.

    Notes
    -----
    - Padding is performed with `-inf` so padded values never become maxima.
    - `argmax_idx` is required for correct gradient routing in the backward pass.
    """
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x.shape
    k_h, k_w = k
    p_h, p_w = p

    H_out, W_out = _out_hw(H, W, k, s, p)

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    y = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    argmax_idx = np.empty((N, C, H_out, W_out), dtype=np.int64)

    s_h, s_w = s
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    flat_idx = int(np.argmax(patch))
                    y[n, c, i, j] = patch.reshape(-1)[flat_idx]

                    ph = flat_idx // k_w
                    pw = flat_idx % k_w
                    h = h0 + ph
                    w_ = w0 + pw
                    argmax_idx[n, c, i, j] = h * W_pad + w_

    return y, argmax_idx


def maxpool2d_backward_cpu(
    grad_out: np.ndarray,
    argmax_idx: np.ndarray,
    *,
    x_shape: tuple[int, int, int, int],
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    """
    Naive MaxPool2D backward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient with respect to output, shape (N, C, H_out, W_out).
    argmax_idx : np.ndarray
        Argmax indices returned by the forward pass.
    x_shape : tuple[int, int, int, int]
        Original input shape (N, C, H, W).
    kernel_size, stride, padding
        Same hyperparameters used during the forward pass.

    Returns
    -------
    np.ndarray
        Gradient with respect to input, shape (N, C, H, W).

    Notes
    -----
    - Gradients are routed only to the input locations that won the max
      operation during the forward pass.
    - Padding regions do not receive gradients.
    """
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x_shape
    p_h, p_w = p
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    grad_x_pad = np.zeros((N, C, H_pad, W_pad), dtype=grad_out.dtype)

    H_out, W_out = grad_out.shape[2], grad_out.shape[3]
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    idx = int(argmax_idx[n, c, i, j])
                    h = idx // W_pad
                    w_ = idx % W_pad
                    grad_x_pad[n, c, h, w_] += grad_out[n, c, i, j]

    return grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]


def avgpool2d_forward_cpu(
    x: np.ndarray,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    """
    Naive AvgPool2D forward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C, H, W).
    kernel_size, stride, padding
        Pooling hyperparameters.

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C, H_out, W_out).

    Notes
    -----
    - Zero-padding is applied.
    - The average is computed over the full kernel area (k_h * k_w),
      including padded values.
    """
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x.shape
    k_h, k_w = k
    p_h, p_w = p

    H_out, W_out = _out_hw(H, W, k, s, p)

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    y = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    s_h, s_w = s
    denom = float(k_h * k_w)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    y[n, c, i, j] = np.sum(patch) / denom

    return y


def avgpool2d_backward_cpu(
    grad_out: np.ndarray,
    *,
    x_shape: tuple[int, int, int, int],
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    """
    Naive AvgPool2D backward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient with respect to output.
    x_shape : tuple[int, int, int, int]
        Original input shape.
    kernel_size, stride, padding
        Pooling hyperparameters.

    Returns
    -------
    np.ndarray
        Gradient with respect to input.

    Notes
    -----
    - Gradients are distributed uniformly over each pooling window.
    - Padding regions are ignored after accumulation.
    """
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x_shape
    k_h, k_w = k
    p_h, p_w = p
    s_h, s_w = s

    H_out, W_out = grad_out.shape[2], grad_out.shape[3]
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    grad_x_pad = np.zeros((N, C, H_pad, W_pad), dtype=grad_out.dtype)
    denom = float(k_h * k_w)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    go = grad_out[n, c, i, j] / denom
                    grad_x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w] += go

    return grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]


def global_avgpool2d_forward_cpu(x: np.ndarray) -> np.ndarray:
    """
    Global average pooling forward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C, H, W).

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C, 1, 1).

    Notes
    -----
    - Each channel is reduced to its mean over spatial dimensions.
    """
    return x.mean(axis=(2, 3), keepdims=True).astype(x.dtype, copy=False)


def global_avgpool2d_backward_cpu(
    grad_out: np.ndarray, *, x_shape: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Global average pooling backward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient with respect to output, shape (N, C, 1, 1).
    x_shape : tuple[int, int, int, int]
        Original input shape.

    Returns
    -------
    np.ndarray
        Gradient with respect to input, shape (N, C, H, W).

    Notes
    -----
    - The gradient is distributed evenly across all spatial locations.
    """
    N, C, H, W = x_shape
    scale = 1.0 / float(H * W)
    return (np.ones((N, C, H, W), dtype=grad_out.dtype) * grad_out * scale).astype(
        grad_out.dtype, copy=False
    )
