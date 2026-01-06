"""
Conv2DTranspose CPU backend boundary helpers.

This module defines *boundary-layer* functions that bridge KeyDNN infrastructure
`Tensor` objects to CPU NumPy conv2d-transpose kernels and back.

Purpose
-------
The conv2d-transpose compute kernels (`conv2d_transpose_forward_cpu`,
`conv2d_transpose_backward_cpu`) operate on `np.ndarray`. To keep higher-level
autograd and operator code NumPy-free, all Tensor <-> NumPy conversion is
localized to this module.

Design notes
------------
- These helpers perform *no* shape inference, parameter validation, or autograd
  graph wiring. They only:
  1) convert `Tensor` inputs to NumPy,
  2) call the CPU kernel,
  3) wrap returned arrays into new `Tensor` objects on the same device.
- Outputs are created with explicit `requires_grad` flags supplied by the caller
  (typically the differentiable conv2d-transpose operation).

Backend constraints
-------------------
- CPU-only: the underlying kernels are NumPy implementations.
- The returned tensors use `copy_from_numpy`, so the boundary remains the single
  place where NumPy arrays touch Tensor storage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..tensor._tensor import Tensor
from .conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)


# -----------------------------------------------------------------------------
# Backend boundary helpers (CPU NumPy kernels)
# Keep all NumPy conversion localized here.
# -----------------------------------------------------------------------------
def conv2d_transpose_forward_cpu_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    out_requires_grad: bool,
) -> Tensor:
    """
    Run the CPU conv2d-transpose forward kernel using `Tensor` inputs.

    This helper converts `Tensor` inputs to NumPy arrays, calls the underlying
    CPU kernel, then wraps the result back into a new `Tensor`.

    Parameters
    ----------
    x : Tensor
        Input tensor (NCHW). Must be CPU-backed for conversion via `to_numpy()`.
    w : Tensor
        Weight tensor (IOHW for transpose conv2d). Must be CPU-backed.
    b : Optional[Tensor]
        Optional bias tensor of shape (C_out,). If provided, must be CPU-backed.
    stride : Tuple[int, int]
        Transposed convolution stride as (stride_h, stride_w).
    padding : Tuple[int, int]
        Transposed convolution padding as (pad_h, pad_w).
    output_padding : Tuple[int, int]
        Additional output padding as (op_h, op_w). This is the same concept as
        PyTorch's output_padding: it only affects the output shape, and must be
        < stride per dimension (validated by higher layers / kernel).
    out_requires_grad : bool
        Whether the returned output tensor should track gradients.

    Returns
    -------
    Tensor
        Output tensor produced by the conv2d-transpose forward kernel. The output
        lives on `x.device` and has `requires_grad=out_requires_grad`.

    Notes
    -----
    - This function is the designated Tensor <-> NumPy bridge for conv2d-transpose
      forward. Higher-level conv2d-transpose/autograd code should remain NumPy-free.
    - No autograd `Context` is attached here; that is handled by the differentiable
      conv2d-transpose operation wrapper.
    """
    x_np: np.ndarray = x.to_numpy()
    w_np: np.ndarray = w.to_numpy()
    b_np: Optional[np.ndarray] = None if b is None else b.to_numpy()

    y_np: np.ndarray = conv2d_transpose_forward_cpu(
        x_np,
        w_np,
        b_np,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    out = Tensor(
        shape=y_np.shape,
        device=x.device,
        requires_grad=out_requires_grad,
        ctx=None,
    )
    out.copy_from_numpy(y_np)  # boundary copy
    return out


def conv2d_transpose_backward_cpu_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    grad_out: Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Run the CPU conv2d-transpose backward kernel using `Tensor` inputs.

    This helper converts `Tensor` inputs and `grad_out` to NumPy arrays, calls the
    underlying CPU backward kernel, then wraps the returned gradients into new
    non-differentiable `Tensor` objects.

    Parameters
    ----------
    x : Tensor
        Forward input tensor used in the conv2d-transpose operation. Must be CPU-backed.
    w : Tensor
        Forward weight tensor (IOHW) used in the conv2d-transpose operation. Must be CPU-backed.
    b : Optional[Tensor]
        Forward bias tensor of shape (C_out,), or None. Must be CPU-backed when provided.
    grad_out : Tensor
        Gradient with respect to the conv2d-transpose output (dL/dY). Must be CPU-backed.
    stride : Tuple[int, int]
        Stride used in the forward pass.
    padding : Tuple[int, int]
        Padding used in the forward pass.
    output_padding : Tuple[int, int]
        Output padding used in the forward pass.

    Returns
    -------
    Tuple[Tensor, Tensor, Optional[Tensor]]
        (grad_x, grad_w, grad_b) where:
        - grad_x : Tensor
            Gradient with respect to `x` (dL/dX), shape matches `x`.
        - grad_w : Tensor
            Gradient with respect to `w` (dL/dW), shape matches `w`.
        - grad_b : Optional[Tensor]
            Gradient with respect to `b` (dL/db), or None if `b` is None.

    Notes
    -----
    - Returned gradient tensors have `requires_grad=False` because they are
      terminal results of a backward kernel.
    - This function performs no accumulation into `.grad` fields; the autograd
      engine is responsible for routing and accumulation.
    - This function is the designated Tensor <-> NumPy bridge for conv2d-transpose backward.
    """
    x_np: np.ndarray = x.to_numpy()
    w_np: np.ndarray = w.to_numpy()
    b_np: Optional[np.ndarray] = None if b is None else b.to_numpy()
    go_np: np.ndarray = grad_out.to_numpy()

    gx_np, gw_np, gb_np = conv2d_transpose_backward_cpu(
        x_np,
        w_np,
        b_np,
        go_np,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    grad_x = Tensor(shape=gx_np.shape, device=x.device, requires_grad=False, ctx=None)
    grad_x.copy_from_numpy(gx_np)

    grad_w = Tensor(shape=gw_np.shape, device=w.device, requires_grad=False, ctx=None)
    grad_w.copy_from_numpy(gw_np)

    grad_b = None
    if gb_np is not None:
        grad_b = Tensor(
            shape=gb_np.shape,
            device=b.device if b is not None else x.device,
            requires_grad=False,
            ctx=None,
        )
        grad_b.copy_from_numpy(gb_np)

    return grad_x, grad_w, grad_b
