"""
Conv2D CPU backend boundary helpers.

This module defines *boundary-layer* functions that bridge KeyDNN infrastructure
`Tensor` objects to CPU NumPy conv2d kernels and back.

Purpose
-------
The conv2d compute kernels (`conv2d_forward_cpu`, `conv2d_backward_cpu`) operate
on `np.ndarray`. To keep higher-level autograd and operator code NumPy-free,
all Tensor <-> NumPy conversion is localized to this module.

Design notes
------------
- These helpers perform *no* shape inference, parameter validation, or autograd
  graph wiring. They only:
  1) convert `Tensor` inputs to NumPy,
  2) call the CPU kernel,
  3) wrap returned arrays into new `Tensor` objects on the same device.
- Outputs are created with explicit `requires_grad` flags supplied by the caller
  (typically the differentiable conv2d operation).

Backend constraints
-------------------
- CPU-only: the underlying kernels are NumPy implementations.
- The returned tensors use `copy_from_numpy`, so the boundary remains the single
  place where NumPy arrays touch Tensor storage.
"""

from typing import Optional, Tuple

import numpy as np

from ..tensor._tensor import Tensor
from .conv2d_cpu import (
    conv2d_forward_cpu,
    conv2d_backward_cpu,
)


# -----------------------------------------------------------------------------
# Backend boundary helpers (CPU NumPy kernels)
# Keep all NumPy conversion localized here.
# -----------------------------------------------------------------------------
def conv2d_forward_cpu_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    out_requires_grad: bool,
) -> Tensor:
    """
    Run the CPU conv2d forward kernel using `Tensor` inputs.

    This helper converts `Tensor` inputs to NumPy arrays, calls the underlying
    CPU kernel, then wraps the result back into a new `Tensor`.

    Parameters
    ----------
    x : Tensor
        Input tensor (typically NCHW layout). Must be CPU-backed for conversion
        via `to_numpy()`.
    w : Tensor
        Weight/filter tensor (kernel parameters). Must be CPU-backed.
    b : Optional[Tensor]
        Optional bias tensor. If provided, must be CPU-backed.
    stride : Tuple[int, int]
        Convolution stride as (stride_h, stride_w).
    padding : Tuple[int, int]
        Zero-padding as (pad_h, pad_w).
    out_requires_grad : bool
        Whether the returned output tensor should track gradients. This is
        decided by the caller (e.g., based on parents' `requires_grad`).

    Returns
    -------
    Tensor
        Output tensor produced by the conv2d forward kernel. The output lives on
        `x.device` and has `requires_grad=out_requires_grad`.

    Notes
    -----
    - This function is the designated Tensor <-> NumPy bridge for conv2d forward.
      Higher-level conv2d/autograd code should remain NumPy-free.
    - No autograd `Context` is attached here; that is handled by the differentiable
      conv2d operation wrapper.
    """
    # Kernel still expects NumPy arrays (boundary)
    x_np: np.ndarray = x.to_numpy()
    w_np: np.ndarray = w.to_numpy()
    b_np: Optional[np.ndarray] = None if b is None else b.to_numpy()

    y_np: np.ndarray = conv2d_forward_cpu(
        x_np, w_np, b_np, stride=stride, padding=padding
    )

    out = Tensor(
        shape=y_np.shape, device=x.device, requires_grad=out_requires_grad, ctx=None
    )
    out.copy_from_numpy(y_np)  # boundary copy
    return out


def conv2d_backward_cpu_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    grad_out: Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Run the CPU conv2d backward kernel using `Tensor` inputs.

    This helper converts `Tensor` inputs and `grad_out` to NumPy arrays, calls the
    underlying CPU backward kernel, then wraps the returned gradients into new
    non-differentiable `Tensor` objects.

    Parameters
    ----------
    x : Tensor
        Forward input tensor used in the conv2d operation. Must be CPU-backed.
    w : Tensor
        Forward weight tensor used in the conv2d operation. Must be CPU-backed.
    b : Optional[Tensor]
        Forward bias tensor used in the conv2d operation (if any). Must be CPU-backed
        when provided.
    grad_out : Tensor
        Gradient with respect to the conv2d output (dL/dY). Must be CPU-backed.
    stride : Tuple[int, int]
        Convolution stride as (stride_h, stride_w), matching the forward call.
    padding : Tuple[int, int]
        Zero-padding as (pad_h, pad_w), matching the forward call.

    Returns
    -------
    Tuple[Tensor, Tensor, Optional[Tensor]]
        (grad_x, grad_w, grad_b) where:
        - grad_x : Tensor
            Gradient with respect to `x` (dL/dX).
        - grad_w : Tensor
            Gradient with respect to `w` (dL/dW).
        - grad_b : Optional[Tensor]
            Gradient with respect to `b` (dL/db), or None if `b` is None.

    Notes
    -----
    - Returned gradient tensors have `requires_grad=False` because they are
      terminal results of a backward kernel.
    - This function performs no accumulation into `.grad` fields; the autograd
      engine is responsible for routing and accumulation.
    - This function is the designated Tensor <-> NumPy bridge for conv2d backward.
    """
    x_np: np.ndarray = x.to_numpy()
    w_np: np.ndarray = w.to_numpy()
    b_np: Optional[np.ndarray] = None if b is None else b.to_numpy()
    go_np: np.ndarray = grad_out.to_numpy()

    gx_np, gw_np, gb_np = conv2d_backward_cpu(
        x_np, w_np, b_np, go_np, stride=stride, padding=padding
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
