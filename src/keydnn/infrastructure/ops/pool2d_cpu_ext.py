"""
2D pooling primitives with device-dispatch boundaries.

This module defines small, backend-facing wrapper functions for 2D pooling
operators (max pooling, average pooling, and global average pooling).

Responsibilities
----------------
- Normalize user-facing hyperparameters (`kernel_size`, `stride`, `padding`)
  to canonical `(H, W)` pairs via `_pair`.
- Call the concrete CPU kernel implementations that operate on NumPy arrays.
- Convert NumPy outputs back into framework `Tensor` objects via
  `Tensor._from_numpy`, keeping the NumPy boundary inside infrastructure code.

Notes
-----
- These wrappers currently target the CPU (NumPy) backend only.
- A future device dispatch layer (CPU/CUDA) can be added here without changing
  higher-level module code.
- Pooling kernels are treated as non-differentiable at the Tensor level here;
  autograd integration (Context attachment) is expected to happen in higher-level
  ops/modules that call these primitives.
"""

from __future__ import annotations
from typing import Optional, Tuple

from ..tensor._tensor import Tensor
from .pool2d_cpu import (
    _pair,
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


def maxpool2d_forward(
    x: Tensor,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> tuple[Tensor, object]:
    """
    Compute the forward pass of 2D max pooling (CPU-only wrapper).

    Parameters
    ----------
    x : Tensor
        Input tensor. This wrapper expects `x.to_numpy()` to be available
        (i.e., CPU tensor in the current implementation).
    kernel_size : int | tuple[int, int]
        Pooling window size. If an int is provided, it is interpreted as
        `(kernel_size, kernel_size)`.
    stride : Optional[int | tuple[int, int]], optional
        Stride between pooling windows. If None, defaults to `kernel_size`.
        If an int is provided, it is interpreted as `(stride, stride)`.
    padding : int | tuple[int, int], optional
        Implicit zero-padding added to both sides. If an int is provided,
        it is interpreted as `(padding, padding)`. Defaults to 0.

    Returns
    -------
    tuple[Tensor, object]
        A pair `(y, argmax_idx)` where:
        - `y` is the pooled output tensor (created via `Tensor._from_numpy`)
        - `argmax_idx` is backend-defined metadata required by the backward pass
          to route gradients to the max locations.

    Notes
    -----
    - This function performs parameter normalization via `_pair`.
    - The output tensor is created with `requires_grad=False`; higher-level
      autograd ops are expected to attach graph context if needed.
    """
    # device dispatch lives here later (cpu/cuda)
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    # still uses NumPy kernel internally, but boundary is hidden via Tensor
    y_np, argmax_idx = maxpool2d_forward_cpu(
        x.to_numpy(), kernel_size=k, stride=s, padding=p
    )
    y = Tensor._from_numpy(y_np, device=x.device, requires_grad=False)
    return y, argmax_idx


def maxpool2d_backward(
    grad_out: Tensor,
    *,
    argmax_idx: object,
    x_shape,
    kernel_size,
    stride,
    padding,
) -> Tensor:
    """
    Compute the backward pass of 2D max pooling (CPU-only wrapper).

    Parameters
    ----------
    grad_out : Tensor
        Gradient with respect to the maxpool output.
    argmax_idx : object
        Backend-defined indices/metadata produced by `maxpool2d_forward`.
        Used to scatter gradients back to the corresponding max positions.
    x_shape : Any
        Shape of the original input tensor `x` used in the forward pass.
        The exact type is backend-dependent (commonly a tuple[int, ...]).
    kernel_size : Any
        Canonical kernel size used in the forward pass (typically (kh, kw)).
    stride : Any
        Canonical stride used in the forward pass (typically (sh, sw)).
    padding : Any
        Canonical padding used in the forward pass (typically (ph, pw)).

    Returns
    -------
    Tensor
        Gradient with respect to the input `x`, as a new tensor created via
        `Tensor._from_numpy`.

    Notes
    -----
    This wrapper expects the caller to pass in the exact metadata used during
    forward. Validation of these values is delegated to the CPU kernel.
    """
    gx_np = maxpool2d_backward_cpu(
        grad_out.to_numpy(),
        argmax_idx,
        x_shape=x_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return Tensor._from_numpy(gx_np, device=grad_out.device, requires_grad=False)


def avgpool2d_forward(
    x: Tensor,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> Tensor:
    """
    Compute the forward pass of 2D average pooling (CPU-only wrapper).

    Parameters
    ----------
    x : Tensor
        Input tensor (CPU tensor in the current implementation).
    kernel_size : int | tuple[int, int]
        Pooling window size. If an int is provided, it is interpreted as
        `(kernel_size, kernel_size)`.
    stride : Optional[int | tuple[int, int]], optional
        Stride between pooling windows. If None, defaults to `kernel_size`.
    padding : int | tuple[int, int], optional
        Implicit zero-padding added to both sides. Defaults to 0.

    Returns
    -------
    Tensor
        The pooled output tensor created via `Tensor._from_numpy`.

    Notes
    -----
    This wrapper normalizes parameters via `_pair` and calls the NumPy CPU
    kernel. The output tensor is created with `requires_grad=False`.
    """
    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    # CPU kernel consumes numpy; boundary is via Tensor API
    y_np = avgpool2d_forward_cpu(x.to_numpy(), kernel_size=k, stride=s, padding=p)
    return Tensor._from_numpy(y_np, device=x.device, requires_grad=False)


def avgpool2d_backward(
    grad_out: Tensor,
    *,
    x_shape,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tensor:
    """
    Compute the backward pass of 2D average pooling (CPU-only wrapper).

    Parameters
    ----------
    grad_out : Tensor
        Gradient with respect to the avgpool output.
    x_shape : Any
        Shape of the original input tensor `x` used in the forward pass.
    kernel_size : tuple[int, int]
        Pooling window size used in the forward pass.
    stride : tuple[int, int]
        Stride used in the forward pass.
    padding : tuple[int, int]
        Padding used in the forward pass.

    Returns
    -------
    Tensor
        Gradient with respect to the input `x`, created via `Tensor._from_numpy`.
    """
    gx_np = avgpool2d_backward_cpu(
        grad_out.to_numpy(),
        x_shape=x_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return Tensor._from_numpy(gx_np, device=grad_out.device, requires_grad=False)


def global_avgpool2d_forward(x: Tensor) -> Tensor:
    """
    Compute the forward pass of 2D global average pooling (CPU-only wrapper).

    Global average pooling reduces each channel to a single value by averaging
    over the spatial dimensions.

    Parameters
    ----------
    x : Tensor
        Input tensor (CPU tensor in the current implementation).

    Returns
    -------
    Tensor
        Output tensor created via `Tensor._from_numpy`.
    """
    y_np = global_avgpool2d_forward_cpu(x.to_numpy())
    return Tensor._from_numpy(y_np, device=x.device, requires_grad=False)


def global_avgpool2d_backward(grad_out: Tensor, *, x_shape) -> Tensor:
    """
    Compute the backward pass of 2D global average pooling (CPU-only wrapper).

    Parameters
    ----------
    grad_out : Tensor
        Gradient with respect to the global average pooled output.
    x_shape : Any
        Shape of the original input tensor `x` used in the forward pass.

    Returns
    -------
    Tensor
        Gradient with respect to the input `x`, created via `Tensor._from_numpy`.
    """
    gx_np = global_avgpool2d_backward_cpu(grad_out.to_numpy(), x_shape=x_shape)
    return Tensor._from_numpy(gx_np, device=grad_out.device, requires_grad=False)
