"""
Autograd `Function` implementations for 2D pooling (CPU backend).

This module bridges the NumPy CPU pooling kernels in `ops.pool2d_cpu` with the
KeyDNN autograd system (`Tensor`, `Context`). Each pooling operator is
implemented as a `Function` with:

- `forward(ctx, ...)` that computes the output tensor and saves any information
  required for gradient computation.
- `backward(ctx, grad_out)` that computes gradients with respect to the input
  tensor(s), returning them in the same order as `ctx.parents`.

Implemented operators
---------------------
- `MaxPool2dFn`:
    Stores argmax indices from the forward pass to route gradients correctly.
- `AvgPool2dFn`:
    Distributes output gradients uniformly over each pooling window.
- `GlobalAvgPool2dFn`:
    Reduces spatial dimensions and distributes gradients evenly over H*W.

Design notes
------------
- All implementations assume **NCHW** layout.
- These functions are CPU-only because they rely on NumPy kernels.
- The returned tensors use `_tensor_from_numpy` to remain consistent with
  the rest of the codebase (no dependency on a global `from_numpy` helper).
- Backward methods respect `requires_grad` and return `None` for inputs that
  do not require gradients.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

from .._tensor import Tensor, Context
from .._function import Function
from ..ops.pool2d_cpu import (
    _pair,
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


def _tensor_from_numpy(
    arr: np.ndarray, *, device, requires_grad: bool = False
) -> Tensor:
    """
    Create a `Tensor` on the given device and copy NumPy data into it.

    Parameters
    ----------
    arr : np.ndarray
        Source array containing the desired tensor data.
    device
        Target device placement for the tensor (e.g., `Device("cpu")`).
    requires_grad : bool, optional
        Whether the created tensor should participate in autograd.
        Defaults to False.

    Returns
    -------
    Tensor
        A new tensor with shape matching `arr.shape` whose storage is populated
        by copying data from `arr`.

    Notes
    -----
    - This helper enforces the codebase convention of allocating a tensor first
      and then copying data via `copy_from_numpy`.
    - The resulting tensor has `ctx=None` (no autograd history attached here).
    """
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class MaxPool2dFn(Function):
    """
    Autograd-enabled MaxPool2D operation.

    The forward pass performs windowed max pooling over spatial dimensions and
    records argmax indices so that the backward pass can route gradients to the
    exact input locations that produced each pooled output value.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": original input shape
        - "kernel_size": (k_h, k_w)
        - "stride": (s_h, s_w)
        - "padding": (p_h, p_w)
        - "argmax_idx": argmax indices from the forward pass

    Notes
    -----
    - Operates on NCHW tensors.
    - CPU-only (NumPy).
    """

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        *,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
    ) -> Tensor:
        """
        Compute max pooling output and save metadata for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors and metadata for backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).
        kernel_size : int or tuple[int, int]
            Pooling window size.
        stride : int or tuple[int, int] or None, optional
            Pooling stride. If None, defaults to `kernel_size`.
        padding : int or tuple[int, int], optional
            Spatial padding applied before pooling.

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out).

        Notes
        -----
        - Argmax indices are stored in `ctx.saved_meta["argmax_idx"]` as a NumPy
          array to avoid creating additional autograd edges.
        """
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)

        y_np, argmax_idx = maxpool2d_forward_cpu(
            x.to_numpy(), kernel_size=k, stride=s, padding=p
        )

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape
        ctx.saved_meta["kernel_size"] = k
        ctx.saved_meta["stride"] = s
        ctx.saved_meta["padding"] = p
        ctx.saved_meta["argmax_idx"] = argmax_idx  # numpy array (meta)

        out = _tensor_from_numpy(y_np, device=x.device, requires_grad=x.requires_grad)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through max pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the output, shape (N, C, H_out, W_out).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has shape
            (N, C, H, W), or `(None,)` if the input does not require gradients.

        Notes
        -----
        - The gradient is routed only to the input positions that were maxima
          during the forward pass.
        """
        (x,) = ctx.saved_tensors
        x_shape = ctx.saved_meta["x_shape"]
        k = ctx.saved_meta["kernel_size"]
        s = ctx.saved_meta["stride"]
        p = ctx.saved_meta["padding"]
        argmax_idx = ctx.saved_meta["argmax_idx"]

        if not x.requires_grad:
            return (None,)

        gx_np = maxpool2d_backward_cpu(
            grad_out.to_numpy(),
            argmax_idx,
            x_shape=x_shape,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        gx = _tensor_from_numpy(gx_np, device=x.device, requires_grad=False)
        return (gx,)


class AvgPool2dFn(Function):
    """
    Autograd-enabled AvgPool2D operation.

    The forward pass performs windowed average pooling over spatial dimensions.
    The backward pass distributes gradients uniformly across each pooling window.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": original input shape
        - "kernel_size": (k_h, k_w)
        - "stride": (s_h, s_w)
        - "padding": (p_h, p_w)

    Notes
    -----
    - Operates on NCHW tensors.
    - Uses zero-padding and averages over the full kernel area.
    - CPU-only (NumPy).
    """

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        *,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
    ) -> Tensor:
        """
        Compute average pooling output and save metadata for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors and metadata for backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).
        kernel_size : int or tuple[int, int]
            Pooling window size.
        stride : int or tuple[int, int] or None, optional
            Pooling stride. If None, defaults to `kernel_size`.
        padding : int or tuple[int, int], optional
            Spatial padding applied before pooling.

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out).
        """
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)

        y_np = avgpool2d_forward_cpu(x.to_numpy(), kernel_size=k, stride=s, padding=p)

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape
        ctx.saved_meta["kernel_size"] = k
        ctx.saved_meta["stride"] = s
        ctx.saved_meta["padding"] = p

        out = _tensor_from_numpy(y_np, device=x.device, requires_grad=x.requires_grad)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through average pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the output, shape (N, C, H_out, W_out).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has shape
            (N, C, H, W), or `(None,)` if the input does not require gradients.
        """
        (x,) = ctx.saved_tensors
        x_shape = ctx.saved_meta["x_shape"]
        k = ctx.saved_meta["kernel_size"]
        s = ctx.saved_meta["stride"]
        p = ctx.saved_meta["padding"]

        if not x.requires_grad:
            return (None,)

        gx_np = avgpool2d_backward_cpu(
            grad_out.to_numpy(),
            x_shape=x_shape,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        gx = _tensor_from_numpy(gx_np, device=x.device, requires_grad=False)
        return (gx,)


class GlobalAvgPool2dFn(Function):
    """
    Autograd-enabled Global Average Pooling (GAP) operation.

    Global average pooling reduces each channel to a single spatial value by
    averaging over height and width:

        (N, C, H, W) -> (N, C, 1, 1)

    The backward pass distributes gradients uniformly across all H*W input
    positions per channel.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": original input shape

    Notes
    -----
    - Operates on NCHW tensors.
    - CPU-only (NumPy).
    """

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        """
        Compute global average pooling output and save input shape.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors and metadata for backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, 1, 1).
        """
        y_np = global_avgpool2d_forward_cpu(x.to_numpy())

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape

        out = _tensor_from_numpy(y_np, device=x.device, requires_grad=x.requires_grad)
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through global average pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to output, shape (N, C, 1, 1).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has shape
            (N, C, H, W), or `(None,)` if the input does not require gradients.
        """
        (x,) = ctx.saved_tensors
        if not x.requires_grad:
            return (None,)

        gx_np = global_avgpool2d_backward_cpu(
            grad_out.to_numpy(), x_shape=ctx.saved_meta["x_shape"]
        )
        gx = _tensor_from_numpy(gx_np, device=x.device, requires_grad=False)
        return (gx,)
