"""
Autograd `Function` implementation for flattening tensors (CPU backend).

This module defines `FlattenFn`, a reshape-only autograd primitive that
collapses all non-batch dimensions into a single feature dimension:

    (N, d1, d2, ..., dk) -> (N, d1*d2*...*dk)

`FlattenFn` is commonly used to bridge convolutional / pooling outputs
(NCHW or similar) into fully-connected layers.

Design notes
------------
- This operation is a pure reshape (no arithmetic).
- The backward pass reshapes the incoming gradient back to the original
  input shape saved during the forward pass.
- `Function.forward()` returns an output tensor without attaching a `Context`
  by itself. The corresponding `Module` wrapper is responsible for attaching
  the `Context` to enable graph traversal (consistent with other KeyDNN ops).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..tensor._tensor_context import Context

from .._tensor import Tensor
from .._function import Function


class FlattenFn(Function):
    """
    Autograd `Function` for flattening tensors.

    This operation preserves the batch dimension and collapses all remaining
    dimensions into a single feature dimension.

    Shape semantics
    ---------------
    Forward:
        x:  (N, d1, d2, ..., dk)
        y:  (N, d1*d2*...*dk)

    Backward:
        grad_out: (N, d1*d2*...*dk)
        grad_x:   (N, d1, d2, ..., dk)

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "orig_shape": original `x.shape` required to restore gradients

    Notes
    -----
    - This is a reshape-only operation; it does not modify values.
    - The backward pass only performs a reshape, so it is numerically stable.
    """

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        """
        Flatten an input tensor across all non-batch dimensions.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors/metadata for backward.
        x : Tensor
            Input tensor with shape (N, ...). The first dimension is treated
            as the batch dimension and is preserved in the output.

        Returns
        -------
        Tensor
            Flattened tensor with shape (N, -1), where `-1` is the product of
            all non-batch dimensions.

        Notes
        -----
        - The original input shape is saved to `ctx.saved_meta["orig_shape"]`
          so that the backward pass can reshape gradients correctly.
        """
        if len(x.shape) < 1:
            raise ValueError("Flatten expects at least 1D input")

        N = x.shape[0]
        feature_dim = 1
        for d in x.shape[1:]:
            feature_dim *= d

        ctx.save_for_backward(x)
        ctx.saved_meta["orig_shape"] = x.shape

        out = x.reshape((N, feature_dim))
        return out

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate through flatten by reshaping gradients to the input shape.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the flattened output, shape (N, -1).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has the original
            input shape, or `(None,)` if the input does not require gradients.

        Notes
        -----
        - This method returns gradients aligned with the `parents` order used by
          the module wrapper (typically `(x,)`).
        """
        (x,) = ctx.saved_tensors
        if not x.requires_grad:
            return (None,)

        orig_shape: Tuple[int, ...] = ctx.saved_meta["orig_shape"]
        grad_x = grad_out.reshape(orig_shape)
        return (grad_x,)
