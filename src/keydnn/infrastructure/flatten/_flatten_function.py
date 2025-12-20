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
import numpy as np

from .._tensor import Tensor, Context
from .._function import Function


def _tensor_from_numpy(
    arr: np.ndarray, *, device, requires_grad: bool = False
) -> Tensor:
    """
    Create a `Tensor` on the given device and copy NumPy data into it.

    Parameters
    ----------
    arr : np.ndarray
        Source array containing tensor data.
    device
        Target device placement (e.g., CPU device).
    requires_grad : bool, optional
        Whether the created tensor should participate in autograd.
        Defaults to False.

    Returns
    -------
    Tensor
        A new tensor with `shape == arr.shape` whose underlying storage is
        populated from `arr`.

    Notes
    -----
    This helper follows the codebase pattern of:
    1) allocate tensor storage via `Tensor(shape=..., device=...)`
    2) populate storage via `copy_from_numpy(...)`
    """
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


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
        x_np = x.to_numpy()
        N = x_np.shape[0]
        out_np = x_np.reshape(N, -1)

        ctx.save_for_backward(x)
        ctx.saved_meta["orig_shape"] = x.shape

        out = _tensor_from_numpy(out_np, device=x.device, requires_grad=x.requires_grad)
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
        grad_np = grad_out.to_numpy().reshape(orig_shape)

        grad_x = _tensor_from_numpy(grad_np, device=x.device, requires_grad=False)
        return (grad_x,)
