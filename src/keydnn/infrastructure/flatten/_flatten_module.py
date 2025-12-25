"""
Flatten module for KeyDNN.

This module provides a high-level `Module` wrapper around `FlattenFn`,
exposing a user-facing layer that is compatible with the KeyDNN autograd
engine.

`Flatten` is typically used to connect spatial feature maps produced by
convolutional / pooling layers to fully connected layers by collapsing all
non-batch dimensions into a single feature dimension.

Shape semantics
---------------
Input:
    (N, d1, d2, ..., dk)

Output:
    (N, d1 * d2 * ... * dk)

Design notes
------------
- `Flatten` is stateless and has no trainable parameters.
- Autograd support is provided by attaching a `Context` to the output tensor
  during forward when `x.requires_grad` is True.
- The actual reshape logic lives in `FlattenFn` to keep the module lightweight
  and consistent with the Function/Module separation used across KeyDNN.
"""

from __future__ import annotations

from ..tensor._tensor_context import Context

from ...domain.model._stateless_mixin import StatelessConfigMixin
from ..module._serialization_core import register_module

from .._module import Module
from ..tensor._tensor import Tensor
from ._flatten_function import FlattenFn


@register_module()
class Flatten(StatelessConfigMixin, Module):
    """
    Flatten layer.

    This layer reshapes an input tensor by preserving the batch dimension and
    collapsing all remaining dimensions into a single feature dimension.

    This is commonly used in CNN architectures to transition from
    convolutional/pooling outputs (e.g., NCHW tensors) to a dense `Linear`
    layer.

    Notes
    -----
    - This operation is a pure reshape (no numerical transformation).
    - Gradients are propagated by reshaping `grad_out` back to the original
      input shape (implemented in `FlattenFn.backward`).
    - The output tensor only receives an attached `Context` when the input
      requires gradients, avoiding unnecessary autograd overhead.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply flattening to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N, ...), where the first dimension is
            treated as the batch dimension.

        Returns
        -------
        Tensor
            Flattened tensor with shape (N, -1), where `-1` is the product of
            all non-batch dimensions of `x`.

        Notes
        -----
        - This method constructs a `Context` that records `x` as its single
          parent and delegates gradient computation to `FlattenFn.backward`.
        - The `Context` is attached to the output tensor only when
          `x.requires_grad` is True, consistent with other KeyDNN modules.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: FlattenFn.backward(ctx, grad_out),
        )

        out = FlattenFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out
