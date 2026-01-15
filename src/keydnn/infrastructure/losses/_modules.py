"""
Module-based loss layers.

This module provides infrastructure-level `Module` wrappers around the
function-style autograd loss primitives (e.g., `MSEFn`, `BinaryCrossEntropyFn`).

Why both Function and Module forms exist
----------------------------------------
- `Function` classes (e.g., `MSEFn`) implement the math and derivatives in a
  reusable, low-level form that plugs directly into the autograd engine.
- `Module` classes (e.g., `MSE`) provide a Keras-like, layer-style interface
  for composing models and calling losses via `loss(pred, target)` in training
  loops, while still preserving explicit autograd wiring.

Autograd integration
--------------------
Each loss module constructs a `Context` during `forward` and wires a
`backward_fn` that delegates to the corresponding `Function.backward`.
If any parent requires gradients, the context is attached to the scalar output
tensor so the autograd engine can traverse the graph.

Notes
-----
- Loss modules are *stateless* in this initial set (no label smoothing, no
  reduction modes beyond those implemented by the `Fn` primitives).
- Some losses return gradients for both `pred` and `target` (SSE/MSE),
  while others treat `target` as a constant and return `None` for its gradient
  (BCE/CCE). This module preserves that behavior.
- No shape broadcasting is introduced here; shape rules remain enforced by the
  underlying `Tensor` ops and the `Fn` implementations.
"""

from __future__ import annotations

from ..tensor._tensor_context import Context
from ...domain.model._stateless_mixin import StatelessConfigMixin
from ..module._serialization_core import register_module
from ..tensor._tensor import Tensor
from .._module import Module

from ._functions import (
    SSEFn,
    MSEFn,
    BinaryCrossEntropyFn,
    CategoricalCrossEntropyFn,
)


def _any_requires_grad(*xs: Tensor) -> bool:
    """
    Return True if any provided tensor requires gradients.

    Parameters
    ----------
    *xs : Tensor
        Tensors to check.

    Returns
    -------
    bool
        True if at least one tensor has `requires_grad=True`.
    """
    return any(bool(x.requires_grad) for x in xs)


@register_module()
class SSE(StatelessConfigMixin, Module):
    """
    Sum of Squared Errors (SSE) loss module.

    This module wraps `SSEFn` and computes:

        SSE(pred, target) = sum((pred - target)^2)

    Notes
    -----
    - Returns a scalar tensor.
    - The backward pass returns gradients for both `pred` and `target`.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute SSE(pred, target) with explicit autograd wiring.

        Parameters
        ----------
        pred : Tensor
            Predicted values.
        target : Tensor
            Ground-truth values (same shape as `pred`).

        Returns
        -------
        Tensor
            Scalar tensor containing the SSE loss.
        """
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: SSEFn.backward(ctx, grad_out),
        )
        out = SSEFn.forward(ctx, pred, target)

        if _any_requires_grad(pred, target):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class MSE(StatelessConfigMixin, Module):
    """
    Mean Squared Error (MSE) loss module.

    This module wraps `MSEFn` and computes:

        MSE(pred, target) = mean((pred - target)^2)

    Notes
    -----
    - Returns a scalar tensor.
    - The backward pass returns gradients for both `pred` and `target`.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE(pred, target) with explicit autograd wiring.

        Parameters
        ----------
        pred : Tensor
            Predicted values.
        target : Tensor
            Ground-truth values (same shape as `pred`).

        Returns
        -------
        Tensor
            Scalar tensor containing the MSE loss.
        """
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: MSEFn.backward(ctx, grad_out),
        )
        out = MSEFn.forward(ctx, pred, target)

        if _any_requires_grad(pred, target):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class BinaryCrossEntropy(StatelessConfigMixin, Module):
    """
    Binary Cross Entropy (BCE) loss module.

    This module wraps `BinaryCrossEntropyFn` and computes the mean BCE:

        BCE(pred, target) =
            mean( -[ target * log(pred) + (1 - target) * log(1 - pred) ] )

    Notes
    -----
    - `pred` is expected to be probabilities in (0, 1) (e.g., sigmoid output).
    - `target` is expected to be binary (0 or 1) with the same shape as `pred`.
    - Returns a scalar tensor.
    - The backward pass returns a gradient for `pred` and `None` for `target`.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE(pred, target) with explicit autograd wiring.

        Parameters
        ----------
        pred : Tensor
            Predicted probabilities in (0, 1).
        target : Tensor
            Binary targets (0 or 1), same shape as `pred`.

        Returns
        -------
        Tensor
            Scalar tensor containing the mean BCE loss.
        """
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: BinaryCrossEntropyFn.backward(ctx, grad_out),
        )
        out = BinaryCrossEntropyFn.forward(ctx, pred, target)

        if _any_requires_grad(pred, target):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class CategoricalCrossEntropy(StatelessConfigMixin, Module):
    """
    Categorical Cross Entropy (CCE) loss module.

    This module wraps `CategoricalCrossEntropyFn` and computes:

        CCE(pred, target) = -sum(target * log(pred)) / N

    where N is the batch size (pred.shape[0]).

    Notes
    -----
    - `pred` is expected to have shape (N, C) with class probabilities
      (e.g., softmax output).
    - `target` must be one-hot encoded with the same shape as `pred`.
    - Returns a scalar tensor.
    - The backward pass returns a gradient for `pred` and `None` for `target`.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute CCE(pred, target) with explicit autograd wiring.

        Parameters
        ----------
        pred : Tensor
            Predicted class probabilities of shape (N, C).
        target : Tensor
            One-hot targets of shape (N, C).

        Returns
        -------
        Tensor
            Scalar tensor containing the batch-mean categorical cross entropy.
        """
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: CategoricalCrossEntropyFn.backward(
                ctx, grad_out
            ),
        )
        out = CategoricalCrossEntropyFn.forward(ctx, pred, target)

        if _any_requires_grad(pred, target):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out
