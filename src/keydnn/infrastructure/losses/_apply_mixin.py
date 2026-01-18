"""
Autograd apply helper for Function-style loss primitives.

This module defines a small mixin, `_ApplyMixin`, that provides a unified
`apply()` entry point for `Function` subclasses that take exactly two tensor
inputs (typically `pred` and `target`) and return a scalar loss tensor.

Purpose
-------
In KeyDNN, loss functions are implemented as low-level `Function` classes
to keep their math and gradient definitions explicit and reusable.
However, `Model.fit()` expects losses to be callable as:

    loss(y_pred, y_true) -> scalar Tensor

`_ApplyMixin` bridges this gap by encapsulating the boilerplate required to:
- Construct an autograd `Context`
- Invoke `Function.forward(ctx, ...)`
- Attach a custom backward closure to the returned scalar tensor
- Ensure the outer loss context is preserved (not overwritten by inner ops
  such as `sum()` or `mean()`)

Design Notes
------------
- This mixin intentionally assumes a fixed two-input signature
  `(pred, target)` to keep loss wiring simple and explicit.
- The returned tensor always represents a terminal node in the computation
  graph suitable for `Tensor.backward()`.
- Gradient attachment is conditional on `pred.requires_grad` to avoid
  unnecessary autograd overhead during inference or metric-only evaluation.
- This mixin does not perform any reduction logic itself; all reductions
  are delegated to `Tensor` operations inside the `Function.forward`
  implementation.

Typical Usage
-------------
Loss `Function` classes inherit from `_ApplyMixin` and expose a user-facing
callable via:

    MyLossFn.as_loss()

which returns a `(pred, target) -> Tensor` function compatible with
`Model.fit(loss=...)`.

This keeps loss math centralized in `Function` classes while avoiding
duplication of autograd wiring code across different loss implementations.
"""

from __future__ import annotations
from abc import ABC

from ..tensor._tensor_context import Context
from ..tensor._tensor import Tensor


class _ApplyMixin(ABC):
    """
    Mixin providing an `apply()` entry point for loss-style `Function` classes.

    `_ApplyMixin` encapsulates the standard autograd wiring pattern used by
    loss functions in KeyDNN. It builds an explicit `Context`, calls the
    subclass-defined `forward()` method, and conditionally attaches a
    backward closure to the returned scalar tensor.

    This mixin exists to:
    - Eliminate repeated boilerplate across loss `Function` implementations
    - Ensure the *outer* loss context is preserved, rather than being
      accidentally replaced by contexts created by inner Tensor ops
      (e.g. `sum()`, `mean()`)
    - Provide a uniform callable interface compatible with `Model.fit()`

    Key Behaviors
    -------------
    - Assumes a fixed `(pred, target)` signature for `forward()` and `backward()`
    - Treats the loss output as a terminal scalar in the computation graph
    - Attaches gradients only when `pred.requires_grad` is True
    - Delegates all mathematical operations and reductions to the subclass

    Notes
    -----
    This mixin does *not* implement any loss logic itself. Subclasses must
    define:
        - `forward(ctx, pred, target)`
        - `backward(ctx, grad_out)`

    The mixin simply standardizes how these methods are invoked and wired
    into the autograd engine.
    """

    @classmethod
    def apply(cls, pred: Tensor, target: Tensor) -> Tensor:
        """
        Apply this Function with autograd wiring.

        This builds a `Context`, calls `cls.forward(ctx, ...)`, and if any
        Tensor inputs require gradients, attaches a backward closure onto
        the output tensor.

        Returns
        -------
        Tensor
            The output tensor produced by `forward`.
        """
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: cls.backward(ctx, grad_out),
        )
        out = cls.forward(ctx, pred, target)

        # IMPORTANT: ensure the returned scalar uses THIS ctx,
        # not the ctx from inner ops like sum()/mean().
        if bool(pred.requires_grad):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out
