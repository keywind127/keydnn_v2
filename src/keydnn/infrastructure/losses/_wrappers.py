"""
Callable wrappers for Function-style loss primitives.

These wrappers adapt `Function` loss classes (e.g., `MSEFn`, `BinaryCrossEntropyFn`)
into *plain callables* compatible with `Model.fit()` / `train_on_batch()`:

    loss_fn = LossFromFn(MSEFn)
    loss = loss_fn(y_pred, y_true)  # -> scalar Tensor with autograd context

Why this exists
---------------
Your `Model.fit()` expects a callable:

    loss(y_pred, y_true) -> scalar Tensor

Whereas `Function` losses require an explicit `Context` object to be constructed
in user code.

This module bridges that gap while keeping the `Fn` implementations unchanged.

Notes
-----
- We attach a `Context` to the scalar output only if any parent requires grad.
- For BCE/CCE, the derivative for `target` is `None`, consistent with the `Fn`.
- This is intentionally minimal and stateless.

Typical usage
-------------
from keydnn.infrastructure.losses._wrappers import cce_loss, bce_loss

history = model.fit(
    x_t, y_t,
    loss=cce_loss(),         # <- callable
    optimizer=opt,
    ...
)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Type

from ..tensor._tensor_context import Context
from ..tensor._tensor import Tensor

from ._functions import (
    SSEFn,
    MSEFn,
    BinaryCrossEntropyFn,
    CategoricalCrossEntropyFn,
)


class _LossFnType(Protocol):
    """
    Protocol for loss Function classes.

    LossFns are expected to provide:
    - forward(ctx, pred, target) -> Tensor (scalar)
    - backward(ctx, grad_out) -> tuple[Tensor|None, Tensor|None]
    """

    @staticmethod
    def forward(ctx: Context, pred: Tensor, target: Tensor) -> Tensor: ...

    @staticmethod
    def backward(
        ctx: Context, grad_out: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]: ...


def _any_requires_grad(*xs: Tensor) -> bool:
    """
    Return True if any provided tensor requires gradients.
    """
    return any(bool(getattr(x, "requires_grad", False)) for x in xs)


@dataclass(frozen=True)
class LossFromFn:
    """
    Wrap a `Function` loss class into a callable loss(pred, target) -> scalar Tensor.

    Parameters
    ----------
    fn_cls : type
        A loss `Function` class implementing `forward(ctx, pred, target)` and
        `backward(ctx, grad_out)`.

    Examples
    --------
    loss = LossFromFn(MSEFn)
    out = loss(pred, target)
    out.backward()
    """

    fn_cls: Type[_LossFnType]

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute loss(pred, target) with explicit autograd wiring.

        Returns
        -------
        Tensor
            Scalar loss tensor.
        """
        # Create context + wire backward to the Fn's backward.
        ctx = Context(
            parents=(pred, target),
            backward_fn=lambda grad_out: self.fn_cls.backward(ctx, grad_out),
        )

        out = self.fn_cls.forward(ctx, pred, target)

        # Attach ctx only if needed (so pure forward / eval stays light).
        if _any_requires_grad(pred, target):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


# ----------------------------------------------------------------------
# Convenience factories (so fit(...) can use loss="callable" easily)
# ----------------------------------------------------------------------


def sse_loss() -> Callable[[Tensor, Tensor], Tensor]:
    """Return a callable SSE loss compatible with Model.fit()."""
    return LossFromFn(SSEFn)


def mse_loss() -> Callable[[Tensor, Tensor], Tensor]:
    """Return a callable MSE loss compatible with Model.fit()."""
    return LossFromFn(MSEFn)


def bce_loss() -> Callable[[Tensor, Tensor], Tensor]:
    """Return a callable Binary Cross Entropy loss compatible with Model.fit()."""
    return LossFromFn(BinaryCrossEntropyFn)


def cce_loss() -> Callable[[Tensor, Tensor], Tensor]:
    """Return a callable Categorical Cross Entropy loss compatible with Model.fit()."""
    return LossFromFn(CategoricalCrossEntropyFn)


__all__ = [
    "LossFromFn",
    "sse_loss",
    "mse_loss",
    "bce_loss",
    "cce_loss",
]
