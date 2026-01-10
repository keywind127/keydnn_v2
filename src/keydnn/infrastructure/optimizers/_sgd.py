"""
Stochastic Gradient Descent (SGD) optimizer implementation.

This module provides a minimal SGD optimizer for KeyDNN. The optimizer updates
`Parameter` instances in-place using their accumulated gradients and a fixed
learning rate, optionally applying classical L2 regularization (coupled weight
decay).

Design notes
------------
- Optimizers operate on `Parameter` objects and read gradients from `p.grad`.
- Parameters with `grad is None` are skipped to support partial graphs and
  frozen weights.
- Updates are applied in-place, keeping the optimizer independent from graph
  construction and autograd internals.
- This implementation intentionally omits momentum, Nesterov, and other SGD
  variants to keep the core optimizer minimal and easy to reason about.
- Current implementation is CPU-only in the current KeyDNN implementation.

This module contains only SGD. Other optimizers (e.g., Adam) live in separate
modules under the optimizers package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .._parameter import Parameter


@dataclass
class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates parameters in-place using their accumulated
    gradients and a fixed learning rate.

    Update rule
    -----------
    For each parameter ``p`` with gradient ``g``:

    - If ``weight_decay > 0`` (classical L2 regularization):
        ``g <- g + weight_decay * p``
    - Parameter update:
        ``p <- p - lr * g``

    Parameters
    ----------
    params : Sequence[Parameter]
        Parameters to be optimized.
    lr : float, optional
        Learning rate. Must be positive. Defaults to 1e-3.
    weight_decay : float, optional
        Classical L2 weight decay coefficient (coupled). Must be non-negative.
        Defaults to 0.0.

    Notes
    -----
    - Parameters with ``grad is None`` are skipped.
    - Momentum, Nesterov, and other SGD variants are intentionally omitted
      in this minimal implementation.
    - This optimizer is CPU-only in the current KeyDNN implementation.
    """

    params: Sequence[Parameter]
    lr: float = 1e-3
    weight_decay: float = 0.0

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Construct an SGD optimizer.

        Parameters
        ----------
        params : Iterable[Parameter]
            Iterable of parameters to optimize. The iterable is consumed and
            stored internally.
        lr : float, optional
            Learning rate. Must be > 0. Defaults to 1e-3.
        weight_decay : float, optional
            Classical L2 regularization coefficient. Must be >= 0.
            Defaults to 0.0.

        Raises
        ------
        ValueError
            If ``lr <= 0`` or ``weight_decay < 0``.
        """
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

    def zero_grad(self) -> None:
        """
        Clear gradients for all managed parameters.

        Notes
        -----
        This calls ``zero_grad()`` on each parameter, which clears the stored
        gradient tensor (if any). Training loops typically call `zero_grad()`
        before computing a new backward pass to avoid gradient accumulation.
        """
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """
        Apply one SGD update step to all managed parameters.

        Notes
        -----
        - Parameters with ``grad is None`` are skipped.
        - Weight decay is implemented as classical L2 regularization
          (coupled with the gradient), not as decoupled weight decay.
        - The update is performed in-place on the parameter.
        """
        for p in self.params:
            g = p.grad
            if g is None:
                continue

            # Optional L2 weight decay (decoupled is AdamW; this is classical)
            if self.weight_decay != 0.0:
                g = g + (self.weight_decay * p)

            # In-place update: p <- p - lr * g
            p -= self.lr * g
