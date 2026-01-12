"""
Optimizer base protocol.

This module defines the minimal abstract interface for optimizers used by the
training loop. Optimizers are responsible for updating model parameters based
on accumulated gradients and for resetting gradients between optimization steps.

The interface is intentionally small and framework-agnostic so that concrete
optimizers (e.g., SGD, Adam) can be implemented without coupling to specific
tensor, device, or autograd implementations.
"""

from __future__ import annotations

from abc import ABC


class _Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Concrete optimizers must implement parameter update logic in `step()` and
    gradient reset logic in `zero_grad()`. The training loop is expected to call
    these methods once per optimization step.
    """

    def step(self):
        """
        Perform a single optimization step.

        This method should update all trainable parameters based on their
        current gradients and the optimizer's internal state (e.g., learning
        rate, momentum buffers).

        Notes
        -----
        - This method does not return a value.
        - The exact update semantics depend on the concrete optimizer.
        """
        ...

    def zero_grad(self):
        """
        Reset gradients of all optimized parameters.

        This method should clear or zero out accumulated gradients so that
        subsequent backward passes start from a clean state.

        Notes
        -----
        - Typically called before or after `step()`, depending on the
          training loop design.
        """
        ...
