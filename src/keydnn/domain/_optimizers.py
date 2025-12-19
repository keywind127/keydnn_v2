"""
Domain-level optimizer contracts for KeyDNN.

This module defines the `IOptimizer` protocol, which specifies the minimal
interface required for optimizer implementations (e.g., SGD, Adam).

Notes
-----
- Domain contracts are backend-agnostic and must not depend on NumPy or
  infrastructure implementations.
- Optimizers are responsible for updating trainable parameters based on their
  stored gradients. The details of gradient computation (autograd engine) are
  outside the scope of this protocol.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class IOptimizer(Protocol):
    """
    Optimizer interface contract.

    An optimizer holds references to trainable parameters and updates them
    in-place according to a specific optimization rule.

    Required methods
    ----------------
    - `step()` applies one optimization update to managed parameters.
    - `zero_grad()` clears gradients for managed parameters.
    """

    def step(self) -> None:
        """
        Apply one optimization step.

        Implementations should skip parameters that do not currently have
        gradients (e.g., `grad is None`).
        """
        ...

    def zero_grad(self) -> None:
        """
        Clear gradients for all managed parameters.
        """
        ...

    @property
    def params(self) -> Iterable[object]:
        """
        Return the parameters managed by this optimizer.

        Notes
        -----
        The protocol does not constrain the parameter type to keep the domain
        layer decoupled from infrastructure. Infrastructure implementations
        should return an iterable of `Parameter` objects.
        """
        ...
