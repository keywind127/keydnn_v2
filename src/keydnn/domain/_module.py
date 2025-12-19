"""
Module (layer) interface definitions.

This module defines the domain-level interface for neural network modules
(layers) using structural subtyping via `typing.Protocol`.

Any object that implements the required methods is considered a valid module,
independent of inheritance, enabling flexible composition and clean separation
between domain contracts and infrastructure implementations.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ._tensor import ITensor
from ._parameter import IParameter


@runtime_checkable
class IModule(Protocol):
    """
    Domain-level module (layer) interface.

    A module represents a composable unit of computation (e.g., layers,
    activation functions, or containers of other modules). This interface
    defines the minimal contract required for an object to participate in
    forward execution and parameter collection.

    Structural typing is used instead of inheritance, allowing implementations
    to remain lightweight and decoupled from the domain layer.

    Notes
    -----
    - Any object implementing both `forward` and `parameters` is considered
      a valid module.
    - This interface is safe to use with `isinstance` checks due to the
      `@runtime_checkable` decorator.
    """

    def forward(self, x: ITensor) -> ITensor:
        """
        Execute the forward computation of the module.

        Parameters
        ----------
        x : ITensor
            Input tensor to the module.

        Returns
        -------
        ITensor
            Output tensor produced by the module.
        """
        ...

    def parameters(self) -> Iterable[IParameter]:
        """
        Return the trainable parameters of the module.

        This method is used by optimizers and training loops to discover
        all parameters that should receive gradient updates.

        Returns
        -------
        Iterable[IParameter]
            An iterable over the module's trainable parameters.
        """
        ...
