"""
Trainable parameter interface definitions.

This module defines the domain-level interface for trainable parameters used
by optimization algorithms. The interface abstracts over concrete tensor
implementations and backend details, providing a minimal, structural contract
for parameter management during training.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional

from ._tensor import ITensor


@runtime_checkable
class IParameter(Protocol):
    """
    Domain-level interface for trainable parameters.

    An `IParameter` represents a tensor-like object that participates in
    optimization. It exposes control over gradient accumulation and provides
    access to the associated gradient tensor.

    This interface is intentionally structural (duck-typed) and backend-agnostic,
    allowing different tensor and parameter implementations to interoperate
    as long as they satisfy the contract.

    Notes
    -----
    - Parameters may be frozen or unfrozen via the `requires_grad` flag.
    - Optimizers rely on this interface to discover and update parameters.
    """

    # ---- training control ----
    @property
    def requires_grad(self) -> bool:
        """
        Indicate whether this parameter should accumulate gradients.

        Returns
        -------
        bool
            True if gradients should be accumulated for this parameter,
            False if the parameter is frozen.
        """
        ...

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient accumulation for this parameter.

        Parameters
        ----------
        value : bool
            If True, gradients will be accumulated during backpropagation.
            If False, the parameter will be excluded from gradient updates.
        """
        ...

    # ---- gradient access ----
    @property
    def grad(self) -> Optional["ITensor"]:
        """
        Return the gradient tensor associated with this parameter.

        The gradient is typically populated during backpropagation and may
        be None if gradients have not yet been computed or have been cleared.

        Returns
        -------
        Optional[ITensor]
            The gradient tensor, or None if no gradient is currently stored.
        """
        ...

    def zero_grad(self) -> None:
        """
        Clear the stored gradient for this parameter.

        This method is typically called at the start of an optimization step
        to prevent unintended gradient accumulation across iterations.
        """
        ...
