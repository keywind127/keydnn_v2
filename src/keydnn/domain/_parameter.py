from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional

from ._tensor import ITensor


@runtime_checkable
class IParameter(Protocol):
    """
    Domain-level interface for trainable parameters.

    An IParameter represents a trainable tensor-like object that:
    - participates in optimization
    - may accumulate gradients
    - can be frozen or unfrozen via requires_grad

    This interface is intentionally structural (duck-typed) and
    backend-agnostic.
    """

    # ---- training control ----
    @property
    def requires_grad(self) -> bool:
        """Whether this parameter should accumulate gradients."""
        ...

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None: ...

    # ---- gradient access ----
    @property
    def grad(self) -> Optional["ITensor"]:
        """Gradient tensor associated with this parameter, if any."""
        ...

    def zero_grad(self) -> None:
        """Clear the stored gradient."""
        ...
