from __future__ import annotations

from typing import Protocol, runtime_checkable
from ._device import Device


@runtime_checkable
class ITensor(Protocol):
    """
    Domain-level tensor interface (structural typing).

    Any object that provides the required properties is considered
    a tensor, regardless of inheritance or concrete implementation.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def device(self) -> Device: ...
