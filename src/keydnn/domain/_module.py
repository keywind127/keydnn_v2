from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ._tensor import ITensor
from ._parameter import IParameter


@runtime_checkable
class IModule(Protocol):
    """
    Domain-level module/layer interface (structural typing).

    Any object that provides forward() and parameters() is considered a module.
    """

    def forward(self, x: ITensor) -> ITensor: ...

    def parameters(self) -> Iterable[IParameter]: ...
