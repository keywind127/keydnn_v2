from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Iterator, Optional

from ..domain._module import IModule
from ..domain._parameter import IParameter


class Module(IModule):
    """
    Infrastructure base class for layers/modules.

    Provides:
    - parameter registration
    - parameters() traversal
    - __call__ -> forward convenience
    """

    def __init__(self) -> None:
        self._parameters: Dict[str, IParameter] = {}

    def register_parameter(self, name: str, param: Optional[IParameter]) -> None:
        """
        Register a parameter with this module.

        - If param is None (e.g., bias disabled), nothing is registered.
        - Overwrites existing name (intentional; matches typical DL frameworks).
        """
        if param is None:
            return
        self._parameters[name] = param

    def parameters(self) -> Iterable[IParameter]:
        return self._parameters.values()

    def named_parameters(self) -> Iterator[tuple[str, IParameter]]:
        return iter(self._parameters.items())

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
