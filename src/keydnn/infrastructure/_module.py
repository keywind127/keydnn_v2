"""
Infrastructure module base class.

This module provides a concrete `Module` implementation that satisfies the
domain-level `IModule` protocol. It implements common conveniences used by
neural network layers, including:

- parameter registration and storage
- parameter traversal (`parameters`, `named_parameters`)
- `__call__` forwarding to `forward` for ergonomic invocation

This class is part of the infrastructure layer and is intended to be subclassed
by concrete layers (e.g., Linear, Conv2D, activations-as-modules, containers).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Iterator, Optional

from ..domain._module import IModule
from ..domain._parameter import IParameter


class Module(IModule):
    """
    Infrastructure base class for layers/modules.

    This class provides a lightweight foundation for implementing trainable
    layers. Subclasses typically:
    - create `Parameter` instances (or other `IParameter` implementations),
    - register them via `register_parameter`,
    - implement `forward` to define computation.

    Attributes
    ----------
    _parameters : Dict[str, IParameter]
        Mapping from parameter name to parameter object for this module.

    Notes
    -----
    - This implementation only handles *direct* parameters registered on the
      module itself. Recursive traversal over submodules (e.g., `self._modules`)
      can be added later if/when you implement module composition containers.
    - `__call__` delegates to `forward`, matching the common deep learning
      framework convention.
    """

    def __init__(self) -> None:
        """
        Initialize an empty module with no registered parameters.
        """
        self._parameters: Dict[str, IParameter] = {}

    def register_parameter(self, name: str, param: Optional[IParameter]) -> None:
        """
        Register a parameter with this module.

        Parameters
        ----------
        name : str
            Name under which the parameter will be stored (e.g., "weight", "bias").
        param : Optional[IParameter]
            Parameter instance to register. If None, registration is skipped
            (useful for optional parameters like disabled bias).

        Notes
        -----
        - If `param` is None, nothing is registered.
        - If the name already exists, it is overwritten intentionally. This
          mirrors common deep learning frameworks and supports reassignment
          during initialization or reconfiguration.
        """
        if param is None:
            return
        self._parameters[name] = param

    def parameters(self) -> Iterable[IParameter]:
        """
        Return an iterable over this module's registered parameters.

        Returns
        -------
        Iterable[IParameter]
            Iterable of parameters registered directly on this module.

        Notes
        -----
        This method does not currently recurse into submodules. If you later add
        submodule registration, extend this method accordingly.
        """
        return self._parameters.values()

    def named_parameters(self) -> Iterator[tuple[str, IParameter]]:
        """
        Return an iterator over (name, parameter) pairs.

        Returns
        -------
        Iterator[tuple[str, IParameter]]
            Iterator yielding (parameter_name, parameter) for each registered
            parameter.

        Notes
        -----
        This method is useful for debugging, logging, and optimizer parameter
        grouping.
        """
        return iter(self._parameters.items())

    def forward(self, x):
        """
        Execute the forward computation of the module.

        Parameters
        ----------
        x : ITensor
            Input tensor (or tensor-like) for the module.

        Returns
        -------
        ITensor
            Output tensor produced by the module.

        Raises
        ------
        NotImplementedError
            Always raised in the base class. Subclasses must implement this.
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Call the module as a function, delegating to `forward`.

        This enables ergonomic usage such as:

            y = module(x)

        Parameters
        ----------
        x : ITensor
            Input tensor (or tensor-like) for the module.

        Returns
        -------
        ITensor
            Output tensor produced by `forward`.
        """
        return self.forward(x)
