"""
Infrastructure module base class.

This module provides a concrete `Module` implementation that satisfies the
domain-level `IModule` protocol. It implements common conveniences used by
neural network layers, including:

- parameter registration and storage
- submodule registration and storage
- recursive parameter traversal (`parameters`, `named_parameters`)
- `__call__` forwarding to `forward` for ergonomic invocation

This class is part of the infrastructure layer and is intended to be subclassed
by concrete layers (e.g., Linear, Conv2D, activations-as-modules, containers).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Iterator, Optional, Any

from ..domain._module import IModule
from ..domain._parameter import IParameter


class Module(IModule):
    """
    Infrastructure base class for layers/modules.

    This class provides a lightweight foundation for implementing trainable
    layers. Subclasses typically:
    - create `Parameter` instances (or other `IParameter` implementations),
    - register them (explicitly via `register_parameter` or implicitly via attribute assignment),
    - implement `forward` to define computation.

    Attributes
    ----------
    _parameters : Dict[str, IParameter]
        Mapping from parameter name to parameter object for this module.
    _modules : Dict[str, Module]
        Mapping from child module name to child module object for this module.

    Notes
    -----
    - This implementation supports *recursive* traversal over registered submodules.
    - Parameters and submodules can be registered explicitly (register_*) or
      implicitly by assigning them as attributes, e.g.:
          self.weight = Parameter(...)
          self.block = Sequential(...)
    - `__call__` delegates to `forward`, matching common deep learning framework conventions.
    """

    def __init__(self) -> None:
        """
        Initialize an empty module with no registered parameters/submodules.
        """
        # Use super().__setattr__ to avoid triggering our __setattr__ logic.
        super().__setattr__("_parameters", {})  # type: ignore[assignment]
        super().__setattr__("_modules", {})  # type: ignore[assignment]

    def __setattr__(self, name: str, value) -> None:
        """
        Intercept attribute assignment to auto-register Parameters and child Modules.

        This mirrors common deep learning framework behavior and ensures that:
        - optimizers can discover parameters via `parameters()`
        - containers can recurse into submodules
        """
        # Let internal bookkeeping attributes pass through without registration.
        if name in {"_parameters", "_modules"}:
            super().__setattr__(name, value)
            return

        # Lazy imports to avoid hard cycles at import time.
        try:
            from ._parameter import Parameter  # concrete infra Parameter
        except Exception:  # pragma: no cover
            Parameter = None  # type: ignore[assignment]

        # If user assigns None, treat it as "unregister" if present.
        if value is None:
            if hasattr(self, "_parameters") and name in self._parameters:
                self._parameters.pop(name, None)
            if hasattr(self, "_modules") and name in self._modules:
                self._modules.pop(name, None)
            super().__setattr__(name, value)
            return

        # Auto-register concrete Parameters.
        if Parameter is not None and isinstance(value, Parameter):
            self._parameters[name] = value

        # Auto-register child Modules (infrastructure modules).
        elif isinstance(value, Module):
            self._modules[name] = value

        super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Optional[IParameter]) -> None:
        """
        Register a parameter with this module.

        Parameters
        ----------
        name : str
            Name under which the parameter will be stored (e.g., "weight", "bias").
        param : Optional[IParameter]
            Parameter instance to register. If None, registration is skipped.

        Notes
        -----
        - If `param` is None, nothing is registered.
        - If the name already exists, it is overwritten intentionally.
        - This also sets the attribute on the module so `self.<name>` works.
        """
        if param is None:
            return
        self._parameters[name] = param
        super().__setattr__(name, param)

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        """
        Register a child module with this module.

        Parameters
        ----------
        name : str
            Name under which the module will be stored.
        module : Optional[Module]
            Child module to register. If None, registration is skipped.

        Notes
        -----
        - If `module` is None, nothing is registered.
        - This also sets the attribute on the module so `self.<name>` works.
        """
        if module is None:
            return
        self._modules[name] = module
        super().__setattr__(name, module)

    def parameters(self) -> Iterable[IParameter]:
        """
        Return an iterable over this module's parameters (recursive).

        Returns
        -------
        Iterable[IParameter]
            Iterable of parameters registered on this module and all submodules.
        """
        # Yield own parameters first
        for p in self._parameters.values():
            yield p
        # Then recurse
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, IParameter]]:
        """
        Return an iterator over (name, parameter) pairs (recursive).

        Parameters
        ----------
        prefix : str
            Prefix to prepend to parameter names (used for recursion).

        Returns
        -------
        Iterator[tuple[str, IParameter]]
            Iterator yielding (fully_qualified_name, parameter).
        """
        base = prefix + "." if prefix else ""

        for name, p in self._parameters.items():
            yield (f"{base}{name}", p)

        for child_name, child in self._modules.items():
            child_prefix = f"{base}{child_name}"
            yield from child.named_parameters(child_prefix)

    def forward(self, x):
        """
        Execute the forward computation of the module.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Call the module as a function, delegating to `forward`.
        """
        return self.forward(x)

    # ------------------------------------------------------------------
    # Serialization hooks (opt-in contract)
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for this module.

        Subclasses that participate in JSON-based model save/load MUST
        override this method.

        Raises
        ------
        NotImplementedError
            If the module does not support JSON serialization.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_config(). "
            "This module cannot be serialized to JSON."
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Module":
        """
        Reconstruct a module from a JSON configuration.

        Subclasses that participate in JSON-based model save/load MUST
        override this method.

        Raises
        ------
        NotImplementedError
            If the module does not support JSON deserialization.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_config(). "
            "This module cannot be deserialized from JSON."
        )
