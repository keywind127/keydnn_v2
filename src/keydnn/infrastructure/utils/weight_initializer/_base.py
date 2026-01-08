"""
Weight initializer registry and dispatch utilities.

This module defines the concrete `WeightInitializer` used by the infrastructure
layer to apply registered weight initialization strategies (e.g. Kaiming,
Xavier) to `Tensor` instances.

Design
------
- Initializers are registered by string name via a decorator-based registry.
- Each initializer is a callable that mutates a tensor *in-place* and returns it.
- The dispatcher resolves an initializer by name at construction time and
  invokes it via `__call__`.

Separation of concerns
----------------------
- The registry and dispatch mechanism lives in the infrastructure layer.
- The mathematical definitions of initialization strategies are implemented
  as standalone functions and registered here.
- The domain layer depends only on the abstract `_WeightInitializer` contract
  and is agnostic to concrete initialization policies.

Usage example
-------------
Registering an initializer:

    @WeightInitializer.register_initializer("kaiming")
    def kaiming(tensor: Tensor) -> Tensor:
        ...

Applying an initializer:

    init = WeightInitializer("kaiming")
    init(weight_tensor)

Notes
-----
- Registration keys must be unique unless explicitly overwritten.
- Initializers are expected to handle fan-in / fan-out computation internally
  based on the tensor shape.
- This design enables easy extension and avoids hard-coding initialization
  logic into layer implementations.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, TypeVar

from ....domain.utils._weight_initialization import _WeightInitializer
from ...tensor._tensor import Tensor

T = TypeVar("T", bound=Callable[..., Tensor])


class WeightInitializer(_WeightInitializer):
    """
    Registry-backed weight initializer dispatcher.

    Usage
    -----
    Register:
        @WeightInitializer.register_initializer("kaiming")
        def kaiming(tensor: Tensor) -> Tensor: ...

    Dispatch:
        init = WeightInitializer("kaiming")
        init(tensor)

    Notes
    -----
    - Initializers are stored by string name in a class-level registry.
    - The initializer callable should mutate `tensor` in-place and return it.
    """

    INITIALIZERS: ClassVar[Dict[str, Callable[..., Tensor]]] = {}

    def __init__(self, initializer_name: str) -> None:
        try:
            self._initializer: Callable[..., Tensor] = self.INITIALIZERS[
                initializer_name
            ]
        except KeyError as e:
            available = ", ".join(sorted(self.INITIALIZERS)) or "<none>"
            raise ValueError(
                f"Unsupported initializer name: {initializer_name!r}. "
                f"Available: {available}"
            ) from e

    @classmethod
    def register_initializer(
        cls, name: str, *, overwrite: bool = False
    ) -> Callable[[T], T]:
        """
        Decorator to register a weight initializer under `name`.

        Parameters
        ----------
        name:
            Registry key used to retrieve the initializer later.
        overwrite:
            If False (default), raises if `name` is already registered.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Initializer name must be a non-empty string")

        def decorator(func: T) -> T:
            if not overwrite and name in cls.INITIALIZERS:
                raise ValueError(f"Initializer already registered: {name!r}")
            cls.INITIALIZERS[name] = func
            return func

        return decorator

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """Return registered initializer names (sorted)."""
        return tuple(sorted(cls.INITIALIZERS))

    @classmethod
    def get(cls, name: str) -> Callable[..., Tensor]:
        """Get a registered initializer callable by name."""
        return cls.INITIALIZERS[name]

    def __call__(self, tensor: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self._initializer(tensor, *args, **kwargs)
