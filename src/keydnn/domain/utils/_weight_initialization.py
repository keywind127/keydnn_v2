"""
Abstract interfaces and utilities for weight initialization.

This module defines the abstract base class for weight initializers used
throughout the framework, along with shared helper functions for computing
fan-in and fan-out values from tensor shapes.

The concrete implementation and registry logic live in the infrastructure
layer. This module exists in the domain layer to define contracts and shared
mathematical utilities without binding to any specific backend.
"""

from typing import Callable, Dict, TypeVar
from abc import ABC

from .._tensor import ITensor


T = TypeVar("T", bound=Callable[..., ITensor])


class _WeightInitializer(ABC):
    """
    Abstract base class for weight initializer dispatchers.

    This class defines the contract for registry-based weight initialization.
    Concrete subclasses are responsible for implementing registry behavior
    and dispatch logic.

    Design notes
    ------------
    - Initializers are identified by string names.
    - Each initializer is a callable that mutates a tensor in-place and
      returns it.
    - This class does not prescribe how initializers are stored or invoked;
      it only defines the expected interface.
    """

    INITIALIZERS: Dict[str, Callable] = {}

    def __init__(self, initializer_name: str) -> None:
        """
        Construct a weight initializer dispatcher.

        Parameters
        ----------
        initializer_name:
            The string key identifying a registered initializer.
        """
        ...

    @classmethod
    def register_initializer(
        cls, name: str, *, overwrite: bool = False
    ) -> Callable[[T], T]:
        """
        Register a weight initializer under a given name.

        This method returns a decorator that associates a callable with the
        provided registry key.

        Parameters
        ----------
        name:
            Name used to identify the initializer.
        overwrite:
            Whether to allow overwriting an existing registration.

        Returns
        -------
        Callable
            A decorator that registers the initializer function.
        """
        ...

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """
        Return the names of all registered initializers.

        Returns
        -------
        tuple[str, ...]
            A sorted tuple of registered initializer names.
        """
        ...

    @classmethod
    def get(cls, name: str) -> Callable[..., ITensor]:
        """
        Get a registered initializer callable by name.

        Parameters
        ----------
        name:
            Name of the initializer.

        Returns
        -------
        Callable[..., ITensor]
            The initializer callable associated with the given name.
        """
        ...

    def __call__(self, tensor: ITensor, *args, **kwargs) -> ITensor:
        """
        Apply the initializer to a tensor.

        Parameters
        ----------
        tensor:
            The tensor to be initialized.
        *args, **kwargs:
            Optional arguments forwarded to the initializer.

        Returns
        -------
        ITensor
            The initialized tensor.
        """
        ...


def _calculate_fan_in(shape: tuple[int, ...]) -> int:
    """
    Compute the fan-in value for a tensor shape.

    Fan-in is defined as the number of input connections contributing to a
    single output unit.

    Parameters
    ----------
    shape:
        Shape of the weight tensor.

    Returns
    -------
    int
        The computed fan-in value.
    """
    if len(shape) == 0:
        return 1  # scalar
    if len(shape) == 1:
        # bias-like vector; fan_in isn't really defined
        return shape[0]
    if len(shape) == 2:
        # Linear: (out_features, in_features)
        return shape[1]
    # ConvNd: (out_channels, in_channels, k1, k2, ...)
    receptive_field = 1
    for d in shape[2:]:
        receptive_field *= int(d)
    return int(shape[1]) * receptive_field


def _calculate_fan_in_and_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    """
    Compute both fan-in and fan-out values for a tensor shape.

    Fan-in represents the number of inputs to a single output unit, while
    fan-out represents the number of outputs influenced by a single input
    unit.

    Parameters
    ----------
    shape:
        Shape of the weight tensor.

    Returns
    -------
    tuple[int, int]
        A tuple of (fan_in, fan_out).
    """
    if len(shape) == 0:
        return 1, 1  # scalar
    if len(shape) == 1:
        # Bias or vector parameter
        return shape[0], shape[0]
    if len(shape) == 2:
        # Linear: (out_features, in_features)
        fan_out, fan_in = shape
        return fan_in, fan_out

    # ConvNd: (out_channels, in_channels, k1, k2, ...)
    receptive_field = 1
    for d in shape[2:]:
        receptive_field *= int(d)

    fan_in = int(shape[1]) * receptive_field
    fan_out = int(shape[0]) * receptive_field
    return fan_in, fan_out
