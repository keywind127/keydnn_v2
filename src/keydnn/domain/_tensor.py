"""
Tensor interface definitions.

This module defines the domain-level interface for tensor-like objects using
structural typing. The interface captures the minimal, backend-agnostic
properties required for tensors to participate in computation graphs,
modules, and optimization workflows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from ._device import Device


@runtime_checkable
class ITensor(Protocol):
    """
    Domain-level tensor interface.

    An `ITensor` represents a multi-dimensional array that participates in
    numerical computation and, optionally, automatic differentiation.

    This interface defines the minimal contract required by the domain layer,
    allowing concrete tensor implementations to vary across backends (e.g.,
    NumPy, CUDA) without affecting higher-level logic.

    Notes
    -----
    - Structural typing (duck typing) is used instead of inheritance.
    - This interface is intentionally minimal and may be extended by
      infrastructure-level implementations.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the tensor.

        The shape is represented as a tuple of integers, where each element
        corresponds to the size of the tensor along a particular dimension.

        Returns
        -------
        tuple[int, ...]
            The tensor's shape.
        """
        ...

    @property
    def device(self) -> Device:
        """
        Return the device on which this tensor resides.

        The device indicates where the tensor's underlying data is stored
        and where computations involving this tensor should be executed.

        Returns
        -------
        Device
            The device associated with this tensor.
        """
        ...
