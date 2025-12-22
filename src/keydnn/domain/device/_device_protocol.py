"""
Device abstraction contracts for KeyDNN.

This module defines a duck-typed `DeviceLike` protocol that represents a
computation device descriptor (e.g., CPU or CUDA) without coupling to a
specific concrete class implementation.

By relying on structural typing instead of class identity, this abstraction
allows infrastructure- and domain-layer components to interoperate safely
without brittle `isinstance` checks or pattern matching on concrete device
types.

Design notes
------------
- Uses `typing.Protocol` and `@runtime_checkable` to enable both static and
  runtime validation of device-like objects.
- Avoids direct dependency on a concrete `Device` class, improving modularity
  and testability.
- Supports multiple backends (CPU, CUDA) while keeping higher layers backend-
  agnostic.
- Enables flexible extension to additional device types in the future without
  modifying existing call sites.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class DeviceLike(Protocol):
    """
    Duck-typed device contract.

    Any object that provides these members can be used as a computation device
    descriptor within the framework, regardless of its concrete class identity.

    Notes
    -----
    This Protocol exists to avoid brittle class-identity coupling introduced by
    `match/case Device()` and to support structural typing across layers.
    """

    type: object
    index: Optional[int]

    def is_cpu(self) -> bool: ...
    def is_cuda(self) -> bool: ...
    def __str__(self) -> str: ...
