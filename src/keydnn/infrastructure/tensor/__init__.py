"""
Tensor public exports.

This module exposes the core `Tensor` abstraction used throughout the framework.
`Tensor` represents multi-dimensional data with optional gradient tracking and
device placement, and serves as the foundational building block for all
computations, layers, and models.

Design intent
-------------
- Provide a stable, canonical import path for the `Tensor` type.
- Centralize the public-facing tensor API.
- Hide internal tensor implementation details from consumers.
"""

from ._tensor import Tensor


__all__ = [
    Tensor.__name__,
]
