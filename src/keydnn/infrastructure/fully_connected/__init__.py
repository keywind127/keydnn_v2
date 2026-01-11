"""
Fully connected (linear) layer public exports.

This module defines the public API surface for dense / linear layers. It
re-exports multiple linear-layer variants under a single, stable namespace
to simplify imports and maintain consistency across the framework.

Exports
-------
- `Dense`: Keras-style fully connected layer.
- `LazyLinear`: Deferred-shape linear layer that initializes parameters on
  first forward pass.
- `Linear`: Core linear transformation module.

Design intent
-------------
- Provide a unified import location for all linear layer variants.
- Support both eager and lazy parameter initialization patterns.
- Align naming and behavior with common deep learning frameworks.
"""

from ._dense import Dense, LazyLinear
from ._linear import Linear


__all__ = [
    "Dense",
    "LazyLinear",
    "Linear",
]
