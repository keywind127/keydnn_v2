"""
Flatten layer public exports.

This module defines the public API for the flatten operation, which reshapes
multi-dimensional tensors into a 2D representation suitable for fully
connected layers and other dense operations.

Exports
-------
- `FlattenFn`: Low-level functional flatten operator used by autograd.
- `Flatten`: High-level module wrapping the flatten operation.

Design intent
-------------
- Provide a stable import path for flatten-related functionality.
- Separate functional and module-level implementations cleanly.
- Align with the API structure used by other layer modules.
"""

from ._flatten_function import FlattenFn
from ._flatten_module import Flatten

__all__ = [
    FlattenFn.__name__,
    Flatten.__name__,
]
