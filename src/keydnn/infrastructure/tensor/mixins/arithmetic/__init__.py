"""
Arithmetic mixins and device-specific implementations for Tensor operations.

This package aggregates arithmetic-related Tensor mixins and their concrete
control-path implementations, including:

- true division      (``__truediv__`` / ``__rtruediv__``)
- addition           (``__add__`` / ``__radd__``)
- subtraction        (``__sub__`` / ``__rsub__``)
- multiplication     (``__mul__`` / ``__rmul__``)

Each arithmetic operation is implemented using the control-path dispatch
mechanism, allowing device-specific (e.g., CPU, CUDA) implementations to be
selected at runtime while exposing a single, stable public API on the Tensor
class.

Design notes
------------
- Concrete implementation modules are imported for their *side effects*:
  registering control paths with the tensor control-path manager.
- These implementation modules are not part of the public API and should not
  be imported directly by users.
- The base mixin defines the canonical operator signatures and backward
  semantics, while backend-specific logic is isolated elsewhere.

Public API
----------
Only the base mixin class is exported as part of the public interface:

- ``TensorMixinArithmetic``
"""

from ._tensor_division import *
from ._tensor_addition import *
from ._tensor_subtraction import *
from ._tensor_multiplication import *
from ._base import TensorMixinArithmetic

__all__ = [
    TensorMixinArithmetic.__name__,
]
