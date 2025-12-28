"""
Unary mixins and device-specific implementations for Tensor operations.

This package aggregates unary Tensor operations and their concrete control-path
implementations, including:

- ``exp``   : elementwise exponential
- ``log``   : elementwise natural logarithm
- ``sqrt``  : elementwise square root
- ``__neg__`` : elementwise negation (unary minus)

Each unary operation is implemented using the control-path dispatch mechanism,
allowing device-specific (e.g., CPU, CUDA) implementations to be selected at
runtime while exposing a single, stable public API on the Tensor class.

Design notes
------------
- Concrete implementation modules are imported for their *side effects*:
  registering control paths with the tensor control-path manager.
- These modules are not part of the public API and should not be imported
  directly by users.
- The base mixin defines the canonical method signatures and autograd semantics,
  while backend-specific logic is isolated in the implementation modules.

Public API
----------
Only the base mixin class is exported as part of the public interface:

- ``TensorMixinUnary``
"""

from ._tensor_exp import *
from ._tensor_log import *
from ._tensor_neg import *
from ._tensor_sqrt import *
from ._base import TensorMixinUnary

__all__ = [
    TensorMixinUnary.__name__,
]
