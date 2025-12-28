"""
Reduction mixins and device-specific implementations for Tensor operations.

This package aggregates reduction-related Tensor mixins and their concrete
control-path implementations, including:

- ``max``  : maximum reduction
- ``mean`` : arithmetic mean reduction
- ``sum``  : summation reduction

Each reduction operation is implemented using the control-path dispatch
mechanism, allowing device-specific (e.g., CPU, CUDA) implementations to be
selected at runtime while exposing a single, stable public API on the Tensor
class.

Public API
----------
Only the base mixin class is exported as part of the public interface:

- ``TensorMixinReduction``

The concrete implementations (e.g., ``_tensor_max``, ``_tensor_mean``,
``_tensor_sum``) are imported for side effects so that their control paths are
registered, but they are not intended to be used directly.
"""

from ._tensor_max import *
from ._tensor_mean import *
from ._tensor_sum import *
from ._base import TensorMixinReduction

__all__ = [
    TensorMixinReduction.__name__,
]
