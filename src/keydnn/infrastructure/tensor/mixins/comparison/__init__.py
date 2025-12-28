"""
Comparison mixins and device-specific implementations for Tensor operations.

This package groups comparison-related Tensor functionality and their concrete
control-path implementations. At present, it includes elementwise comparison
operators such as greater-than (``__gt__``), with additional comparisons
(e.g., ``__ge__``, ``__lt__``, ``__le__``) composed at the mixin level.

Design notes
------------
- Concrete implementation modules are imported for their *side effects*:
  registering device-specific control paths with the tensor control-path
  manager.
- Backend-specific logic (CPU, CUDA) is isolated in implementation modules,
  keeping the mixin definition declarative and backend-agnostic.
- Comparison operators intentionally do **not** participate in autograd;
  all comparison results have ``requires_grad=False``.

Public API
----------
Only the base mixin class is exported as part of the public interface:

- ``TensorMixinComparison``
"""

from ._tensor_gt import *
from ._base import TensorMixinComparison

__all__ = [
    TensorMixinComparison.__name__,
]
