"""
Comparison mixins and device-specific implementations for Tensor operations.

This package groups elementwise comparison functionality for tensors together
with their concrete device-specific control-path implementations. It provides
the public comparison API via a declarative mixin, while registering concrete
CPU and CUDA implementations through control-path dispatch.

Supported comparison operators include:

- Greater-than          (``__gt__``)
- Greater-than-or-equal (``__ge__``)
- Less-than             (``__lt__``)
- Less-than-or-equal    (``__le__``)
- Equality              (``__eq__``)
- Not-equal             (``__ne__``)

Design notes
------------
- Concrete implementation modules are imported for their *side effects*:
  importing them registers device-specific control paths with the
  tensor control-path manager.
- Backend-specific logic (CPU vs CUDA) is isolated in implementation modules,
  keeping the base mixin declarative and backend-agnostic.
- CUDA implementations may dispatch to dedicated scalar kernels when the
  right-hand operand is a scalar, avoiding temporary tensor materialization.
- Broadcasting is intentionally not supported; tensor–tensor comparisons
  require identical shapes.
- Comparison operators intentionally do **not** participate in autograd;
  all comparison results have ``requires_grad=False`` and produce numeric
  float32 mask tensors (``1.0`` for True, ``0.0`` for False).

Public API
----------
Only the base mixin class is exported as part of the public interface:

- ``TensorMixinComparison``

All concrete comparison operators are exposed indirectly through this mixin
and resolved at runtime via control-path dispatch.
"""

from ._tensor_gt import *
from ._tensor_ge import *
from ._tensor_lt import *
from ._tensor_le import *
from ._tensor_eq import *
from ._tensor_ne import *
from ._base import TensorMixinComparison

__all__ = [
    TensorMixinComparison.__name__,
]
