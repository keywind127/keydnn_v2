"""
Weight initialization public API.

This module aggregates and exposes all supported weight initialization
strategies, including Xavier (Glorot) and Kaiming (He) initializers, and
registers them into the global `WeightInitializer` registry via import
side effects.

Importing this module ensures that all built-in initializers are available
for lookup and dispatch through `WeightInitializer`.

Exports
-------
- WeightInitializer:
    The registry-backed initializer dispatcher used to apply a selected
    initialization strategy to tensors.

Notes
-----
- Individual initializer implementations are defined in submodules and
  registered at import time.
- This module intentionally re-exports only the dispatcher class as part
  of the public API; concrete initializer functions are accessed indirectly
  via registry names.
"""

from ._xavier import *
from ._kaiming import *
from ._base import WeightInitializer

__all__ = [
    WeightInitializer.__name__,
]
