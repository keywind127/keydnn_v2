"""
Public random seeding API for KeyDNN.

This module re-exports the framework's canonical random seeding utilities
from the infrastructure layer, providing a stable and user-facing entry
point for controlling reproducibility.

Users should prefer importing and calling functions from this module
rather than accessing infrastructure internals directly.

Examples
--------
>>> import keydnn as kd
>>> kd.random.seed(42)

Notes
-----
- This API currently controls Python and NumPy randomness only.
- Future extensions may include device-specific or per-operation RNG
  management without changing this public interface.
"""

from ....infrastructure.random import seed, get_seed


__all__ = [
    "seed",
    "get_seed",
]
