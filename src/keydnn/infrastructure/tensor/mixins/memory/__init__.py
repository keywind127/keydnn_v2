"""
Tensor memory operation registrations for KeyDNN.

This package aggregates backend-specific implementations of memory-related
`Tensor` operations and registers them through the tensor control-path
dispatch system. The imported modules provide concrete CPU and CUDA handlers
for methods defined on `TensorMixinMemory`, including:

- `fill`              : in-place scalar fill (CPU / CUDA)
- `clone`             : deep copy of tensor storage (CPU / CUDA)
- `to_numpy`          : materialize tensor data as a NumPy ndarray
- `transpose`         : 2D transpose with autograd support
- `broadcast_to`      : explicit broadcast expansion with gradient reduction
- `copy_from_numpy`   : copy data from NumPy into tensor storage

Each operation is registered via `tensor_control_path_manager`, allowing
runtime dispatch based on:
- the mixin (`TensorMixinMemory`)
- the logical Tensor method (e.g. `fill`, `clone`)
- the target device (`cpu`, `cuda:0`, etc.)

Public API
----------
Only `TensorMixinMemory` is re-exported as part of the public interface.
All concrete backend implementations are imported for their side effects
(registration) and are not intended to be accessed directly.
"""

from ._tensor_fill import *
from ._tensor_clone import *
from ._tensor_tonumpy import *
from ._tensor_transpose import *
from ._tensor_broadcast import *
from ._tensor_sum_to_shape import *
from ._tensor_copyfromnumpy import *
from ._base import TensorMixinMemory

__all__ = [
    TensorMixinMemory.__name__,
]
