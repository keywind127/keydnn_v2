"""
Tensor fill implementations (CPU and CUDA) for KeyDNN.

This module provides backend-specific implementations of `Tensor.fill()` and
registers them via `tensor_control_path_manager`:

- `tensor_fill_cpu`: fills the CPU tensor in-place using NumPy's `.fill(...)`.
- `tensor_fill_gpu`: fills the CUDA tensor in-place using a native CUDA fill op
  exposed through the ops layer.

Notes
-----
- Both implementations mutate the tensor in-place and return `None`.
- The CUDA implementation ensures that a device buffer is allocated prior to
  invoking the native kernel.
- These functions are registered for `Device("cpu")` and `Device("cuda:0")`.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM
from typing import Union, Type


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.fill, Device("cuda:0"))
def tensor_fill_gpu(self: ITensor, value: float) -> None:
    """
    Fill a CUDA tensor in-place with a scalar value.

    This function implements `Tensor.fill()` for CUDA tensors registered on
    `"cuda:0"`. It ensures the tensor has an allocated device buffer and then
    dispatches to a native CUDA fill routine.

    Parameters
    ----------
    value : float
        Scalar value to write into every element of the tensor.

    Notes
    -----
    - This operation mutates the tensor in-place and returns `None`.
    - Allocation is ensured via `_ensure_cuda_alloc(dtype=self.dtype)` before
      invoking the CUDA kernel.
    - The CUDA device index is derived from `self.device.index` and forwarded to
      the ops layer.
    """
    # ensure device buffer exists before calling native fill
    self._ensure_cuda_alloc(dtype=self.dtype)

    from ....ops.fill_cuda_ext import fill_ as _fill_cuda_

    _fill_cuda_(self, float(value), device=int(self.device.index or 0), sync=True)


@tensor_control_path_manager(TMM, TMM.fill, Device("cpu"))
def tensor_fill_cpu(self: ITensor, value: float) -> None:
    """
    Fill a CPU tensor in-place with a scalar value.

    This function implements `Tensor.fill()` for CPU tensors. It delegates to the
    underlying NumPy array's `.fill(...)` method.

    Parameters
    ----------
    value : float
        Scalar value to write into every element of the tensor.

    Notes
    -----
    - This operation mutates the tensor in-place and returns `None`.
    - Dtype casting follows NumPy semantics of the underlying buffer.
    """
    self.data.fill(value)
