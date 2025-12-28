"""
Tensor clone implementations (CPU and CUDA) for KeyDNN.

This module provides backend-specific implementations of `Tensor.clone()` and
registers them via `tensor_control_path_manager`:

- `tensor_clone_cpu`: clones a CPU tensor by deep-copying its NumPy buffer.
- `tensor_clone_gpu`: clones a CUDA tensor by allocating a new device buffer and
  performing a device-to-device memcpy.

Design notes
------------
- `clone()` is treated as a raw storage copy: returned tensors do not carry an
  autograd context (`ctx=None`) and default to `requires_grad=False`.
- CUDA cloning is implemented as a direct D2D copy to avoid unnecessary host
  round-trips.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM
from typing import Union, Type


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.clone, Device("cuda:0"))
def tensor_clone_gpu(self: ITensor) -> "ITensor":
    """
    Clone a CUDA tensor by allocating a new device buffer and copying bytes (D2D).

    This function implements `Tensor.clone()` for CUDA tensors registered on
    device `"cuda:0"`. It ensures both source and destination tensors have valid
    device allocations and then performs a device-to-device memcpy for the full
    contiguous buffer.

    Returns
    -------
    ITensor
        A new tensor on the same CUDA device with identical contents.

    Notes
    -----
    - The clone is a raw storage copy: `requires_grad=False` and `ctx=None`.
    - The copy size is computed as `numel(self.shape) * itemsize(dtype)`.
    - If `numel == 0`, no memcpy is performed.
    """
    import numpy as np

    Tensor = type(self)

    # -------------------------
    # CUDA path
    # -------------------------

    from ....native_cuda.python.ops import memcpy_ctypes as mc

    y = Tensor(shape=self.shape, device=self.device, requires_grad=False, ctx=None)
    y._ensure_cuda_alloc(dtype=getattr(self, "dtype", np.float32))

    self._ensure_cuda_alloc(dtype=getattr(self, "dtype", np.float32))

    lib = self._get_cuda_lib()

    dtype = getattr(self, "dtype", np.float32)
    itemsize = int(np.dtype(dtype).itemsize)

    numel = 1
    for d in self.shape:
        numel *= int(d)

    nbytes = int(numel * itemsize)

    if nbytes > 0:
        mc.memcpy_dtod(
            lib,
            dst_dev=int(y.data),
            src_dev=int(self.data),
            nbytes=nbytes,
            sync=True,
        )

    return y


@tensor_control_path_manager(TMM, TMM.clone, Device("cpu"))
def tensor_clone_cpu(self: ITensor) -> "ITensor":
    """
    Clone a CPU tensor by deep-copying its NumPy buffer.

    This function implements `Tensor.clone()` for CPU tensors. It materializes
    the source tensor as a NumPy array, deep-copies it, and writes the result
    into a newly created CPU tensor.

    Returns
    -------
    ITensor
        A new CPU tensor with identical contents.

    Notes
    -----
    - The clone is a raw storage copy: `requires_grad=False` and `ctx=None`.
    - The implementation uses `self.to_numpy().copy()` to ensure independence
      from the original tensor's storage.
    """
    Tensor = type(self)

    out = Tensor(shape=self.shape, device=self.device, requires_grad=False, ctx=None)

    # -------------------------
    # CPU path
    # -------------------------
    out.copy_from_numpy(self.to_numpy().copy())
    return out
