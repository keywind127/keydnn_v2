"""
Tensor -> NumPy materialization (CPU and CUDA) for KeyDNN.

This module provides backend-specific implementations of `Tensor.to_numpy()` and
registers them via `tensor_control_path_manager`:

- `tensor_tonumpy_cpu`: returns the underlying CPU storage (NumPy ndarray).
- `tensor_tonumpy_gpu`: allocates a host NumPy array and copies device memory
  into it via CUDA ctypes wrappers.

CUDA implementation details
---------------------------
- Uses a shared, cached CUDA DLL handle stored on the concrete `Tensor` type
  (`Tensor._CUDA_LIB`) to avoid issues caused by loading multiple handles to the
  same native library.
- Determines dtype from `self.dtype` with a fallback to `self._dtype`.
- Allocates a C-contiguous host array and performs a device-to-host memcpy.

Notes
-----
- These functions are registered for `Device("cpu")` and `Device("cuda:0")`.
- The CUDA path requires the native CUDA ctypes wrappers to be importable.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM
from typing import Union, Type

import numpy as np


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.to_numpy, Device("cuda:0"))
def tensor_tonumpy_gpu(self: ITensor) -> np.ndarray:
    """
    Materialize a CUDA tensor as a NumPy ndarray by copying device memory to host.

    This function implements `Tensor.to_numpy()` for CUDA tensors registered on
    `"cuda:0"`. It allocates a C-contiguous NumPy array on the host and performs
    a device-to-host memcpy using the native CUDA ctypes wrappers.

    Returns
    -------
    np.ndarray
        A newly allocated NumPy array containing a copy of the tensor data.

    Raises
    ------
    RuntimeError
        If the native CUDA ctypes wrappers are unavailable, or if the tensor's
        dtype cannot be determined.

    Notes
    -----
    - A shared CUDA DLL handle is cached on the concrete tensor class under
      `Tensor._CUDA_LIB` to avoid multi-handle loading issues.
    - Dtype is resolved from `self.dtype`, with a fallback to `self._dtype`.
    - The returned array is always a host copy and does not share storage with
      the device buffer.
    """
    Tensor = type(self)

    # -----------------------
    # CUDA path
    # -----------------------

    # Helper: share a single CUDA DLL handle to avoid multi-handle issues.
    def _get_cuda_lib():
        """
        Return (and cache) a singleton handle to the native CUDA library.

        Caches the loaded shared library handle on the Tensor class to prevent
        repeated loads that can lead to multiple-handle conflicts in ctypes.
        """
        lib = getattr(Tensor, "_CUDA_LIB", None)
        if lib is None:
            from ....native_cuda.python import maxpool2d_ctypes as m

            lib = m.load_keydnn_cuda_native()
            setattr(Tensor, "_CUDA_LIB", lib)
        return lib

    try:
        from ....native_cuda.python import maxpool2d_ctypes as m
    except Exception as e:
        raise RuntimeError(
            f"to_numpy() CUDA transfer requires native CUDA ctypes wrappers: {e!r}"
        )

    # Figure out dtype
    dtype = getattr(self, "dtype", None)
    if dtype is None:
        # Some codebases store dtype in _dtype; try that as a fallback.
        dtype = getattr(self, "_dtype", None)
    if dtype is None:
        raise RuntimeError("CUDA Tensor has no dtype; cannot materialize to NumPy.")

    dtype = np.dtype(dtype)

    # Allocate host array and copy device -> host
    out = np.empty(self.shape, dtype=dtype, order="C")

    dev_ptr = int(getattr(self, "data"))
    nbytes = int(out.nbytes)

    lib = _get_cuda_lib()
    # Expected signature: (lib, dst_host_ndarray, src_dev_ptr, nbytes)
    m.cudaMemcpyDtoH(lib, out, dev_ptr, nbytes)

    return out


@tensor_control_path_manager(TMM, TMM.to_numpy, Device("cpu"))
def tensor_tonumpy_cpu(self: ITensor) -> np.ndarray:
    """
    Return the underlying NumPy storage for a CPU tensor.

    This function implements `Tensor.to_numpy()` for CPU tensors. It returns
    the tensor's backing NumPy ndarray directly.

    Returns
    -------
    np.ndarray
        The underlying CPU array.

    Notes
    -----
    - This preserves the existing CPU behavior (may be a view into internal
      storage depending on the concrete tensor implementation).
    """
    # -----------------------
    # CPU path (unchanged)
    # -----------------------
    return self.data
