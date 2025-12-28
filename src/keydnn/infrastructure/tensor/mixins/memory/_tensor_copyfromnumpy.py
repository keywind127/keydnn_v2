"""
NumPy -> Tensor copy implementations (CPU and CUDA) for KeyDNN.

This module provides backend-specific implementations of `Tensor.copy_from_numpy`
and registers them via `tensor_control_path_manager`:

- `tensor_copyfromnumpy_cpu`: copies from a NumPy (or array-like) value into the
  tensor's CPU storage using NumPy assignment.
- `tensor_copyfromnumpy_gpu`: copies from a NumPy (or array-like) value into the
  tensor's CUDA storage using a host-to-device memcpy.

Backward-compatibility guarantees
---------------------------------
Both implementations preserve the legacy behavior of `copy_from_numpy`:
- Accept array-like inputs (NumPy arrays, NumPy scalars, Python scalars, lists).
- Normalize input via `np.asarray(..., dtype=self.dtype)`.
- Enforce a strict shape match, including scalar tensors with shape `()`.

CUDA behavior
-------------
The CUDA implementation ensures a destination device allocation exists, converts
the host array to a C-contiguous buffer, selects the appropriate CUDA device, and
performs a raw HtoD memcpy through the ops layer.

Notes
-----
- These functions are registered for `Device("cpu")` and `Device("cuda:0")`.
- Zero-byte transfers return immediately and do not invoke memcpy.
"""

from ..._tensor_builder import tensor_control_path_manager

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM
from typing import Union

import numpy as np


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.copy_from_numpy, Device("cuda:0"))
def tensor_copyfromnumpy_gpu(self: ITensor, arr: np.ndarray) -> None:
    """
    Copy data from a NumPy array (or array-like / scalar) into a CUDA tensor.

    The input is normalized to `np.asarray(arr, dtype=self.dtype)` and must match
    `self.shape` exactly (including scalar shape `()`).

    Parameters
    ----------
    arr : np.ndarray
        Source data. May be any array-like object accepted by `np.asarray`,
        including Python scalars and NumPy scalar types.

    Raises
    ------
    ValueError
        If the shape of `arr` does not match `self.shape`.
    RuntimeError
        If CUDA allocation fails (device pointer remains 0) for a non-empty copy.

    Notes
    -----
    - Ensures the destination device buffer exists by calling `_ensure_cuda_alloc`
      when `self.data == 0`.
    - Ensures the host buffer is C-contiguous before performing raw memcpy.
    - For empty tensors (0 bytes), this is a no-op.
    """
    import numpy as np

    # IMPORTANT: preserve legacy behavior:
    # - accept np.float32, python float, lists, etc.
    # - materialize as ndarray
    dt = np.dtype(self.dtype)  # default float32 in your codebase
    arr_nd = np.asarray(arr, dtype=dt)

    # Preserve original strict shape check (including scalar shape == ())
    if arr_nd.shape != self.shape:
        raise ValueError(f"Shape mismatch: tensor {self.shape} vs array {arr_nd.shape}")

    # -----------------------
    # CUDA path
    # -----------------------
    # Ensure device buffer exists for destination
    if int(self.data) == 0:
        self._ensure_cuda_alloc(dtype=dt)

    dst_dev = int(self.data)
    if dst_dev == 0 and arr_nd.nbytes != 0:
        raise RuntimeError(
            "CUDA copy_from_numpy: destination devptr is 0 after allocation"
        )

    # Make sure host buffer is contiguous
    x_c = np.ascontiguousarray(arr_nd)
    nbytes = int(x_c.nbytes)
    if nbytes == 0:
        return

    # Use ops-layer memcpy (HtoD)
    from ....ops.pool2d_cuda import cuda_set_device
    from ....ops.memcpy_cuda import memcpy_htod as _memcpy_htod

    lib = self._get_cuda_lib()
    cuda_set_device(lib, int(self.device.index or 0))

    _memcpy_htod(
        lib,
        dst_dev=int(dst_dev),
        src_host=x_c,
        nbytes=nbytes,
        sync=True,
    )
    return


@tensor_control_path_manager(TMM, TMM.copy_from_numpy, Device("cpu"))
def tensor_copyfromnumpy_cpu(self: ITensor, arr: np.ndarray) -> None:
    """
    Copy data from a NumPy array (or array-like / scalar) into a CPU tensor.

    The input is normalized to `np.asarray(arr, dtype=self.dtype)` and must match
    `self.shape` exactly (including scalar shape `()`).

    Parameters
    ----------
    arr : np.ndarray
        Source data. May be any array-like object accepted by `np.asarray`,
        including Python scalars and NumPy scalar types.

    Raises
    ------
    ValueError
        If the shape of `arr` does not match `self.shape`.

    Notes
    -----
    - This preserves the original CPU semantics by writing into `self.data[...]`.
    - Dtype conversion is performed by `np.asarray(..., dtype=self.dtype)`.
    """
    import numpy as np

    # IMPORTANT: preserve legacy behavior:
    # - accept np.float32, python float, lists, etc.
    # - materialize as ndarray
    dt = np.dtype(self.dtype)  # default float32 in your codebase
    arr_nd = np.asarray(arr, dtype=dt)

    # Preserve original strict shape check (including scalar shape == ())
    if arr_nd.shape != self.shape:
        raise ValueError(f"Shape mismatch: tensor {self.shape} vs array {arr_nd.shape}")

    # -----------------------
    # CPU path (original semantics)
    # -----------------------
    self.data[...] = arr_nd
    return
