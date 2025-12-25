"""
CUDA buffer initialization utilities.

This module provides thin Python wrappers around native CUDA kernels and
runtime calls used to initialize contiguous 1D device buffers. The helpers
here are intentionally low-level and operate directly on raw device pointers.

Implemented operations
----------------------
- `fill_cuda`:
    Fill a device buffer with a scalar floating-point value using a native
    CUDA kernel.
- `zeros_cuda`:
    Zero a device buffer using byte-wise `cudaMemset`.
- `ones_cuda`:
    Fill a device buffer with ones, with a robust fallback to host-to-device
    memcpy when the native fill kernel is unavailable or unstable.

Design notes
------------
- All functions treat `numel == 0` as a no-op to avoid invoking native kernels
  with invalid pointers.
- Only floating-point buffers (`float32`, `float64`) are supported.
- Synchronization is optional and controlled via the `sync` flag.
- This module is used internally by Tensor factory methods (e.g. `zeros`,
  `ones`) and is not intended as a public API.
"""

from __future__ import annotations

import numpy as np

from ..native_cuda.python.global_avgpool2d_ctypes import cuda_synchronize
from ..native_cuda.python.maxpool2d_ctypes import cuda_memset

# We need an HtoD memcpy for the ones() fallback
from ..native_cuda.python.avgpool2d_ctypes import cuda_memcpy_h2d


def fill_cuda(
    lib,
    *,
    y_dev: int,
    numel: int,
    value: float,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Fill a contiguous 1D device buffer with a scalar value.

    This function dispatches a native CUDA fill kernel that writes the same
    floating-point value into every element of a device buffer.

    Parameters
    ----------
    lib
        Loaded CUDA shared library handle providing the native fill kernel.
    y_dev : int
        Device pointer to the output buffer.
    numel : int
        Number of elements in the buffer.
    value : float
        Scalar value to write into each element.
    dtype : np.dtype
        Buffer data type. Must be `np.float32` or `np.float64`.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel launch.

    Notes
    -----
    - `numel == 0` is treated as a no-op to avoid native-kernel pointer
      constraints.
    - This function assumes the buffer is contiguous and 1D.
    """
    from ..native_cuda.python.fill_ctypes import cuda_fill as _fill

    if dtype not in (np.float32, np.float64):
        raise TypeError(f"fill_cuda supports float32/float64 only, got {dtype}")

    if int(numel) == 0:
        return

    _fill(
        lib,
        y_dev=int(y_dev),
        numel=int(numel),
        value=float(value),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def zeros_cuda(
    lib,
    *,
    y_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Fill a contiguous 1D device buffer with zeros.

    This function uses a byte-wise `cudaMemset` call to zero out the underlying
    memory region corresponding to the buffer.

    Parameters
    ----------
    lib
        Loaded CUDA shared library handle.
    y_dev : int
        Device pointer to the output buffer.
    numel : int
        Number of elements in the buffer.
    dtype : np.dtype
        Buffer data type. Must be `np.float32` or `np.float64`.
    sync : bool, optional
        If True, synchronizes the CUDA device after the memset.

    Notes
    -----
    - Byte-wise memset is safe for zero initialization of floating-point
      buffers.
    - `numel == 0` is treated as a no-op.
    """
    if dtype not in (np.float32, np.float64):
        raise TypeError(f"zeros_cuda supports float32/float64 only, got {dtype}")

    if int(numel) == 0:
        return

    nbytes = int(numel) * int(np.dtype(dtype).itemsize)
    cuda_memset(lib, int(y_dev), 0, int(nbytes))

    if sync:
        cuda_synchronize(lib)


def ones_cuda(
    lib,
    *,
    y_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Fill a contiguous 1D device buffer with ones.

    This function attempts to use the native CUDA scalar fill kernel. If the
    kernel invocation fails (e.g., due to an unimplemented or unstable native
    path), it falls back to materializing a host buffer filled with ones and
    performing a host-to-device memcpy.

    Parameters
    ----------
    lib
        Loaded CUDA shared library handle.
    y_dev : int
        Device pointer to the output buffer.
    numel : int
        Number of elements in the buffer.
    dtype : np.dtype
        Buffer data type. Must be `np.float32` or `np.float64`.
    sync : bool, optional
        If True, synchronizes the CUDA device after the operation.

    Strategy
    --------
    1. Attempt native scalar fill via `fill_cuda`.
    2. On failure, allocate a host NumPy array of ones and copy it to the
       device via `cuda_memcpy_h2d`.

    Notes
    -----
    - The fallback path prioritizes correctness over performance and is
      intended as a temporary safeguard while native kernels stabilize.
    - `numel == 0` is treated as a no-op.
    """
    if dtype not in (np.float32, np.float64):
        raise TypeError(f"ones_cuda supports float32/float64 only, got {dtype}")

    if int(numel) == 0:
        return

    try:
        fill_cuda(
            lib,
            y_dev=int(y_dev),
            numel=int(numel),
            value=1.0,
            dtype=dtype,
            sync=sync,
        )
        return
    except RuntimeError:
        # Fallback: materialize on host then memcpy H2D.
        # This preserves correctness and unblocks tests even if the native fill kernel
        # is not reliable yet.
        host = np.ones((int(numel),), dtype=np.dtype(dtype))
        cuda_memcpy_h2d(lib, int(y_dev), host)

        if sync:
            cuda_synchronize(lib)
