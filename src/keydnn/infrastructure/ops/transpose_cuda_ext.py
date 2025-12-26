# src/keydnn/infrastructure/ops/transpose_cuda_ext.py
"""
CUDA transpose primitives with Tensor boundaries (device-pointer based).

This module is the Tensor-facing counterpart of `transpose_cuda.py`.

Key design
----------
- Operates on CUDA tensors without converting them to NumPy.
- Reads `Tensor.data` as a device pointer (DevPtr).
- Allocates output device buffers.
- Invokes the ops-layer function `transpose2d_cuda(...)` which calls the
  native DLL kernel (and may fallback internally to D2H->NumPy->H2D if needed).
- Wraps the output pointer into a CUDA `Tensor` via `Tensor._from_devptr`.

Scope
-----
- 2D transpose only:
    X: (rows, cols) -> Y: (cols, rows)
- Supported dtypes: float32 / float64
- No higher-rank permutation/broadcasting in this wrapper.

Ownership
---------
The returned Tensor owns the output device pointer. If an exception occurs
before Tensor construction, we free the allocated device memory.
"""

from __future__ import annotations

import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from .transpose_cuda import transpose2d_cuda as _transpose2d_devptr


def _require_cuda(x: Tensor, name: str) -> None:
    """Validate that a tensor is on CUDA."""
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """Validate dtype is float32/float64 and return dtype."""
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_2d(x: Tensor, name: str) -> tuple[int, int]:
    """Validate that a tensor is 2D and return (rows, cols)."""
    if len(x.shape) != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    r, c = int(x.shape[0]), int(x.shape[1])
    if r <= 0 or c <= 0:
        raise ValueError(f"{name} must have positive dimensions, got shape={x.shape}")
    return r, c


def transpose2d_forward(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Transpose a 2D CUDA tensor: y = x.T

    Parameters
    ----------
    x : Tensor
        CUDA Tensor of shape (rows, cols), dtype float32/float64.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer transpose. Default True.

    Returns
    -------
    Tensor
        CUDA Tensor of shape (cols, rows), same dtype as `x`.

    Raises
    ------
    TypeError
        If `x` is not CUDA or dtype is not float32/float64.
    ValueError
        If `x` is not 2D.

    Notes
    -----
    - This wrapper allocates `y` device memory and returns it wrapped as Tensor.
    - The ops-layer `transpose2d_cuda` may internally fallback to a correctness
      path using memcpy+NumPy transpose if the native kernel reports failure.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")
    rows, cols = _require_2d(x, "x")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(rows * cols * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    try:
        _transpose2d_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            rows=int(rows),
            cols=int(cols),
            dtype=np.dtype(dt),  # important: ops expects a numpy dtype instance
            sync=bool(sync),
        )

        return Tensor._from_devptr(
            int(y_dev),
            shape=(cols, rows),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )

    except Exception:
        cuda_free(lib, y_dev)
        raise


# Convenience aliases (match ops-layer naming)
transpose2d = transpose2d_forward
transpose = transpose2d_forward

__all__ = ["transpose2d_forward", "transpose2d", "transpose"]
