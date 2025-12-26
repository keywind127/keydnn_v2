# src/keydnn/infrastructure/ops/unary_cuda_ext.py
"""
CUDA unary primitives with Tensor boundaries (device-pointer based).

This module is the Tensor-facing counterpart of `unary_cuda.py`.

Key design
----------
- Accepts CUDA `Tensor` inputs (no NumPy conversion for CUDA tensors).
- Treats `Tensor.data` as a device pointer (DevPtr).
- Allocates output buffers on device.
- Invokes ops-layer functions (e.g., `exp_cuda`) which call the native DLL.
- Returns output as CUDA `Tensor` via `Tensor._from_devptr`.

Scope
-----
- Elementwise unary ops on contiguous 1D device buffers, wrapped here for
  arbitrary-shaped tensors by using `numel = prod(shape)`.
- Currently implemented:
    - exp_forward: y = exp(x)

Supported dtypes
----------------
- float32 / float64 only (consistent with native/ops-layer restrictions).

Ownership
---------
The returned Tensor owns its output device pointer. If an exception occurs
before Tensor construction, the allocated device buffer is freed.
"""

from __future__ import annotations

import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from .unary_cuda import exp_cuda as _exp_devptr


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


def _numel(shape) -> int:
    """Compute number of elements for a shape tuple."""
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def exp_forward(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Elementwise exp on CUDA: y = exp(x)

    Parameters
    ----------
    x : Tensor
        CUDA Tensor of any shape, dtype float32/float64.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Accepted for API symmetry; forwarded to ops-layer exp (ignored there).
        Default True.

    Returns
    -------
    Tensor
        CUDA Tensor with the same shape/dtype as `x`.

    Raises
    ------
    TypeError
        If `x` is not CUDA or dtype is not float32/float64.
    ValueError
        If `x` has numel <= 0.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    numel = _numel(x.shape)
    if numel <= 0:
        raise ValueError(f"exp_forward requires numel > 0, got {numel}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(numel * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    try:
        _exp_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            numel=int(numel),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )

        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(int(d) for d in x.shape),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )

    except Exception:
        cuda_free(lib, y_dev)
        raise


# Convenience aliases
exp = exp_forward
cuda_exp = exp_forward

__all__ = ["exp_forward", "exp", "cuda_exp"]
