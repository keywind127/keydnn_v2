"""
CUDA bias-add primitives with Tensor boundaries (device-pointer based).

This module provides Tensor-facing wrappers for the CUDA bias-add kernels
exposed via `native_cuda.python.ops.bias_add_ctypes`. It implements the common
linear-layer bias pattern for 2D activations:

- Forward (out-of-place): `y = x + b`
- In-place: `y += b`

where:
- `x` / `y` have shape (batch, out_features)
- `b` has shape (out_features,)

Design
------
- Inputs are KeyDNN `Tensor` objects that must live on CUDA.
- `Tensor.data` is treated as a raw device pointer (uintptr_t as Python int).
- Forward allocates an output device buffer and returns a CUDA `Tensor` via
  `Tensor._from_devptr`.
- In-place updates mutate the existing `y` device buffer and return None.

Validation and constraints
--------------------------
- Only the specific bias pattern above is supported (no general broadcasting).
- `x`/`y` must be 2D and `b` must be 1D with matching `out_features`.
- Tensors must be on the same CUDA device (checked via string comparison).
- Supported dtypes: float32 / float64, and `b.dtype` must match `x/y.dtype`.

Resource management
-------------------
- `bias_add_forward` allocates `y_dev` via `cuda_malloc` and frees it on failure
  before re-raising.
- `bias_add_inplace` performs no allocation.

Notes
-----
The `sync` argument is accepted for API symmetry with other CUDA ops. The ctypes
wrappers used here do not expose an explicit synchronization flag; any
synchronization is handled by the underlying native implementation or by the
caller at a higher level.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from ..native_cuda.python.ops.bias_add_ctypes import (
    bias_add_cuda as _bias_add,
    bias_add_inplace_cuda as _bias_add_inplace,
)


def _numel(shape: Tuple[int, ...]) -> int:
    """
    Compute the number of elements for a tensor shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Tensor shape.

    Returns
    -------
    int
        Product of all dimensions in `shape` (1 for an empty shape).
    """
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def bias_add_forward(
    x: Tensor, b: Tensor, *, device: int = 0, sync: bool = True
) -> Tensor:
    """
    Compute bias add on CUDA: `y = x + b`.

    Parameters
    ----------
    x : Tensor
        CUDA tensor of shape (batch, out_features), dtype float32/float64.
    b : Tensor
        CUDA tensor of shape (out_features,), same dtype/device as `x`.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device` before allocation and
        kernel launch. Defaults to 0.
    sync : bool, optional
        Accepted for API symmetry. The underlying ctypes wrapper does not take a
        sync flag; synchronization behavior is determined by the native kernel
        or the caller. Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor `y` with shape (batch, out_features) and the
        same dtype as `x`.

    Raises
    ------
    TypeError
        If `x` or `b` is not CUDA, or dtype is unsupported/mismatched.
    ValueError
        If shapes do not match the required bias pattern or devices differ.
    RuntimeError
        If allocation or the underlying CUDA kernel fails.

    Ownership
    ---------
    The returned Tensor owns its output device pointer. If an exception occurs
    before Tensor construction, the allocated device buffer is freed.
    """
    if not x.device.is_cuda() or not b.device.is_cuda():
        raise TypeError(
            f"bias_add_forward requires CUDA tensors, got x={x.device} b={b.device}"
        )
    if str(x.device) != str(b.device):
        raise ValueError(f"device mismatch: x.device={x.device} vs b.device={b.device}")

    if tuple(x.shape) != (int(x.shape[0]), int(x.shape[1])) or len(x.shape) != 2:
        raise ValueError(f"x must be 2D (batch, out), got shape={x.shape}")
    if len(b.shape) != 1:
        raise ValueError(f"b must be 1D (out,), got shape={b.shape}")
    if int(b.shape[0]) != int(x.shape[1]):
        raise ValueError(f"shape mismatch: x.shape={x.shape} vs b.shape={b.shape}")

    dt_x = np.dtype(x.dtype)
    dt_b = np.dtype(b.dtype)
    if dt_x not in (np.float32, np.float64) or dt_b != dt_x:
        raise TypeError(f"dtype mismatch/unsupported: x.dtype={dt_x}, b.dtype={dt_b}")

    batch = int(x.shape[0])
    out = int(x.shape[1])
    numel = batch * out
    nbytes = int(numel * dt_x.itemsize)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))
    y_dev = cuda_malloc(lib, nbytes)

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=x.device.index,
        dev_ptr=int(y_dev),
        nbytes=int(nbytes),
        dtype=dt_x,
    )

    try:
        _bias_add(
            lib,
            x_dev=int(x.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            batch=batch,
            out_features=out,
            dtype=dt_x,
        )
        _ = bool(sync)
        return Tensor._from_storage(
            storage,
            shape=(batch, out),
            dtype=dt_x,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, int(y_dev))
        raise


def bias_add_inplace(
    y: Tensor, b: Tensor, *, device: int = 0, sync: bool = True
) -> None:
    """
    Perform in-place bias add on CUDA: `y += b`.

    Parameters
    ----------
    y : Tensor
        CUDA tensor of shape (batch, out_features), dtype float32/float64.
        Updated in-place.
    b : Tensor
        CUDA tensor of shape (out_features,), same dtype/device as `y`.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device` before kernel launch.
        Defaults to 0.
    sync : bool, optional
        Accepted for API symmetry. The underlying ctypes wrapper does not take a
        sync flag; synchronization behavior is determined by the native kernel
        or the caller. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `y` or `b` is not CUDA, or dtype is unsupported/mismatched.
    ValueError
        If shapes do not match the required bias pattern or devices differ.
    RuntimeError
        If the underlying CUDA kernel fails.

    Notes
    -----
    This function does not allocate memory. Callers must ensure `y` has an
    allocated device buffer.
    """
    if not y.device.is_cuda() or not b.device.is_cuda():
        raise TypeError(
            f"bias_add_inplace requires CUDA tensors, got y={y.device} b={b.device}"
        )
    if str(y.device) != str(b.device):
        raise ValueError(f"device mismatch: y.device={y.device} vs b.device={b.device}")

    if len(y.shape) != 2:
        raise ValueError(f"y must be 2D (batch, out), got shape={y.shape}")
    if len(b.shape) != 1:
        raise ValueError(f"b must be 1D (out,), got shape={b.shape}")
    if int(b.shape[0]) != int(y.shape[1]):
        raise ValueError(f"shape mismatch: y.shape={y.shape} vs b.shape={b.shape}")

    dt_y = np.dtype(y.dtype)
    dt_b = np.dtype(b.dtype)
    if dt_y not in (np.float32, np.float64) or dt_b != dt_y:
        raise TypeError(f"dtype mismatch/unsupported: y.dtype={dt_y}, b.dtype={dt_b}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    _bias_add_inplace(
        lib,
        y_dev=int(y.data),
        b_dev=int(b.data),
        batch=int(y.shape[0]),
        out_features=int(y.shape[1]),
        dtype=dt_y,
    )
    _ = bool(sync)


__all__ = ["bias_add_forward", "bias_add_inplace"]
