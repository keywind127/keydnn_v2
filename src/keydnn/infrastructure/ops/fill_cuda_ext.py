# infrastructure/ops/fill_cuda_ext.py
"""
CUDA buffer initialization utilities with Tensor boundaries.

This module provides Tensor-first wrappers around device-pointer-first CUDA
initialization helpers implemented in `fill_cuda.py`. The goal is to offer
convenient, allocation-aware APIs for initializing CUDA-backed `Tensor` objects
without introducing unnecessary host round-trips.

Provided APIs
-------------
In-place (mutating existing device buffers):
- `fill_(x, value)`: fill an existing CUDA tensor with a scalar
- `zeros_(x)`: zero an existing CUDA tensor (typically via memset)
- `ones_(x)`: fill an existing CUDA tensor with ones (may use fallback path)

Out-of-place (allocate + fill):
- `zeros_like(x)`: allocate a new CUDA tensor with the same shape/dtype as `x`
  and fill with zeros
- `ones_like(x)`: allocate a new CUDA tensor with the same shape/dtype as `x`
  and fill with ones
- `full_like(x, value)`: allocate a new CUDA tensor with the same shape/dtype as
  `x` and fill with a scalar

Optional factories (primarily useful for tests / plumbing):
- `zeros(shape, dtype, device_obj)`
- `ones(shape, dtype, device_obj)`

Design and constraints
----------------------
- Validates CUDA placement at the Tensor boundary.
- Restricts dtypes to float32/float64 to match the underlying fill kernels.
- Treats empty tensors (numel == 0 / nbytes == 0) as valid: no-op fills and
  allocations use `y_dev = 0`.

Resource management
-------------------
Out-of-place helpers allocate device memory via `cuda_malloc`. If an exception
occurs before constructing the return Tensor, the allocated buffer is freed to
avoid leaks.

Notes
-----
- `fill_` uses `Tensor._get_cuda_lib()` when `lib` is None to share the same
  loaded library/runtime context as other Tensor CUDA operations.
- Some functions use `load_keydnn_cuda_native()` directly; if you standardize on
  `Tensor._get_cuda_lib()` later, this module can be updated without changing
  API behavior.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from ..tensor._tensor import Tensor
from ..native_cuda.python.avgpool2d_ctypes import load_keydnn_cuda_native
from ..native_cuda.python.maxpool2d_ctypes import (
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)

from .fill_cuda import fill_cuda, zeros_cuda, ones_cuda


def _numel(shape: Tuple[int, ...]) -> int:
    """
    Compute the number of elements for a shape.

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


def _require_cuda(x: Tensor, name: str) -> None:
    """
    Require that a tensor is placed on a CUDA device.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Name used in error messages.

    Raises
    ------
    TypeError
        If `x` is not a CUDA tensor.
    """
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64_dtype(dtype: np.dtype, name: str) -> np.dtype:
    """
    Require that a dtype is float32 or float64.

    Parameters
    ----------
    dtype : np.dtype
        Dtype to validate (any numpy-coercible dtype is accepted).
    name : str
        Name used in error messages.

    Returns
    -------
    np.dtype
        Normalized dtype (np.float32 or np.float64).

    Raises
    ------
    TypeError
        If dtype is not float32/float64.
    """
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """
    Require that a tensor dtype is float32 or float64.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Name used in error messages.

    Returns
    -------
    np.dtype
        Normalized dtype (np.float32 or np.float64).

    Raises
    ------
    TypeError
        If `x.dtype` is not float32/float64.
    """
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def fill_(
    x: Tensor,
    value: float,
    *,
    lib=None,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    In-place scalar fill for a CUDA tensor.

    Parameters
    ----------
    x : Tensor
        CUDA tensor whose device buffer will be filled in-place.
    value : float
        Scalar value to write into all elements of `x`.
    lib : ctypes.CDLL | None, optional
        Native library handle. If None, uses `Tensor._get_cuda_lib()` so the
        same shared handle is used across the codebase.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in the underlying fill
        helper (`fill_cuda`). Defaults to True.

    Returns
    -------
    Tensor
        The same tensor `x` (mutated in-place).

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If the underlying CUDA operation fails (propagated from wrappers).
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    # IMPORTANT: use a shared DLL handle
    if lib is None:
        lib = Tensor._get_cuda_lib()

    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    fill_cuda(
        lib,
        y_dev=int(x.data),
        numel=int(n),
        value=float(value),
        dtype=dt,
        sync=bool(sync),
    )
    return x


def zeros_(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    In-place zero fill for a CUDA tensor.

    This uses the `zeros_cuda` helper, which typically performs a device memset
    (byte-wise zero) for efficiency.

    Parameters
    ----------
    x : Tensor
        CUDA tensor whose buffer will be zeroed in-place.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `zeros_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        The same tensor `x` (mutated in-place).

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If the underlying CUDA operation fails (propagated from wrappers).
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64_dtype(np.dtype(x.dtype), "x.dtype")

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    zeros_cuda(
        lib,
        y_dev=int(x.data),
        numel=int(n),
        dtype=dt,
        sync=bool(sync),
    )
    return x


def ones_(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    In-place ones fill for a CUDA tensor.

    Parameters
    ----------
    x : Tensor
        CUDA tensor whose buffer will be filled with ones in-place.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `ones_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        The same tensor `x` (mutated in-place).

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If the underlying CUDA operation fails (propagated from wrappers).

    Notes
    -----
    `ones_cuda` may use the native scalar fill kernel where possible, and can
    fall back to a host->device memcpy (inside `ones_cuda`) if the native path
    is unstable.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64_dtype(np.dtype(x.dtype), "x.dtype")

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    ones_cuda(
        lib,
        y_dev=int(x.data),
        numel=int(n),
        dtype=dt,
        sync=bool(sync),
    )
    return x


def zeros_like(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Allocate a new CUDA tensor and fill it with zeros.

    Parameters
    ----------
    x : Tensor
        Reference CUDA tensor providing shape, dtype, and device placement.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `zeros_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor with the same shape/dtype as `x`, filled
        with zeros.

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If allocation or the underlying CUDA operation fails.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64_dtype(np.dtype(x.dtype), "x.dtype")

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = int(cuda_malloc(lib, int(nbytes))) if nbytes != 0 else 0

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=x.device.index,
        dev_ptr=int(y_dev),
        nbytes=nbytes,
        dtype=dt,
    )

    try:
        zeros_cuda(
            lib,
            y_dev=int(y_dev),
            numel=int(n),
            dtype=dt,
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage,
            shape=tuple(int(d) for d in x.shape),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        if y_dev:
            cuda_free(lib, int(y_dev))
        raise


def ones_like(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Allocate a new CUDA tensor and fill it with ones.

    Parameters
    ----------
    x : Tensor
        Reference CUDA tensor providing shape, dtype, and device placement.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `ones_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor with the same shape/dtype as `x`, filled
        with ones.

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If allocation or the underlying CUDA operation fails.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64_dtype(np.dtype(x.dtype), "x.dtype")

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = int(cuda_malloc(lib, int(nbytes))) if nbytes != 0 else 0

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=x.device.index,
        dev_ptr=int(y_dev),
        nbytes=nbytes,
        dtype=dt,
    )

    try:
        ones_cuda(
            lib,
            y_dev=int(y_dev),
            numel=int(n),
            dtype=dt,
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage,
            shape=tuple(int(d) for d in x.shape),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        if y_dev:
            cuda_free(lib, int(y_dev))
        raise


def full_like(x: Tensor, value: float, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Allocate a new CUDA tensor and fill it with a scalar value.

    Parameters
    ----------
    x : Tensor
        Reference CUDA tensor providing shape, dtype, and device placement.
    value : float
        Scalar value to write into all elements.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `fill_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor with the same shape/dtype as `x`, filled
        with `value`.

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If allocation or the underlying CUDA operation fails.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64_dtype(np.dtype(x.dtype), "x.dtype")

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = int(cuda_malloc(lib, int(nbytes))) if nbytes != 0 else 0

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=x.device.index,
        dev_ptr=int(y_dev),
        nbytes=nbytes,
        dtype=dt,
    )

    try:
        fill_cuda(
            lib,
            y_dev=int(y_dev),
            numel=int(n),
            value=float(value),
            dtype=dt,
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage,
            shape=tuple(int(d) for d in x.shape),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        if y_dev:
            cuda_free(lib, int(y_dev))
        raise


# Optional: simple factories if you want them for tests / Tensor.zeros/ones plumbing.
def zeros(
    shape: Iterable[int],
    *,
    dtype: np.dtype,
    device_obj,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Allocate a new CUDA tensor with the given shape and fill it with zeros.

    Parameters
    ----------
    shape : Iterable[int]
        Desired tensor shape.
    dtype : np.dtype
        Desired dtype (float32 or float64).
    device_obj : Any
        KeyDNN device object to attach to the returned Tensor. Must be CUDA and
        implement `is_cuda()`.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `zeros_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor of shape `shape` and dtype `dtype`, filled
        with zeros.

    Raises
    ------
    TypeError
        If `device_obj` is not CUDA or dtype is unsupported.
    RuntimeError
        If allocation or the underlying CUDA operation fails.
    """
    if not device_obj.is_cuda():
        raise TypeError(f"device_obj must be CUDA; got {device_obj}")
    dt = _require_f32_f64_dtype(np.dtype(dtype), "dtype")
    shp = tuple(int(d) for d in shape)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(shp)
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = int(cuda_malloc(lib, int(nbytes))) if nbytes != 0 else 0

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=int(device),
        dev_ptr=int(y_dev),
        nbytes=nbytes,
        dtype=dt,
    )

    try:
        zeros_cuda(lib, y_dev=int(y_dev), numel=int(n), dtype=dt, sync=bool(sync))
        return Tensor._from_storage(
            storage,
            shape=shp,
            dtype=dt,
            device=device_obj,
            requires_grad=False,
        )
    except Exception:
        if y_dev:
            cuda_free(lib, int(y_dev))
        raise


def ones(
    shape: Iterable[int],
    *,
    dtype: np.dtype,
    device_obj,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Allocate a new CUDA tensor with the given shape and fill it with ones.

    Parameters
    ----------
    shape : Iterable[int]
        Desired tensor shape.
    dtype : np.dtype
        Desired dtype (float32 or float64).
    device_obj : Any
        KeyDNN device object to attach to the returned Tensor. Must be CUDA and
        implement `is_cuda()`.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, requests synchronization behavior in `ones_cuda`.
        Defaults to True.

    Returns
    -------
    Tensor
        Newly allocated CUDA tensor of shape `shape` and dtype `dtype`, filled
        with ones.

    Raises
    ------
    TypeError
        If `device_obj` is not CUDA or dtype is unsupported.
    RuntimeError
        If allocation or the underlying CUDA operation fails.
    """
    if not device_obj.is_cuda():
        raise TypeError(f"device_obj must be CUDA; got {device_obj}")
    dt = _require_f32_f64_dtype(np.dtype(dtype), "dtype")
    shp = tuple(int(d) for d in shape)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device))

    n = _numel(shp)
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = int(cuda_malloc(lib, int(nbytes))) if nbytes != 0 else 0

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=int(device),
        dev_ptr=int(y_dev),
        nbytes=int(nbytes),
        dtype=dt,
    )

    try:
        ones_cuda(lib, y_dev=int(y_dev), numel=int(n), dtype=dt, sync=bool(sync))
        return Tensor._from_storage(
            storage,
            shape=shp,
            dtype=dt,
            device=device_obj,
            requires_grad=False,
        )
    except Exception:
        if y_dev:
            cuda_free(lib, int(y_dev))
        raise
