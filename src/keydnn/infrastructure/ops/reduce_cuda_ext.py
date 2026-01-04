# src/keydnn/infrastructure/ops/reduce_cuda_ext.py
"""
CUDA reduce primitives with Tensor boundaries (device-pointer based).

This module is the Tensor-facing counterpart of `reduce_cuda.py`.

Key design
----------
- Operates on CUDA tensors without converting them to NumPy.
- Reads `Tensor.data` as a device pointer (DevPtr).
- Allocates output device buffers.
- Invokes ops-layer functions (e.g., `sum_all_cuda`, `sum_axis2d_forward_cuda`) which
  call the native DLL kernels via ctypes.
- Wraps output pointers into CUDA `Tensor` via `Tensor._from_devptr`.

Scope
-----
- sum_all_forward / mean_all_forward:
    Reduce a contiguous device buffer to a device scalar.
- sum_axis2d_forward:
    2D sum reduction along axis {0, 1}.
- Backward helpers:
    - sum_backward_fill_forward / mean_backward_fill_forward:
        Broadcast a device scalar into a device buffer.
    - sum_axis2d_backward:
        Broadcast a reduced gradient (1D) back to a 2D gradient buffer.

Supported dtypes
----------------
- float32 / float64 only.

Ownership
---------
The returned Tensor owns the output device pointer. If an exception occurs
before Tensor construction, the allocated device buffer is freed.
"""

from __future__ import annotations

import numpy as np

from ..tensor._tensor import Tensor
from ..tensor._cuda_storage import _CudaStorage
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free

from .reduce_cuda import (
    sum_all_cuda as _sum_all_devptr,
    mean_all_cuda as _mean_all_devptr,
    sum_backward_fill_cuda as _sum_bwd_fill_devptr,
    mean_backward_fill_cuda as _mean_bwd_fill_devptr,
    max_axis2d_forward_cuda as _max_axis2d_fwd_devptr,
    max_axis2d_backward_cuda as _max_axis2d_bwd_devptr,
    sum_axis2d_forward_cuda as _sum_axis2d_fwd_devptr,
    sum_axis2d_backward_cuda as _sum_axis2d_bwd_devptr,
    sum_to_shape_cuda as _sum_to_shape_devptr,
)


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


def _require_2d(x: Tensor, name: str) -> tuple[int, int]:
    """Validate that a tensor is 2D and return (rows, cols)."""
    if len(x.shape) != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    r, c = int(x.shape[0]), int(x.shape[1])
    if r <= 0 or c <= 0:
        raise ValueError(f"{name} must have positive dimensions, got shape={x.shape}")
    return r, c


# -----------------------------------------------------------------------------
# all-reduce (scalar)
# -----------------------------------------------------------------------------


def sum_all_forward(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Sum-reduce all elements of a CUDA tensor into a device scalar.

    Parameters
    ----------
    x : Tensor
        CUDA Tensor of any shape, dtype float32/float64.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper (which may call cuda_synchronize). Default True.

    Returns
    -------
    Tensor
        CUDA scalar Tensor (shape=()) containing the sum of all elements.
    """
    device_index: int = x.device.index
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    numel = _numel(x.shape)
    if numel <= 0:
        raise ValueError(f"sum_all_forward requires numel > 0, got {numel}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(np.dtype(dt).itemsize)  # scalar
    y_dev = cuda_malloc(lib, nbytes_y)

    from ..tensor._cuda_storage import _CudaStorage

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(y_dev),
        nbytes=int(nbytes_y),
        dtype=dt,
    )

    try:
        _sum_all_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            numel=int(numel),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_yd,
            shape=(),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        storage_yd.decref()
        raise


def mean_all_forward(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Mean-reduce all elements of a CUDA tensor into a device scalar.

    Parameters
    ----------
    x : Tensor
        CUDA Tensor of any shape, dtype float32/float64.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper (which may call cuda_synchronize). Default True.

    Returns
    -------
    Tensor
        CUDA scalar Tensor (shape=()) containing the mean of all elements.
    """
    device_index: int = x.device.index
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    numel = _numel(x.shape)
    if numel <= 0:
        raise ValueError(f"mean_all_forward requires numel > 0, got {numel}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(np.dtype(dt).itemsize)  # scalar
    y_dev = cuda_malloc(lib, nbytes_y)

    from ..tensor._cuda_storage import _CudaStorage

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(y_dev),
        nbytes=int(nbytes_y),
        dtype=dt,
    )

    try:
        _mean_all_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            numel=int(numel),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_yd,
            shape=(),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        storage_yd.decref()
        raise


# -----------------------------------------------------------------------------
# 2D sum reduction
# -----------------------------------------------------------------------------


def sum_axis2d_forward(
    x: Tensor, *, axis: int, device: int = 0, sync: bool = True
) -> Tensor:
    """
    Sum-reduce a 2D CUDA tensor along an axis.

    Parameters
    ----------
    x : Tensor
        CUDA Tensor of shape (rows, cols), dtype float32/float64.
    axis : int
        Reduction axis. Must be 0 (reduce rows -> shape (cols,)) or
        1 (reduce cols -> shape (rows,)).
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper (which may call cuda_synchronize). Default True.

    Returns
    -------
    Tensor
        CUDA Tensor containing reduced sums:
        - axis == 0: shape (cols,)
        - axis == 1: shape (rows,)
    """
    device_index: int = x.device.index
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")
    rows, cols = _require_2d(x, "x")

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    out_shape = (cols,) if axis == 0 else (rows,)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int((cols if axis == 0 else rows) * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(y_dev),
        nbytes=int(nbytes_y),
        dtype=dt,
    )

    try:
        _sum_axis2d_fwd_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            rows=int(rows),
            cols=int(cols),
            axis=int(axis),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_yd,
            shape=out_shape,
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        storage_yd.decref()
        raise


def sum_axis2d_backward(
    grad_out: Tensor,
    *,
    rows: int,
    cols: int,
    axis: int,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Backward broadcast for `sum_axis2d_forward`.

    Parameters
    ----------
    grad_out : Tensor
        CUDA Tensor of shape (rows,) if axis==1 else (cols,), dtype float32/float64.
    rows : int
        Number of rows in the original input matrix.
    cols : int
        Number of columns in the original input matrix.
    axis : int
        Reduction axis used in the forward pass (0 or 1).
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper (which may call cuda_synchronize). Default True.

    Returns
    -------
    Tensor
        CUDA Tensor of shape (rows, cols) containing broadcasted gradients.
    """
    device_index: int = grad_out.device.index
    _require_cuda(grad_out, "grad_out")
    dt = _require_f32_f64(grad_out, "grad_out")

    rows_i = int(rows)
    cols_i = int(cols)
    if rows_i <= 0 or cols_i <= 0:
        raise ValueError(
            f"rows/cols must be positive, got rows={rows_i}, cols={cols_i}"
        )

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    exp_shape = (rows_i,) if axis == 1 else (cols_i,)
    if tuple(int(d) for d in grad_out.shape) != exp_shape:
        raise ValueError(
            f"grad_out shape mismatch: expected {exp_shape} for axis={axis}, got {grad_out.shape}"
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_gx = int(rows_i * cols_i * np.dtype(dt).itemsize)
    gx_dev = cuda_malloc(lib, nbytes_gx)

    storage_gd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(gx_dev),
        nbytes=int(nbytes_gx),
        dtype=dt,
    )

    try:
        _sum_axis2d_bwd_devptr(
            lib,
            grad_out_dev=int(grad_out.data),
            grad_x_dev=int(gx_dev),
            rows=int(rows_i),
            cols=int(cols_i),
            axis=int(axis),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_gd,
            shape=(rows_i, cols_i),
            dtype=dt,
            device=grad_out.device,
            requires_grad=False,
        )
    except Exception:
        storage_gd.decref()
        raise


# -----------------------------------------------------------------------------
# backward fill helpers (scalar -> vector)
# -----------------------------------------------------------------------------


def sum_backward_fill_forward(
    grad_out_scalar: Tensor,
    *,
    numel: int,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Broadcast a CUDA scalar gradient into a CUDA vector (sum backward fill).

    Parameters
    ----------
    grad_out_scalar : Tensor
        CUDA scalar Tensor (shape=()) of dtype float32/float64.
    numel : int
        Length of the output vector.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper. Default True.

    Returns
    -------
    Tensor
        CUDA Tensor of shape (numel,) filled with grad_out_scalar.
    """
    device_index: int = grad_out_scalar.device.index
    _require_cuda(grad_out_scalar, "grad_out_scalar")
    dt = _require_f32_f64(grad_out_scalar, "grad_out_scalar")

    n = int(numel)
    if n <= 0:
        raise ValueError(f"numel must be > 0, got {n}")
    if _numel(grad_out_scalar.shape) != 1:
        raise ValueError(
            f"grad_out_scalar must be scalar (shape=()), got {grad_out_scalar.shape}"
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    gx_dev = cuda_malloc(lib, nbytes)

    storage_gd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(gx_dev),
        nbytes=int(nbytes),
        dtype=dt,
    )

    try:
        _sum_bwd_fill_devptr(
            lib,
            grad_out_dev=int(grad_out_scalar.data),
            grad_x_dev=int(gx_dev),
            numel=int(n),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_gd,
            shape=(n,),
            dtype=dt,
            device=grad_out_scalar.device,
            requires_grad=False,
        )
    except Exception:
        storage_gd.decref()
        raise


def mean_backward_fill_forward(
    grad_out_scalar: Tensor,
    *,
    numel: int,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Broadcast a CUDA scalar gradient into a CUDA vector (mean backward fill).

    Parameters
    ----------
    grad_out_scalar : Tensor
        CUDA scalar Tensor (shape=()) of dtype float32/float64.
    numel : int
        Length of the output vector (used to scale the gradient inside the kernel).
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer wrapper. Default True.

    Returns
    -------
    Tensor
        CUDA Tensor of shape (numel,) filled with grad_out_scalar / numel.
    """
    device_index: int = grad_out_scalar.device.index
    _require_cuda(grad_out_scalar, "grad_out_scalar")
    dt = _require_f32_f64(grad_out_scalar, "grad_out_scalar")

    n = int(numel)
    if n <= 0:
        raise ValueError(f"numel must be > 0, got {n}")
    if _numel(grad_out_scalar.shape) != 1:
        raise ValueError(
            f"grad_out_scalar must be scalar (shape=()), got {grad_out_scalar.shape}"
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    gx_dev = cuda_malloc(lib, nbytes)

    storage_gd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(gx_dev),
        nbytes=int(nbytes),
        dtype=dt,
    )

    try:
        _mean_bwd_fill_devptr(
            lib,
            grad_out_dev=int(grad_out_scalar.data),
            grad_x_dev=int(gx_dev),
            numel=int(n),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_gd,
            shape=(n,),
            dtype=dt,
            device=grad_out_scalar.device,
            requires_grad=False,
        )
    except Exception:
        storage_gd.decref()
        raise


# -----------------------------------------------------------------------------
# max axis2d (optional parity, since ops already exist)
# -----------------------------------------------------------------------------


def max_axis2d_forward(
    x: Tensor, *, axis: int, device: int = 0, sync: bool = True
) -> tuple[Tensor, Tensor]:
    """
    2D max reduction with argmax indices along axis {0, 1}.

    Returns (y, idx) where:
    - y is float tensor (same dtype as x)
    - idx is int64 tensor (device buffer)
    """
    # raise Exception()
    device_index: int = x.device.index
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")
    rows, cols = _require_2d(x, "x")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    out_len = cols if axis == 0 else rows
    y_shape = (out_len,)
    idx_shape = (out_len,)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    y_dev = cuda_malloc(lib, int(out_len * np.dtype(dt).itemsize))
    idx_dev = cuda_malloc(lib, int(out_len * np.dtype(np.int64).itemsize))

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(y_dev),
        nbytes=int(out_len * np.dtype(dt).itemsize),
        dtype=dt,
    )

    storage_id = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(idx_dev),
        nbytes=int(out_len * np.dtype(np.int64).itemsize),
        dtype=dt,
    )

    try:
        _max_axis2d_fwd_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            idx_dev=int(idx_dev),
            rows=int(rows),
            cols=int(cols),
            axis=int(axis),
            dtype=np.dtype(dt),
            sync=bool(sync),
        )

        y = Tensor._from_storage(
            storage_yd,
            shape=y_shape,
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
        idx = Tensor._from_storage(
            storage_id,
            shape=idx_shape,
            dtype=np.dtype(np.int64),
            device=x.device,
            requires_grad=False,
        )
        return y, idx

    except Exception:
        storage_yd.decref()
        storage_id.decref()
        raise


def max_axis2d_backward(
    grad_out: Tensor,
    idx: Tensor,
    *,
    rows: int,
    cols: int,
    axis: int,
    device: int = 0,
    zero_grad_x: bool = True,
    sync: bool = True,
) -> Tensor:
    """
    Backward scatter for `max_axis2d_forward` using stored argmax indices.

    Returns grad_x of shape (rows, cols).
    """
    device_index: int = grad_out.device.index
    _require_cuda(grad_out, "grad_out")
    _require_cuda(idx, "idx")

    dt = _require_f32_f64(grad_out, "grad_out")
    if np.dtype(idx.dtype) != np.dtype(np.int64):
        raise TypeError(f"idx must be int64, got dtype={np.dtype(idx.dtype)}")

    rows_i, cols_i = int(rows), int(cols)
    if rows_i <= 0 or cols_i <= 0:
        raise ValueError(
            f"rows/cols must be positive, got rows={rows_i}, cols={cols_i}"
        )
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    out_len = cols_i if axis == 0 else rows_i
    if tuple(int(d) for d in grad_out.shape) != (out_len,):
        raise ValueError(f"grad_out must have shape ({out_len},), got {grad_out.shape}")
    if tuple(int(d) for d in idx.shape) != (out_len,):
        raise ValueError(f"idx must have shape ({out_len},), got {idx.shape}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    gx_dev = cuda_malloc(lib, int(rows_i * cols_i * np.dtype(dt).itemsize))

    storage_gd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(gx_dev),
        nbytes=int(rows_i * cols_i * np.dtype(dt).itemsize),
        dtype=dt,
    )

    try:
        _max_axis2d_bwd_devptr(
            lib,
            grad_out_dev=int(grad_out.data),
            idx_dev=int(idx.data),
            grad_x_dev=int(gx_dev),
            rows=int(rows_i),
            cols=int(cols_i),
            axis=int(axis),
            dtype=np.dtype(dt),
            zero_grad_x=bool(zero_grad_x),
            sync=bool(sync),
        )
        return Tensor._from_storage(
            storage_gd,
            shape=(rows_i, cols_i),
            dtype=dt,
            device=grad_out.device,
            requires_grad=False,
        )
    except Exception:
        storage_gd.decref()
        raise


def sum_to_shape_forward(
    x: Tensor,
    *,
    out_shape: tuple[int, ...],
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Sum-reduce `x` to `out_shape` using "unbroadcast" semantics.

    This is used in broadcast backward: if x was broadcast from out_shape to x.shape,
    then grad w.r.t. the original tensor is sum_to_shape(grad, out_shape).

    Rules:
    - ranks must match
    - for each dim i:
        - if out_shape[i] == x.shape[i], keep
        - if out_shape[i] == 1 and x.shape[i] > 1, reduce (sum) over that axis
        - otherwise incompatible
    """
    device_index: int = x.device.index
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    in_shape = tuple(int(d) for d in x.shape)
    out_shape_t = tuple(int(d) for d in out_shape)

    # --- rank normalization -------------------------------------------------
    # sum_to_shape semantics allow rank drop by left-padding out_shape with ones.
    # Example: (2,3,4) -> (3,1) is treated as (1,3,1)
    #          (5,7)   -> (7,)  is treated as (1,7)
    in_shape = tuple(int(d) for d in x.shape)  # or however you already compute it
    out_shape_t = tuple(int(d) for d in out_shape)  # your passed target shape

    in_rank = len(in_shape)
    out_rank = len(out_shape_t)

    if out_rank > in_rank:
        raise ValueError(f"rank mismatch: in_shape={in_shape} out_shape={out_shape_t}")

    # Pad on the LEFT with ones so ranks match
    pad = in_rank - out_rank
    out_shape_padded = (1,) * pad + out_shape_t

    # Validate broadcast-compat: each dim must be equal or target dim == 1
    for sd, td in zip(in_shape, out_shape_padded):
        if td != 1 and td != sd:
            raise ValueError(
                f"shape mismatch: cannot sum from in_shape={in_shape} to out_shape={out_shape_t}"
            )
    # -----------------------------------------------------------------------

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    out_numel = _numel(out_shape_t)
    if out_numel <= 0:
        raise ValueError(
            f"out_shape must have positive numel, got out_shape={out_shape_t}"
        )

    y_dev = cuda_malloc(lib, int(out_numel * np.dtype(dt).itemsize))

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=int(y_dev),
        nbytes=int(out_numel * np.dtype(dt).itemsize),
        dtype=dt,
    )

    try:
        _sum_to_shape_devptr(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            in_shape=in_shape,
            out_shape=out_shape_padded,
            dtype=np.dtype(dt),
            zero_y=True,
            sync=bool(sync),
        )

        return Tensor._from_storage(
            storage_yd,
            shape=out_shape_t,
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        storage_yd.decref()
        raise


# Convenience aliases
sum_all = sum_all_forward
mean_all = mean_all_forward
sum_to_shape = sum_to_shape_forward


__all__ = [
    "sum_all_forward",
    "mean_all_forward",
    "sum_axis2d_forward",
    "sum_axis2d_backward",
    "sum_backward_fill_forward",
    "mean_backward_fill_forward",
    "max_axis2d_forward",
    "max_axis2d_backward",
    "sum_all",
    "mean_all",
    "sum_to_shape_forward",
    "sum_to_shape",
]
