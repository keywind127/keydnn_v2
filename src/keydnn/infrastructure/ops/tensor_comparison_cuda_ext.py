# infrastructure/ops/tensor_comparison_cuda_ext.py
"""
CUDA tensor comparison primitives with Tensor boundaries (device-pointer based).

This module provides a Tensor-first CUDA API for a set of elementwise comparison
operations by wrapping low-level ctypes CUDA bindings in:

  infrastructure/native_cuda/python/ops/tensor_comparison_ctypes.py

Key characteristics
-------------------
- Inputs are KeyDNN `Tensor` objects that must already reside on CUDA.
- Outputs are newly allocated CUDA tensors constructed directly from raw device
  pointers via `Tensor._from_devptr`.
- No implicit host/device transfers are performed.
- Broadcasting semantics are intentionally *not* supported.
- Validation is minimal and focused strictly on Tensor boundary correctness.

Supported operations
--------------------
Elementwise (no broadcasting; requires identical shape/device/dtype):
- gt(a, b) -> Tensor
- ge(a, b) -> Tensor
- lt(a, b) -> Tensor
- le(a, b) -> Tensor
- eq(a, b) -> Tensor
- ne(a, b) -> Tensor

Scalar (no broadcasting):
- gt_scalar(a, alpha) -> Tensor
- ge_scalar(a, alpha) -> Tensor
- lt_scalar(a, alpha) -> Tensor
- le_scalar(a, alpha) -> Tensor
- eq_scalar(a, alpha) -> Tensor
- ne_scalar(a, alpha) -> Tensor

Dtype conventions
-----------------
- Supported input dtypes: float32, float64
- All comparison ops produce float32 outputs (mask) regardless of input dtype.

Resource management
-------------------
Each out-of-place operation allocates an output buffer using KeyDNN's CUDA
malloc/free utilities. If a kernel invocation fails, the allocated buffer
is freed before re-raising the exception.

This module is intentionally low-level and mechanical; higher-level semantics
such as autograd integration, broadcasting, or dtype promotion are handled
elsewhere.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..tensor._tensor import Tensor
from ..tensor._cuda_storage import _CudaStorage
from ..native_cuda.python.ops.tensor_comparison_ctypes import (
    # elementwise
    gt_cuda as _gt_cuda,
    ge_cuda as _ge_cuda,
    lt_cuda as _lt_cuda,
    le_cuda as _le_cuda,
    eq_cuda as _eq_cuda,
    ne_cuda as _ne_cuda,
    # scalar
    gt_scalar_cuda as _gt_scalar_cuda,
    ge_scalar_cuda as _ge_scalar_cuda,
    lt_scalar_cuda as _lt_scalar_cuda,
    le_scalar_cuda as _le_scalar_cuda,
    eq_scalar_cuda as _eq_scalar_cuda,
    ne_scalar_cuda as _ne_scalar_cuda,
)

# Reuse your existing DLL loader + CUDA utils (malloc/free/set_device).
from ..native_cuda.python.maxpool2d_ctypes import (
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)


def _get_lib():
    """
    Return the cached CUDA shared library handle used by `Tensor`.

    This function centralizes access to the process-wide CUDA DLL loaded
    by the Tensor infrastructure, ensuring all ops dispatch against the
    same native library instance.
    """
    return Tensor._get_cuda_lib()


def _numel(shape: Tuple[int, ...]) -> int:
    """
    Compute the total number of elements for a given tensor shape.
    """
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _empty_cuda_tensor_like(x: Tensor, *, dtype: np.dtype) -> Tensor:
    """
    Construct an empty CUDA Tensor with the same shape/device as `x`, but with
    the provided dtype. Uses devptr=0 and performs no allocation.
    """
    return Tensor._from_devptr(
        0,
        shape=tuple(x.shape),
        dtype=np.dtype(dtype),
        device=x.device,
        requires_grad=False,
    )


def _require_cuda(x: Tensor, name: str) -> None:
    """
    Ensure that a tensor resides on a CUDA device.
    """
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """
    Ensure that a tensor has dtype float32 or float64.
    """
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_same_shape(a: Tensor, b: Tensor) -> None:
    """
    Ensure two tensors have exactly the same shape.
    """
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"shape mismatch: a.shape={a.shape} vs b.shape={b.shape}")


def _require_same_device(a: Tensor, b: Tensor) -> None:
    """
    Ensure two tensors are placed on the same device.
    """
    if a.device != b.device and str(a.device) != str(b.device):
        raise ValueError(f"device mismatch: a.device={a.device} vs b.device={b.device}")


def _require_same_dtype(a: Tensor, b: Tensor) -> np.dtype:
    """
    Ensure two tensors share the same supported dtype (float32/float64).
    """
    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")
    return dt_a


def _alloc_mask_out_like(x: Tensor, *, n: int, device: int) -> int:
    """
    Allocate a float32 mask output buffer on CUDA.

    Returns
    -------
    int
        Device pointer to float32 buffer of length `n`.
    """
    lib = _get_lib()
    cuda_set_device(lib, int(device))
    return int(cuda_malloc(lib, int(n * np.dtype(np.float32).itemsize)))


def _cmp_binary_out_of_place(
    a: Tensor,
    b: Tensor,
    *,
    device: int,
    op: Callable[..., None],
) -> Tensor:
    """
    Shared implementation for elementwise (binary) comparisons producing float32 masks.
    """
    device_index: int = a.device.index
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)
    dt_in = _require_same_dtype(a, b)

    n = _numel(tuple(a.shape))
    out_dt = np.float32
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=out_dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    y_dev = int(cuda_malloc(lib, int(n * np.dtype(out_dt).itemsize)))

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=y_dev,
        nbytes=int(n * np.dtype(out_dt).itemsize),
        dtype=out_dt,
    )

    try:
        op(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt_in,
        )
        # return Tensor._from_devptr(
        #     int(y_dev),
        #     shape=tuple(a.shape),
        #     dtype=out_dt,
        #     device=a.device,
        #     requires_grad=False,
        # )
        return Tensor._from_storage(
            storage_yd,
            shape=tuple(a.shape),
            dtype=out_dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        # cuda_free(lib, y_dev)
        storage_yd.decref()
        raise


def _cmp_scalar_out_of_place(
    a: Tensor,
    alpha: float,
    *,
    device: int,
    op: Callable[..., None],
) -> Tensor:
    """
    Shared implementation for scalar comparisons producing float32 masks.
    """
    device_index: int = a.device.index
    _require_cuda(a, "a")
    dt_in = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    out_dt = np.float32
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=out_dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    y_dev = int(cuda_malloc(lib, int(n * np.dtype(out_dt).itemsize)))

    storage_yd = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=y_dev,
        nbytes=int(n * np.dtype(out_dt).itemsize),
        dtype=out_dt,
    )

    try:
        op(
            lib,
            a_dev=int(a.data),
            alpha=float(alpha),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt_in,
        )
        # return Tensor._from_devptr(
        #     int(y_dev),
        #     shape=tuple(a.shape),
        #     dtype=out_dt,
        #     device=a.device,
        #     requires_grad=False,
        # )
        return Tensor._from_storage(
            storage_yd,
            shape=tuple(a.shape),
            dtype=out_dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        # cuda_free(lib, y_dev)
        storage_yd.decref()
        raise


# ============================
# Elementwise comparisons
# ============================


def gt(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise greater-than: y = (a > b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_gt_cuda)


def ge(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise greater-or-equal: y = (a >= b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_ge_cuda)


def lt(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise less-than: y = (a < b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_lt_cuda)


def le(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise less-or-equal: y = (a <= b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_le_cuda)


def eq(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise equality: y = (a == b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_eq_cuda)


def ne(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """Elementwise not-equal: y = (a != b) as float32 mask."""
    return _cmp_binary_out_of_place(a, b, device=device, op=_ne_cuda)


# ============================
# Scalar comparisons
# ============================


def gt_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar greater-than: y = (a > alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_gt_scalar_cuda)


def ge_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar greater-or-equal: y = (a >= alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_ge_scalar_cuda)


def lt_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar less-than: y = (a < alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_lt_scalar_cuda)


def le_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar less-or-equal: y = (a <= alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_le_scalar_cuda)


def eq_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar equality: y = (a == alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_eq_scalar_cuda)


def ne_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """Scalar not-equal: y = (a != alpha) as float32 mask."""
    return _cmp_scalar_out_of_place(a, alpha, device=device, op=_ne_scalar_cuda)


# ---- convenience aliases (optional, mirrors arithmetic ext style) ----
cuda_gt = gt
cuda_ge = ge
cuda_lt = lt
cuda_le = le
cuda_eq = eq
cuda_ne = ne

cuda_gt_scalar = gt_scalar
cuda_ge_scalar = ge_scalar
cuda_lt_scalar = lt_scalar
cuda_le_scalar = le_scalar
cuda_eq_scalar = eq_scalar
cuda_ne_scalar = ne_scalar


__all__ = [
    # elementwise
    "gt",
    "ge",
    "lt",
    "le",
    "eq",
    "ne",
    # scalar
    "gt_scalar",
    "ge_scalar",
    "lt_scalar",
    "le_scalar",
    "eq_scalar",
    "ne_scalar",
    # aliases
    "cuda_gt",
    "cuda_ge",
    "cuda_lt",
    "cuda_le",
    "cuda_eq",
    "cuda_ne",
    "cuda_gt_scalar",
    "cuda_ge_scalar",
    "cuda_lt_scalar",
    "cuda_le_scalar",
    "cuda_eq_scalar",
    "cuda_ne_scalar",
]
