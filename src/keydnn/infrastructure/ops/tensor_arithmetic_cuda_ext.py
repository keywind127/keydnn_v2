# infrastructure/ops/tensor_arithmetic_cuda_ext.py
"""
CUDA tensor arithmetic primitives with Tensor boundaries (device-pointer based).

This module provides a Tensor-first CUDA API for a small set of elementwise
arithmetic and comparison operations by wrapping low-level ctypes CUDA bindings.

Key characteristics
-------------------
- Inputs are KeyDNN `Tensor` objects that must already reside on CUDA.
- Outputs (for out-of-place ops) are newly allocated CUDA tensors constructed
  directly from raw device pointers via `Tensor._from_devptr`.
- No implicit host/device transfers are performed.
- Broadcasting semantics are intentionally *not* supported.
- Validation is minimal and focused strictly on Tensor boundary correctness.

Supported operations
--------------------
Unary:
- neg(x) -> Tensor

Binary (no broadcasting):
- add(a, b) -> Tensor
- sub(a, b) -> Tensor
- div(a, b) -> Tensor

Comparison:
- gt(a, b) -> Tensor
  Produces float32 outputs regardless of input dtype (matches kernel ABI).

Scalar (no broadcasting):
- add_scalar(a, alpha) -> Tensor
- sub_scalar(a, alpha) -> Tensor
- div_scalar(a, alpha) -> Tensor

In-place (mutate `a`):
- add_inplace(a, b) -> None
- sub_inplace(a, b) -> None
- div_inplace(a, b) -> None
- add_scalar_inplace(a, alpha) -> None
- sub_scalar_inplace(a, alpha) -> None
- div_scalar_inplace(a, alpha) -> None

Assumptions
-----------
- All tensors are CUDA tensors with contiguous device buffers.
- Binary ops require identical shapes and devices.
- Supported input dtypes are float32 and float64 only.

Resource management
-------------------
Each out-of-place operation allocates a device buffer using KeyDNN's CUDA
malloc/free utilities. If a kernel invocation fails, the allocated buffer
is freed before re-raising the exception.

This module is intentionally low-level and mechanical; higher-level semantics
such as autograd integration, broadcasting, or dtype promotion are handled
elsewhere.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..tensor._tensor import Tensor
from ..native_cuda.python.ops.tensor_arithmetic_ctypes import (
    neg_cuda as _neg_cuda,
    add_cuda as _add_cuda,
    sub_cuda as _sub_cuda,
    div_cuda as _div_cuda,
    gt_cuda as _gt_cuda,
    # scalar out-of-place
    add_scalar_cuda as _add_scalar_cuda,
    sub_scalar_cuda as _sub_scalar_cuda,
    div_scalar_cuda as _div_scalar_cuda,
    # ----------------------------
    # NEW: in-place variants
    # ----------------------------
    add_inplace_cuda as _add_inplace_cuda,
    sub_inplace_cuda as _sub_inplace_cuda,
    div_inplace_cuda as _div_inplace_cuda,
    add_scalar_inplace_cuda as _add_scalar_inplace_cuda,
    sub_scalar_inplace_cuda as _sub_scalar_inplace_cuda,
    div_scalar_inplace_cuda as _div_scalar_inplace_cuda,
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

    Parameters
    ----------
    shape : Tuple[int, ...]
        Tensor shape.

    Returns
    -------
    int
        Product of all dimensions in `shape`.
    """
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _empty_cuda_tensor_like(x: Tensor, *, dtype: np.dtype) -> Tensor:
    """
    Construct an empty CUDA Tensor with the same shape/device as `x`, but with
    the provided dtype. Uses devptr=0 and performs no allocation.

    Notes
    -----
    Some CUDA allocators and wrappers treat cudaMalloc(0) as an error. For
    numel==0 tensors, out-of-place ops should return an empty CUDA tensor
    without calling cuda_malloc(0).
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

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Human-readable tensor name for error messages.

    Raises
    ------
    TypeError
        If the tensor is not a CUDA tensor.
    """
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """
    Ensure that a tensor has dtype float32 or float64.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Human-readable tensor name for error messages.

    Returns
    -------
    np.dtype
        Normalized NumPy dtype of the tensor.

    Raises
    ------
    TypeError
        If the tensor dtype is not supported.
    """
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_same_shape(a: Tensor, b: Tensor) -> None:
    """
    Ensure two tensors have exactly the same shape.

    Parameters
    ----------
    a, b : Tensor
        Tensors to compare.

    Raises
    ------
    ValueError
        If shapes differ.
    """
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"shape mismatch: a.shape={a.shape} vs b.shape={b.shape}")


def _require_same_device(a: Tensor, b: Tensor) -> None:
    """
    Ensure two tensors are placed on the same device.

    Parameters
    ----------
    a, b : Tensor
        Tensors to compare.

    Raises
    ------
    ValueError
        If devices differ.
    """
    if a.device != b.device and str(a.device) != str(b.device):
        raise ValueError(f"device mismatch: a.device={a.device} vs b.device={b.device}")


# ============================
# Out-of-place ops
# ============================
def neg(x: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise negation on CUDA.

    Computes `y = -x`.

    Parameters
    ----------
    x : Tensor
        Input CUDA tensor.
    device : int, optional
        CUDA device index.

    Returns
    -------
    Tensor
        New CUDA tensor containing the result.
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    n = _numel(tuple(x.shape))
    if n == 0:
        return _empty_cuda_tensor_like(x, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

    try:
        _neg_cuda(lib, x_dev=int(x.data), y_dev=int(y_dev), n=int(n), dtype=dt)
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(x.shape),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def add(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise addition on CUDA.

    Computes `y = a + b`.

    Parameters
    ----------
    a, b : Tensor
        Input CUDA tensors with identical shape, dtype, and device.
    device : int, optional
        CUDA device index.

    Returns
    -------
    Tensor
        New CUDA tensor containing the result.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)

    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")
    dt = dt_a

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

    try:
        _add_cuda(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def sub(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise subtraction on CUDA.

    Computes `y = a - b`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)

    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")
    dt = dt_a

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

    try:
        _sub_cuda(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def div(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise division on CUDA.

    Computes `y = a / b`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)

    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")
    dt = dt_a

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

    try:
        _div_cuda(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def gt(a: Tensor, b: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise greater-than comparison on CUDA.

    Computes `y = (a > b)`.

    Notes
    -----
    The output dtype is always float32, regardless of input dtype.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)

    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")
    dt_in = dt_a

    n = _numel(tuple(a.shape))
    out_dt = np.float32
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=out_dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    y_dev = cuda_malloc(lib, int(n * np.dtype(out_dt).itemsize))

    try:
        _gt_cuda(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt_in,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=out_dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def add_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """
    Elementwise scalar addition on CUDA.

    Computes `y = a + alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    y_dev = cuda_malloc(lib, int(n * np.dtype(dt).itemsize))

    try:
        _add_scalar_cuda(
            lib,
            a_dev=int(a.data),
            alpha=float(alpha),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def sub_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """
    Elementwise scalar subtraction on CUDA.

    Computes `y = a - alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    y_dev = cuda_malloc(lib, int(n * np.dtype(dt).itemsize))

    try:
        _sub_scalar_cuda(
            lib,
            a_dev=int(a.data),
            alpha=float(alpha),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def div_scalar(a: Tensor, alpha: float, *, device: int = 0) -> Tensor:
    """
    Elementwise scalar division on CUDA.

    Computes `y = a / alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n == 0:
        return _empty_cuda_tensor_like(a, dtype=dt)

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    y_dev = cuda_malloc(lib, int(n * np.dtype(dt).itemsize))

    try:
        _div_scalar_cuda(
            lib,
            a_dev=int(a.data),
            alpha=float(alpha),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


# ============================
# In-place ops
# ============================
def add_inplace(a: Tensor, b: Tensor, *, device: int = 0) -> None:
    """
    In-place elementwise addition on CUDA.

    Mutates `a` as `a += b`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)
    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _add_inplace_cuda(lib, a_dev=int(a.data), b_dev=int(b.data), n=int(n), dtype=dt_a)


def sub_inplace(a: Tensor, b: Tensor, *, device: int = 0) -> None:
    """
    In-place elementwise subtraction on CUDA.

    Mutates `a` as `a -= b`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)
    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _sub_inplace_cuda(lib, a_dev=int(a.data), b_dev=int(b.data), n=int(n), dtype=dt_a)


def div_inplace(a: Tensor, b: Tensor, *, device: int = 0) -> None:
    """
    In-place elementwise division on CUDA.

    Mutates `a` as `a /= b`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")
    _require_same_device(a, b)
    _require_same_shape(a, b)
    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(f"dtype mismatch: a.dtype={dt_a} vs b.dtype={dt_b}")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _div_inplace_cuda(lib, a_dev=int(a.data), b_dev=int(b.data), n=int(n), dtype=dt_a)


def add_scalar_inplace(a: Tensor, alpha: float, *, device: int = 0) -> None:
    """
    In-place elementwise scalar addition on CUDA.

    Mutates `a` as `a += alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _add_scalar_inplace_cuda(
        lib, a_dev=int(a.data), alpha=float(alpha), n=int(n), dtype=dt
    )


def sub_scalar_inplace(a: Tensor, alpha: float, *, device: int = 0) -> None:
    """
    In-place elementwise scalar subtraction on CUDA.

    Mutates `a` as `a -= alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _sub_scalar_inplace_cuda(
        lib, a_dev=int(a.data), alpha=float(alpha), n=int(n), dtype=dt
    )


def div_scalar_inplace(a: Tensor, alpha: float, *, device: int = 0) -> None:
    """
    In-place elementwise scalar division on CUDA.

    Mutates `a` as `a /= alpha`.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    n = _numel(tuple(a.shape))
    if n <= 0:
        return

    lib = _get_lib()
    cuda_set_device(lib, int(device))
    _div_scalar_inplace_cuda(
        lib, a_dev=int(a.data), alpha=float(alpha), n=int(n), dtype=dt
    )


# ---- convenience aliases ----
cuda_add_scalar = add_scalar
cuda_sub_scalar = sub_scalar
cuda_div_scalar = div_scalar

__all__ = [
    # out-of-place
    "neg",
    "add",
    "sub",
    "div",
    "gt",
    "add_scalar",
    "sub_scalar",
    "div_scalar",
    # in-place
    "add_inplace",
    "sub_inplace",
    "div_inplace",
    "add_scalar_inplace",
    "sub_scalar_inplace",
    "div_scalar_inplace",
    # aliases
    "cuda_add_scalar",
    "cuda_sub_scalar",
    "cuda_div_scalar",
]
