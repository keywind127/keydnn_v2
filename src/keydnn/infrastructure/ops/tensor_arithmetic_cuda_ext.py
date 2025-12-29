# infrastructure/ops/tensor_arithmetic_cuda_ext.py
"""
CUDA tensor arithmetic primitives with Tensor boundaries (device-pointer based).

This module provides a Tensor-first CUDA API for a small set of elementwise
operations by wrapping low-level ctypes bindings:

- inputs are KeyDNN `Tensor` objects placed on CUDA
- outputs are newly-allocated CUDA tensors created from raw device pointers via
  `Tensor._from_devptr`
- no implicit host transfers (i.e., no `to_numpy()` for CUDA tensors)

Supported ops
-------------
Unary:
- neg(x) -> Tensor

Binary (no broadcasting):
- add(a, b) -> Tensor
- sub(a, b) -> Tensor
- div(a, b) -> Tensor

Compare:
- gt(a, b) -> Tensor
  Produces float32 outputs, even if inputs are float64.

Assumptions and constraints
---------------------------
- Inputs are CUDA tensors backed by contiguous device buffers.
- Binary ops require identical shapes (broadcasting is intentionally not
  implemented here).
- Supported input dtypes: float32 / float64.
- `gt` returns float32 by design (mirrors the native kernel ABI).

Resource management
-------------------
Each op allocates an output device buffer using the project's CUDA malloc/free
utilities. On failure, the allocated output buffer is freed before re-raising.

Notes
-----
This module intentionally stays lightweight and "mechanical": it performs only
the minimal validations needed for correctness at the Tensor boundary, then
dispatches to the underlying ctypes wrappers.
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
    # NEW: scalar variants
    add_scalar_cuda as _add_scalar_cuda,
    sub_scalar_cuda as _sub_scalar_cuda,
    div_scalar_cuda as _div_scalar_cuda,
)

# Reuse your existing DLL loader + CUDA utils (malloc/free/set_device).
from ..native_cuda.python.avgpool2d_ctypes import load_keydnn_cuda_native
from ..native_cuda.python.maxpool2d_ctypes import (
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)


def _get_lib():
    """
    Return the process-wide cached CUDA library handle used by `Tensor`.

    Returns
    -------
    ctypes.CDLL
        The loaded KeyDNN CUDA native library handle.

    Notes
    -----
    This function intentionally delegates to `Tensor._get_cuda_lib()` so all CUDA
    ops share the same loaded library/runtime context within the process.
    """
    # use Tensor's cached handle so we share the same runtime / context
    return Tensor._get_cuda_lib()


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


def _require_cuda(x: Tensor, name: str) -> None:
    """
    Require a tensor to be placed on CUDA.

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


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """
    Require a tensor dtype to be float32 or float64.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Name used in error messages.

    Returns
    -------
    np.dtype
        The normalized NumPy dtype (np.float32 or np.float64).

    Raises
    ------
    TypeError
        If dtype is not float32/float64.
    """
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_same_shape(a: Tensor, b: Tensor) -> None:
    """
    Require two tensors to have exactly the same shape.

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
    Require two tensors to be on the same device.

    Parameters
    ----------
    a, b : Tensor
        Tensors to compare.

    Raises
    ------
    ValueError
        If device placements differ.

    Notes
    -----
    Device objects are typically value objects; equality should work when
    implemented. A string fallback is used for robustness across DeviceLike
    implementations.
    """
    # Device is usually a small value object; equality should work if implemented.
    # Fallback to string compare to be robust across DeviceLike implementations.
    if a.device != b.device and str(a.device) != str(b.device):
        raise ValueError(f"device mismatch: a.device={a.device} vs b.device={b.device}")


def neg(x: Tensor, *, device: int = 0) -> Tensor:
    """
    Elementwise negation on CUDA: `y = -x`.

    Parameters
    ----------
    x : Tensor
        CUDA tensor with dtype float32/float64.
    device : int, optional
        CUDA device ordinal used for `cuda_set_device`. Defaults to 0.

    Returns
    -------
    Tensor
        CUDA tensor with the same shape and dtype as `x`.

    Raises
    ------
    TypeError
        If `x` is not CUDA or has unsupported dtype.
    RuntimeError
        If the underlying CUDA kernel returns an error (propagated from ctypes).
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(x.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

    try:
        _neg_cuda(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt,
        )
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
    Elementwise addition on CUDA: `y = a + b` (no broadcasting).

    Parameters
    ----------
    a, b : Tensor
        CUDA tensors with identical shape, dtype (float32/float64), and device.
    device : int, optional
        CUDA device ordinal used for `cuda_set_device`. Defaults to 0.

    Returns
    -------
    Tensor
        CUDA tensor with the same shape and dtype as inputs.

    Raises
    ------
    TypeError
        If inputs are not CUDA, have unsupported dtype, or dtypes mismatch.
    ValueError
        If shapes differ or devices differ.
    RuntimeError
        If the underlying CUDA kernel returns an error (propagated from ctypes).
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

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
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
    Elementwise subtraction on CUDA: `y = a - b` (no broadcasting).

    Parameters
    ----------
    a, b : Tensor
        CUDA tensors with identical shape, dtype (float32/float64), and device.
    device : int, optional
        CUDA device ordinal used for `cuda_set_device`. Defaults to 0.

    Returns
    -------
    Tensor
        CUDA tensor with the same shape and dtype as inputs.

    Raises
    ------
    TypeError
        If inputs are not CUDA, have unsupported dtype, or dtypes mismatch.
    ValueError
        If shapes differ or devices differ.
    RuntimeError
        If the underlying CUDA kernel returns an error (propagated from ctypes).
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

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
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
    Elementwise division on CUDA: `y = a / b` (no broadcasting).

    Parameters
    ----------
    a, b : Tensor
        CUDA tensors with identical shape, dtype (float32/float64), and device.
    device : int, optional
        CUDA device ordinal used for `cuda_set_device`. Defaults to 0.

    Returns
    -------
    Tensor
        CUDA tensor with the same shape and dtype as inputs.

    Raises
    ------
    TypeError
        If inputs are not CUDA, have unsupported dtype, or dtypes mismatch.
    ValueError
        If shapes differ or devices differ.
    RuntimeError
        If the underlying CUDA kernel returns an error (propagated from ctypes).
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

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
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
    Elementwise greater-than on CUDA: `y = float32(a > b)`.

    Parameters
    ----------
    a, b : Tensor
        CUDA tensors with identical shape, dtype (float32/float64), and device.
    device : int, optional
        CUDA device ordinal used for `cuda_set_device`. Defaults to 0.

    Returns
    -------
    Tensor
        CUDA tensor of dtype float32 with the same shape as inputs.

    Raises
    ------
    TypeError
        If inputs are not CUDA, have unsupported dtype, or dtypes mismatch.
    ValueError
        If shapes differ or devices differ.
    RuntimeError
        If the underlying CUDA kernel returns an error (propagated from ctypes).

    Notes
    -----
    The output dtype is always float32 by design of the native kernel ABI.
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

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
    out_dt = np.float32
    nbytes_y = int(n * np.dtype(out_dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    try:
        _gt_cuda(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            n=int(n),
            dtype=dt_in,  # selects gt_f32 vs gt_f64
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
    Elementwise scalar add on CUDA: `y = a + alpha`.

    Notes
    -----
    - No broadcasting: scalar is a true scalar value (not a Tensor).
    - This avoids materializing a full scalar-filled Tensor on device.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

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
    Elementwise scalar sub on CUDA: `y = a - alpha`.

    Notes
    -----
    - No broadcasting: scalar is a true scalar value (not a Tensor).
    - This avoids materializing a full scalar-filled Tensor on device.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

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
    Elementwise scalar div on CUDA: `y = a / alpha`.

    Notes
    -----
    - No broadcasting: scalar is a true scalar value (not a Tensor).
    - This avoids materializing a full scalar-filled Tensor on device.
    """
    _require_cuda(a, "a")
    dt = _require_f32_f64(a, "a")

    lib = _get_lib()
    cuda_set_device(lib, int(device))

    n = _numel(tuple(a.shape))
    nbytes = int(n * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes)

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


# ---- convenience aliases (match the tensor-tensor style) ----
cuda_add_scalar = add_scalar
cuda_sub_scalar = sub_scalar
cuda_div_scalar = div_scalar

__all__ = [
    "neg",
    "add",
    "sub",
    "div",
    "gt",
    "add_scalar",
    "sub_scalar",
    "div_scalar",
    "cuda_add_scalar",
    "cuda_sub_scalar",
    "cuda_div_scalar",
]
