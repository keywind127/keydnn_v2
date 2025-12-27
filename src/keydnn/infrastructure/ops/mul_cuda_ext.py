# src/keydnn/infrastructure/ops/mul_cuda_ext.py
"""
CUDA elementwise multiply with Tensor boundaries (device-pointer based).

This module exposes a Tensor-first API for elementwise multiplication on CUDA.
It wraps the low-level ctypes kernel wrapper `native_cuda.python.ops.unary_ctypes.mul_cuda`
(despite the file name, `mul_cuda` is a binary op) and handles device allocation
and Tensor construction.

Key design
----------
- Accepts CUDA `Tensor` inputs and never converts CUDA tensors to NumPy.
- Treats `Tensor.data` as a raw device pointer (uintptr_t stored as Python int).
- Allocates the output buffer on device using shared CUDA utilities.
- Invokes the underlying ctypes wrapper to run the CUDA kernel.
- Returns a CUDA `Tensor` via `Tensor._from_devptr`, transferring ownership of
  the output device pointer to the returned Tensor.

Scope and assumptions
---------------------
- Elementwise multiply only (no broadcasting).
- Inputs must have identical shapes.
- Inputs must be on the same CUDA device (checked via equality with a string
  fallback for robustness).
- Supported dtypes: float32 / float64.

Ownership and failure behavior
------------------------------
The returned Tensor owns its output device allocation. If an exception occurs
after allocating the output buffer but before constructing the Tensor, this
module frees the allocation to avoid leaks.

Notes
-----
The `sync` parameter is accepted for API symmetry with other CUDA ops. The
current ctypes wrapper does not accept a sync flag; synchronization behavior is
therefore determined by the native implementation (or may be added later at
this boundary).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..tensor._tensor import Tensor
from ..native_cuda.python.ops.unary_ctypes import mul_cuda as _mul_ctypes

# Reuse existing DLL loader + CUDA utils
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free


def _numel(shape: Tuple[int, ...]) -> int:
    """
    Compute the number of elements for a given shape.

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
        If devices differ.

    Notes
    -----
    Uses `a.device != b.device` as the primary check and falls back to string
    comparison for compatibility with alternative DeviceLike implementations.
    """
    # Device is usually a small value object; equality should work if implemented.
    # Fallback to string compare to be robust across DeviceLike implementations.
    if a.device != b.device and str(a.device) != str(b.device):
        raise ValueError(f"device mismatch: a.device={a.device} vs b.device={b.device}")


def mul_forward(a: Tensor, b: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Elementwise multiply on CUDA: `y = a * b` (no broadcasting).

    This function validates Tensor placement and metadata, allocates an output
    device buffer, dispatches the CUDA kernel via ctypes, and returns a new CUDA
    Tensor that owns the output buffer.

    Parameters
    ----------
    a : Tensor
        CUDA tensor of any shape, dtype float32/float64.
    b : Tensor
        CUDA tensor with the same shape, dtype, and device as `a`.
    device : int, optional
        CUDA device ordinal to set before allocation and kernel launch.
        Defaults to 0.
    sync : bool, optional
        Accepted for API symmetry. The underlying ctypes wrapper does not take a
        sync flag; synchronization may be performed internally by the native
        implementation. Defaults to True.

    Returns
    -------
    Tensor
        CUDA tensor with the same shape and dtype as inputs.

    Raises
    ------
    TypeError
        If inputs are not CUDA or dtype is not float32/float64, or if dtypes
        mismatch.
    ValueError
        If shape/device mismatch or if the computed number of elements is <= 0.
    RuntimeError
        If the underlying CUDA kernel reports failure (propagated from ctypes).

    Ownership
    ---------
    The returned Tensor owns the allocated output device pointer. If an error
    occurs before Tensor construction, the allocation is freed.
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

    numel = _numel(tuple(int(d) for d in a.shape))
    if numel <= 0:
        raise ValueError(f"mul_forward requires numel > 0, got {numel}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(numel * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    try:
        _mul_ctypes(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            y_dev=int(y_dev),
            numel=int(numel),
            dtype=np.dtype(dt),
        )

        # `sync` is accepted for symmetry with other APIs; if you later add a
        # cuda_synchronize(lib) here, you can gate it on `sync`.
        _ = bool(sync)

        return Tensor._from_devptr(
            int(y_dev),
            shape=tuple(int(d) for d in a.shape),
            dtype=dt,
            device=a.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


# Convenience aliases
mul = mul_forward
cuda_mul = mul_forward

__all__ = ["mul_forward", "mul", "cuda_mul"]
