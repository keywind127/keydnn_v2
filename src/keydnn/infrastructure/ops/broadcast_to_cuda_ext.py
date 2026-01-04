# src/keydnn/infrastructure/ops/broadcast_to_cuda_ext.py
"""
CUDA broadcast_to with Tensor boundaries (device-pointer based).

This module exposes a Tensor-first API for `broadcast_to` on CUDA.

It wraps the low-level ctypes kernel wrappers in
`native_cuda.python.ops.broadcast_ctypes` and handles device allocation and
Tensor construction.

Key design
----------
- Accepts CUDA `Tensor` input and never converts CUDA tensors to NumPy.
- Treats `Tensor.data` as a raw device pointer (uintptr_t stored as Python int).
- Allocates output buffers on device using shared CUDA utilities.
- Invokes the underlying ctypes wrapper to run the CUDA kernel.
- Returns a CUDA `Tensor` via `Tensor._from_devptr`, transferring ownership of
  the output device pointer to the returned Tensor.

Scope and assumptions
---------------------
- Output is always materialized (not a view).
- Input and output are treated as contiguous row-major arrays.
- Supported dtypes: float32 / float64.

Failure behavior and ownership
------------------------------
- The returned Tensor owns the output device allocation.
- If an exception occurs after allocation but before Tensor construction, the
  output allocation is freed to avoid leaks.

Notes
-----
The `sync` parameter is accepted for API symmetry. The ctypes wrapper does not
accept a sync flag; synchronization behavior is determined by the native side
(or may be added later at this boundary).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..tensor._tensor import Tensor

from ..native_cuda.python.ops.broadcast_ctypes import (
    broadcast_to_cuda as _broadcast_to_ctypes,
)

# Reuse existing DLL loader + CUDA utils
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free


def _numel(shape: Tuple[int, ...]) -> int:
    """Product of dims (1 for empty shape)."""
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _require_cuda(x: Tensor, name: str) -> None:
    """Require that a tensor is on CUDA."""
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """Require dtype float32/float64 and return normalized dtype."""
    dt = np.dtype(x.dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _require_same_device(a: Tensor, b_device) -> None:
    """
    Require tensor `a` to be on the same device as `b_device`.

    `b_device` can be a Device-like object.
    """
    if a.device != b_device and str(a.device) != str(b_device):
        raise ValueError(f"device mismatch: a.device={a.device} vs target={b_device}")


def _is_broadcastable(in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> bool:
    """
    Check NumPy-style broadcast compatibility (conservative).
    """
    if len(in_shape) > len(out_shape):
        return False
    pad = len(out_shape) - len(in_shape)
    padded = (1,) * pad + tuple(int(d) for d in in_shape)

    for sd, td in zip(padded, out_shape):
        sd = int(sd)
        td = int(td)
        if sd == td:
            continue
        if sd == 1 and td >= 0:
            continue
        # allow td==0 edge cases (empty outputs) as long as sd in {0,1}
        if td == 0 and sd in (0, 1):
            continue
        return False
    return True


def broadcast_to_forward(
    x: Tensor,
    shape: Tuple[int, ...],
    *,
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Broadcast a CUDA tensor to `shape` by materializing an expanded copy on device.

    Parameters
    ----------
    x : Tensor
        CUDA tensor (contiguous assumed), dtype float32/float64.
    shape : tuple[int, ...]
        Target broadcast shape.
    device : int, optional
        CUDA device ordinal to set before allocation and kernel launch. Defaults to 0.
    sync : bool, optional
        Accepted for API symmetry. Defaults to True.

    Returns
    -------
    Tensor
        CUDA tensor of shape `shape`, dtype same as input.

    Raises
    ------
    TypeError
        If `x` is not CUDA or dtype unsupported.
    ValueError
        If `shape` is not broadcast-compatible with `x.shape`.
    RuntimeError
        If underlying CUDA kernel reports failure (propagated from ctypes).
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    out_shape = tuple(int(d) for d in shape)
    in_shape = tuple(int(d) for d in x.shape)

    if not _is_broadcastable(in_shape, out_shape):
        raise ValueError(f"Cannot broadcast shape {in_shape} to {out_shape}")

    out_numel = _numel(out_shape)
    if out_numel < 0:
        raise ValueError(f"broadcast_to_forward requires numel >= 0, got {out_numel}")

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # If output is empty, we still want to return a valid Tensor.
    # Allocate at least 1 byte to keep cuda_malloc happy; Tensor will treat shape as empty anyway.
    nbytes_y = int(max(out_numel, 1) * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=x.device.index,
        dev_ptr=y_dev,
        nbytes=nbytes_y,
        dtype=dt,
    )

    try:
        # If out_numel==0, calling the kernel should be a no-op anyway,
        # but it's safe to skip to avoid touching pointers.
        if out_numel != 0:
            _broadcast_to_ctypes(
                lib,
                x_dev=int(x.data),
                y_dev=int(y_dev),
                in_shape=in_shape,
                out_shape=out_shape,
                dtype=np.dtype(dt),
            )

        _ = bool(sync)

        return Tensor._from_storage(
            storage,
            shape=out_shape,
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    
    except Exception:
        cuda_free(lib, y_dev)
        raise


# Convenience aliases
broadcast_to = broadcast_to_forward
cuda_broadcast_to = broadcast_to_forward

__all__ = [
    "broadcast_to_forward",
    "broadcast_to",
    "cuda_broadcast_to",
]
