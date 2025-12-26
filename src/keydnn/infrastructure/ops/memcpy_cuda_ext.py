# src/keydnn/infrastructure/ops/memcpy_cuda_ext.py
"""
CUDA memcpy primitives with Tensor boundaries (Tensor-in / Tensor-out).

This module provides Tensor-facing helpers that wrap the ops-layer memcpy
functions (host<->device and device<->device) defined in `memcpy_cuda.py`.

Key design
----------
- Uses `Tensor.data` as a raw device pointer (DevPtr) for CUDA tensors.
- Allocates destination device buffers where appropriate.
- Delegates the actual copy to ops-layer entrypoints:
    - memcpy_htod: host -> device
    - memcpy_dtoh: device -> host
    - memcpy_dtod: device -> device

Scope
-----
- `copy_host_to_cuda`:  np.ndarray -> CUDA Tensor
- `copy_cuda_to_host`:  CUDA Tensor -> np.ndarray
- `copy_cuda_to_cuda`:  CUDA Tensor -> CUDA Tensor (new buffer)

Notes
-----
- Host arrays are treated as C-contiguous buffers. Non-contiguous inputs are
  copied into a contiguous temporary.
- This module is intentionally small and explicit; it is a Tensor-boundary
  utility and should not attempt broadcasting or shape inference beyond
  preserving the original tensor's shape.
"""

from __future__ import annotations

import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from .memcpy_cuda import memcpy_htod as _memcpy_htod
from .memcpy_cuda import memcpy_dtoh as _memcpy_dtoh
from .memcpy_cuda import memcpy_dtod as _memcpy_dtod


def _require_cuda(x: Tensor, name: str) -> None:
    """Validate that a tensor is on CUDA."""
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def copy_host_to_cuda(
    x_host: np.ndarray,
    *,
    device_tensor: "object",
    device: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Copy a host NumPy array to CUDA and return a CUDA Tensor.

    Parameters
    ----------
    x_host : np.ndarray
        Host array to copy. Will be converted to C-contiguous if needed.
    device_tensor : object
        A CUDA `Device` instance used for constructing the returned Tensor.
        (We keep this parameter generic to avoid importing Device here and
        risking circular imports; pass `Tensor.device` or `Device.cuda(0)`.)
    device : int, optional
        CUDA device ordinal. Default 0.
    sync : bool, optional
        Forwarded to ops-layer memcpy. Default True.

    Returns
    -------
    Tensor
        CUDA Tensor with the same shape/dtype as `x_host`.

    Raises
    ------
    TypeError
        If `x_host` is not an ndarray.
    ValueError
        If `x_host` has zero bytes (empty arrays are supported, but produce
        a Tensor with devptr=0 in some implementations; this wrapper chooses
        to allocate 0 bytes and return devptr=0 safely).
    """
    if not isinstance(x_host, np.ndarray):
        raise TypeError(f"x_host must be np.ndarray, got {type(x_host)!r}")

    x_c = np.ascontiguousarray(x_host)
    nbytes = int(x_c.nbytes)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    if nbytes == 0:
        # Represent empty tensor without allocating.
        dev_ptr = 0
    else:
        dev_ptr = int(cuda_malloc(lib, nbytes))

    try:
        if nbytes != 0:
            _memcpy_htod(
                lib,
                dst_dev=int(dev_ptr),
                src_host=x_c,
                nbytes=nbytes,
                sync=bool(sync),
            )

        return Tensor._from_devptr(
            int(dev_ptr),
            shape=tuple(int(d) for d in x_c.shape),
            dtype=x_c.dtype,
            device=device_tensor,  # type: ignore[arg-type]
            requires_grad=False,
        )
    except Exception:
        if nbytes != 0 and dev_ptr:
            cuda_free(lib, dev_ptr)
        raise


def copy_cuda_to_host(
    x: Tensor,
    *,
    out: np.ndarray | None = None,
    device: int = 0,
    sync: bool = True,
) -> np.ndarray:
    """
    Copy a CUDA Tensor to host.

    Parameters
    ----------
    x : Tensor
        CUDA Tensor to copy from.
    out : np.ndarray | None, optional
        Optional destination array. Must be C-contiguous and match dtype/shape.
        If None, a new array is allocated.
    device : int, optional
        CUDA device ordinal. Default 0.
    sync : bool, optional
        Forwarded to ops-layer memcpy. Default True.

    Returns
    -------
    np.ndarray
        Host array containing the copied data.

    Raises
    ------
    TypeError
        If `x` is not CUDA, or `out` is not ndarray when provided.
    ValueError
        If `out` does not match shape/dtype or is not C-contiguous.
    """
    _require_cuda(x, "x")

    dt = np.dtype(x.dtype)
    shape = tuple(int(d) for d in x.shape)

    if out is None:
        out_arr = np.empty(shape, dtype=dt, order="C")
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError(f"out must be np.ndarray, got {type(out)!r}")
        if out.shape != shape:
            raise ValueError(f"out.shape mismatch: expected {shape}, got {out.shape}")
        if np.dtype(out.dtype) != dt:
            raise ValueError(f"out.dtype mismatch: expected {dt}, got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        out_arr = out

    nbytes = int(out_arr.nbytes)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    if nbytes == 0:
        return out_arr

    _memcpy_dtoh(
        lib,
        dst_host=out_arr,
        src_dev=int(x.data),
        nbytes=nbytes,
        sync=bool(sync),
    )
    return out_arr


def copy_cuda_to_cuda(x: Tensor, *, device: int = 0, sync: bool = True) -> Tensor:
    """
    Copy a CUDA Tensor to a new CUDA Tensor (device-to-device).

    Parameters
    ----------
    x : Tensor
        Source CUDA Tensor.
    device : int, optional
        CUDA device ordinal. Default 0.
    sync : bool, optional
        Forwarded to ops-layer memcpy. Default True.

    Returns
    -------
    Tensor
        New CUDA Tensor with the same shape/dtype as `x`.

    Notes
    -----
    - This allocates a fresh destination buffer on device and performs D2D copy.
    - Returned tensor is on the same logical `x.device` as the source.
    """
    _require_cuda(x, "x")

    dt = np.dtype(x.dtype)
    shape = tuple(int(d) for d in x.shape)
    nbytes = int(np.prod(shape, dtype=np.int64) * dt.itemsize)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    if nbytes == 0:
        dev_ptr = 0
    else:
        dev_ptr = int(cuda_malloc(lib, nbytes))

    try:
        if nbytes != 0:
            _memcpy_dtod(
                lib,
                dst_dev=int(dev_ptr),
                src_dev=int(x.data),
                nbytes=int(nbytes),
                sync=bool(sync),
            )

        return Tensor._from_devptr(
            int(dev_ptr),
            shape=shape,
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        if nbytes != 0 and dev_ptr:
            cuda_free(lib, dev_ptr)
        raise


__all__ = ["copy_host_to_cuda", "copy_cuda_to_host", "copy_cuda_to_cuda"]
