# infrastructure/ops/matmul_cuda_ext.py
"""
CUDA MatMul primitive with Tensor boundaries (device-pointer based).

This module provides a Tensor-facing wrapper around the device-pointer-first
CUDA matmul ops implementation in `matmul_cuda.py`.

Unlike CPU implementations, this wrapper:
- Never calls `to_numpy()` for CUDA tensors.
- Treats `Tensor.data` as a device pointer (DevPtr).
- Allocates the output buffer on device.
- Invokes the CUDA kernel via the ops-layer function `matmul_cuda(...)`.
- Returns a CUDA Tensor via `Tensor._from_devptr`.

Scope
-----
- This wrapper targets 2D GEMM-style matmul:
    A: (M, K) @ B: (K, N) -> C: (M, N)
- Dtypes supported: float32 / float64
- No broadcasting / batching in this wrapper (keep it explicit and strict).

Notes on ownership
------------------
The output device pointer is *owned by the returned Tensor* (via `_from_devptr`).
If an exception occurs before Tensor construction, we free the allocated buffer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from .matmul_cuda import matmul_cuda as _matmul_devptr


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


def _require_2d(x: Tensor, name: str) -> Tuple[int, int]:
    """Validate that a tensor is 2D and return (rows, cols)."""
    if len(x.shape) != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    r, c = int(x.shape[0]), int(x.shape[1])
    if r <= 0 or c <= 0:
        raise ValueError(f"{name} must have positive dimensions, got shape={x.shape}")
    return r, c


def matmul2d_forward(
    a: Tensor, b: Tensor, *, device: int = 0, sync: bool = True
) -> Tensor:
    """
    2D matrix multiplication on CUDA: C = A @ B.

    Parameters
    ----------
    a : Tensor
        CUDA Tensor of shape (M, K), dtype float32/float64.
    b : Tensor
        CUDA Tensor of shape (K, N), dtype float32/float64.
    device : int, optional
        CUDA device ordinal to set before allocation and launch. Default is 0.
    sync : bool, optional
        Forwarded to ops-layer matmul (accepted for compatibility). Default True.

    Returns
    -------
    Tensor
        CUDA Tensor of shape (M, N), same dtype as `a` (and must match `b`).

    Raises
    ------
    TypeError
        If tensors are not CUDA or dtypes are not float32/float64 or dtype mismatch.
    ValueError
        If inputs are not 2D or shapes are incompatible.

    Notes
    -----
    - This function allocates output device memory and returns it wrapped as Tensor.
    - No implicit casting: `a.dtype` must equal `b.dtype`.
    """
    _require_cuda(a, "a")
    _require_cuda(b, "b")

    dt_a = _require_f32_f64(a, "a")
    dt_b = _require_f32_f64(b, "b")
    if dt_a != dt_b:
        raise TypeError(
            f"matmul2d_forward requires matching dtypes, got a={dt_a}, b={dt_b}"
        )

    M, K_a = _require_2d(a, "a")
    K_b, N = _require_2d(b, "b")
    if K_a != K_b:
        raise ValueError(
            f"matmul2d_forward shape mismatch: a is (M,K)=({M},{K_a}), b is (K,N)=({K_b},{N})"
        )

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # Allocate output C on device
    nbytes_c = int(M * N * np.dtype(dt_a).itemsize)
    c_dev = cuda_malloc(lib, nbytes_c)

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=a.device.index,
        dev_ptr=int(c_dev),
        nbytes=nbytes_c,
        dtype=dt_a,
    )

    try:
        # Call ops-layer kernel (device-pointer first)
        _matmul_devptr(
            lib,
            a_dev=int(a.data),
            b_dev=int(b.data),
            c_dev=int(c_dev),
            n=M,  # tests naming: n,k,m
            k=K_a,
            m=N,
            dtype=dt_a,
            sync=bool(sync),
        )

        return Tensor._from_storage(
            storage,
            shape=(M, N),
            dtype=dt_a,
            device=a.device,
            requires_grad=False,
        )

    except Exception:
        # Only free if we failed before handing ownership to Tensor
        cuda_free(lib, c_dev)
        raise


# Convenience aliases (mirrors the style of your matmul_cuda module)
matmul2d = matmul2d_forward
gemm = matmul2d_forward

__all__ = ["matmul2d_forward", "matmul2d", "gemm"]
