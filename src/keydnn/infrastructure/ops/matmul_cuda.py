"""
CUDA matmul op wrapper with defensive device-pointer validation.

This module provides an ops-layer wrapper around the native CUDA GEMM entrypoint
exposed via `matmul_ctypes`. It exists to:

- normalize call signatures across the project (tests, ops, and higher-level Tensor APIs),
- tolerate unittest "bound method" calling convention quirks, and
- add best-effort safety checks that catch shape/dtype mismatches early by probing
  device memory ranges prior to kernel launch.

The wrapper is intentionally lightweight: it does not implement GEMM itself, and
it does not allocate output buffers. Callers are expected to allocate `c_dev`
with enough space for `M*N*dtype.itemsize` bytes.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ..native_cuda.python.ops.matmul_ctypes import matmul_cuda as _matmul_ctypes


def _is_cdll(obj: object) -> bool:
    """
    Return True if `obj` is a ctypes.CDLL instance.

    This helper supports parsing arguments in cases where unittest stores a
    function as a class attribute and calls it via `self.matmul2d(...)`, which
    injects `self` as the first positional argument.
    """
    return isinstance(obj, ctypes.CDLL)


def _dtype_itemsize(dtype: np.dtype) -> int:
    """
    Return the itemsize in bytes for a supported dtype.

    Parameters
    ----------
    dtype : np.dtype
        The dtype used for device buffers.

    Returns
    -------
    int
        The element size in bytes.

    Raises
    ------
    TypeError
        If `dtype` is not float32 or float64.
    """
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"matmul_cuda supports float32/float64 only, got {dt}")
    return int(dt.itemsize)


def _probe_dev_range(lib: ctypes.CDLL, base_dev: int, nbytes_required: int) -> None:
    """
    Best-effort guard: verify a device memory range looks readable.

    This probes exactly 1 byte at the last required address:
        [base_dev, base_dev + nbytes_required)

    If the address is out-of-range or invalid, native `memcpy_d2h` should fail
    (non-zero status), which is surfaced as a RuntimeError. This prevents silent
    out-of-bounds reads/writes when a caller passes inconsistent dimensions.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native CUDA library handle exposing `keydnn_cuda_memcpy_d2h`.
    base_dev : int
        Base device pointer (address) for the allocation being validated.
    nbytes_required : int
        Number of bytes that must be accessible from `base_dev`.

    Raises
    ------
    RuntimeError
        If the probe indicates the required last byte is not readable.
    """
    if nbytes_required <= 0:
        return

    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        # If the symbol doesn't exist, we can't probe. Skip.
        return

    fn = lib.keydnn_cuda_memcpy_d2h
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    fn.restype = ctypes.c_int

    tmp = (ctypes.c_ubyte * 1)()
    last_addr = int(base_dev) + int(nbytes_required) - 1

    st = int(
        fn(
            ctypes.cast(tmp, ctypes.c_void_p),
            ctypes.c_uint64(last_addr),
            ctypes.c_size_t(1),
        )
    )
    if st != 0:
        raise RuntimeError(
            f"device buffer too small or invalid: probe failed at 0x{last_addr:x} (status={st})"
        )


def matmul_cuda(*args: Any, **kwargs: Any) -> None:
    """
    Ops-layer CUDA matmul wrapper.

    Expected by tests (keyword-driven):
        matmul_cuda(
            lib,
            a_dev=..., b_dev=..., c_dev=...,
            n=..., k=..., m=...,
            dtype=..., sync=True,
            device_index=0
        )

    Dimension conventions
    ---------------------
    Tests commonly use (n, k, m):
        A: (n, k)
        B: (k, m)
        C: (n, m)

    The native entrypoint uses (M, N, K):
        A: (M, K)
        B: (K, N)
        C: (M, N)

    This wrapper normalizes both.

    Notes
    -----
    - Unit tests may store this function as a class attribute and call it via
      `self.matmul2d(...)`, which binds `self` as the first positional argument.
      We therefore parse *args/**kwargs manually and locate the ctypes.CDLL.
    - This wrapper does not allocate output buffers. Callers must ensure `c_dev`
      points to a valid device allocation of at least `M*N*dtype.itemsize` bytes.
    - Optional device selection is applied (if supported by the native library)
      before probing, to ensure the probe runs on the correct device context.

    Raises
    ------
    TypeError
        If required pointers/dims/dtype are missing or malformed.
    ValueError
        If computed dimensions are invalid (non-positive).
    RuntimeError
        If device selection fails or a device pointer probe fails.
    """
    if not args:
        raise TypeError("matmul_cuda expected at least a lib argument")

    # Handle bound-method injection from unittest
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "matmul_cuda expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]

    # Pull expected keyword args (tests use keywords)
    a_dev = kwargs.pop("a_dev", None)
    b_dev = kwargs.pop("b_dev", None)
    c_dev = kwargs.pop("c_dev", None)

    # tests use n, k, m
    n = kwargs.pop("n", None)  # rows of A (M)
    k = kwargs.pop("k", None)  # inner dim (K)
    m = kwargs.pop("m", None)  # cols of B (N)

    # allow native-style names too (don’t pass extras downstream)
    M = kwargs.pop("M", None)
    N = kwargs.pop("N", None)
    K = kwargs.pop("K", None)

    dtype = kwargs.pop("dtype", None)
    _sync = kwargs.pop("sync", True)  # accepted for compatibility; optional

    # tolerate alias keyword spellings
    if a_dev is None:
        a_dev = (
            kwargs.pop("A_dev", None) or kwargs.pop("A", None) or kwargs.pop("a", None)
        )
    if b_dev is None:
        b_dev = (
            kwargs.pop("B_dev", None) or kwargs.pop("B", None) or kwargs.pop("b", None)
        )
    if c_dev is None:
        c_dev = (
            kwargs.pop("C_dev", None) or kwargs.pop("C", None) or kwargs.pop("c", None)
        )

    # Allow optional positional (a_dev, b_dev, c_dev)
    if a_dev is None and len(rest) >= 1:
        a_dev = rest[0]
    if b_dev is None and len(rest) >= 2:
        b_dev = rest[1]
    if c_dev is None and len(rest) >= 3:
        c_dev = rest[2]
    if len(rest) > 3:
        raise TypeError(
            "matmul_cuda accepts at most 3 positional args: a_dev, b_dev, c_dev"
        )

    if a_dev is None or b_dev is None or c_dev is None:
        raise TypeError("matmul_cuda requires a_dev, b_dev, c_dev device pointers")
    if dtype is None:
        raise TypeError("matmul_cuda requires dtype")

    # Map dims:
    # tests: n,k,m  (A: n x k) @ (k x m) => (n x m)
    # native: M,N,K with A: (M,K), B: (K,N), C: (M,N)
    Mv = M if M is not None else n
    Kv = K if K is not None else k
    Nv = N if N is not None else m

    if Mv is None or Nv is None or Kv is None:
        raise TypeError("matmul_cuda requires dims via (n,k,m) or (M,N,K)/(M,K,N)")

    M_i = int(Mv)
    N_i = int(Nv)
    K_i = int(Kv)
    if M_i <= 0 or N_i <= 0 or K_i <= 0:
        raise ValueError(f"matmul_cuda invalid dims: M={M_i}, N={N_i}, K={K_i}")

    itemsize = _dtype_itemsize(dtype)
    a_req = int(M_i) * int(K_i) * int(itemsize)
    b_req = int(K_i) * int(N_i) * int(itemsize)
    c_req = int(M_i) * int(N_i) * int(itemsize)

    device_index = kwargs.pop("device_index", None)
    if device_index is None:
        device_index = kwargs.pop("device", None)  # tolerate alias

    if hasattr(lib, "keydnn_cuda_set_device"):
        fn = lib.keydnn_cuda_set_device
        fn.argtypes = [ctypes.c_int]
        fn.restype = ctypes.c_int
        st = int(fn(int(device_index or 0)))
        if st != 0:
            raise RuntimeError(
                f"cuda_set_device({int(device_index or 0)}) failed: status={st}"
            )

    _probe_dev_range(lib, int(a_dev), a_req)
    _probe_dev_range(lib, int(b_dev), b_req)
    _probe_dev_range(lib, int(c_dev), c_req)

    _matmul_ctypes(
        lib,
        a_dev=int(a_dev),
        b_dev=int(b_dev),
        c_dev=int(c_dev),
        M=M_i,
        N=N_i,
        K=K_i,
        dtype=np.dtype(dtype),
    )

    # Optional sync (kept compatible; DtoH copy usually synchronizes anyway)
    if _sync and hasattr(lib, "keydnn_cuda_synchronize"):
        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        _ = int(fn())


# Aliases for resolve_func convenience
matmul2d_cuda = matmul_cuda
gemm_cuda = matmul_cuda
matmul2d = matmul_cuda

__all__ = [
    "matmul_cuda",
    "matmul2d_cuda",
    "gemm_cuda",
    "matmul2d",
]
