from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ..native_cuda.python.ops.matmul_ctypes import matmul_cuda as _matmul_ctypes


def _is_cdll(obj: object) -> bool:
    return isinstance(obj, ctypes.CDLL)


def _dtype_itemsize(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"matmul_cuda supports float32/float64 only, got {dt}")
    return int(dt.itemsize)


def _probe_dev_range(lib: ctypes.CDLL, base_dev: int, nbytes_required: int) -> None:
    """
    Best-effort guard: verify [base_dev, base_dev+nbytes_required) looks readable by probing the last byte.

    If the address is out of range / invalid, your native memcpy_d2h should fail (non-zero status),
    which we surface as RuntimeError. This prevents silent OOB when caller passes inconsistent dims.
    """
    if nbytes_required <= 0:
        return

    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        # If the symbol doesn't exist, we can't probe. Skip.
        return

    fn = lib.keydnn_cuda_memcpy_d2h
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    fn.restype = ctypes.c_int

    # copy exactly 1 byte from the last required byte
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
    Ops-layer CUDA matmul.

    Expected by tests:
        matmul_cuda(lib, a_dev=..., b_dev=..., c_dev=..., n=..., k=..., m=..., dtype=..., sync=True)

    Notes
    -----
    Unit tests store this function as a class attribute and call it via `self.matmul2d(...)`,
    which would normally bind `self` as the first positional argument. We therefore parse
    *args/**kwargs manually and locate the ctypes.CDLL.
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
    n = kwargs.pop("n", None)  # rows of A
    k = kwargs.pop("k", None)  # inner dim
    m = kwargs.pop("m", None)  # cols of B

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

    # Defensive buffer-size probes to catch shape mismatches (e.g., B is actually smaller than K*N).
    itemsize = _dtype_itemsize(dtype)
    a_req = M_i * K_i * itemsize
    b_req = K_i * N_i * itemsize
    c_req = M_i * N_i * itemsize

    # Optional device selection (important for probe on Windows)
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

    # Optional sync (kept compatible; your DtoH copy usually synchronizes anyway)
    if _sync and hasattr(lib, "keydnn_cuda_synchronize"):
        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        _ = int(fn())


# Aliases for resolve_func convenience
matmul2d_cuda = matmul_cuda
gemm_cuda = matmul_cuda
matmul2d = matmul_cuda

__all__ = ["matmul_cuda", "matmul2d_cuda", "gemm_cuda", "matmul2d"]
