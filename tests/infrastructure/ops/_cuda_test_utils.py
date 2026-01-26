from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class CudaEnv:
    lib: object
    cuda_malloc: Callable
    cuda_free: Callable
    cudaMemcpyHtoD: Callable
    cudaMemcpyDtoH: Callable
    cudaMemcpyDtoD: Callable
    cuda_synchronize: Callable
    cuda_set_device: Optional[Callable] = None


def try_get_cuda_env() -> Optional[CudaEnv]:
    """
    Best-effort loader for the KeyDNN CUDA native library + basic runtime APIs.
    Returns None if CUDA DLL or wrappers are not available (tests should skip).
    """
    try:

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m
    except Exception:
        return None

    try:
        lib = m.load_keydnn_cuda_native()

    except Exception:
        return None

    import ctypes
    from ctypes import wintypes

    GetProcAddress = ctypes.windll.kernel32.GetProcAddress
    GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
    GetProcAddress.restype = wintypes.LPVOID

    cuda_set_device = getattr(m, "cuda_set_device", None)

    required = [
        "cuda_malloc",
        "cuda_free",
        "cudaMemcpyHtoD",
        "cudaMemcpyDtoH",
        "cudaMemcpyDtoD",
        "cuda_synchronize",
    ]
    for name in required:
        if not hasattr(m, name):

            return None

    return CudaEnv(
        lib=lib,
        cuda_malloc=m.cuda_malloc,
        cuda_free=m.cuda_free,
        cudaMemcpyHtoD=m.cudaMemcpyHtoD,
        cudaMemcpyDtoH=m.cudaMemcpyDtoH,
        cudaMemcpyDtoD=m.cudaMemcpyDtoD,
        cuda_synchronize=m.cuda_synchronize,
        cuda_set_device=cuda_set_device if callable(cuda_set_device) else None,
    )


def resolve_func(mod, candidates: list[str]):
    """
    Resolve a function from a module using a list of candidate names.
    Raises AttributeError with a helpful message if none exist.
    """
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"None of the candidate functions exist in module {mod.__name__}: {candidates}"
    )


def assert_allclose_by_dtype(
    actual: np.ndarray, ref: np.ndarray, dtype: np.dtype, *, op: str
) -> None:
    """
    Centralized tolerances for CUDA vs NumPy comparisons.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        if op == "matmul":
            np.testing.assert_allclose(actual, ref, rtol=1e-5, atol=1e-6)
        elif op == "exp":
            np.testing.assert_allclose(actual, ref, rtol=2e-6, atol=3e-7)
        elif op == "transpose":

            np.testing.assert_allclose(actual, ref, rtol=0.0, atol=0.0)
        else:
            np.testing.assert_allclose(actual, ref, rtol=1e-5, atol=1e-6)
        return

    if dtype == np.float64:
        if op == "matmul":
            np.testing.assert_allclose(actual, ref, rtol=1e-12, atol=1e-12)
        elif op == "exp":
            np.testing.assert_allclose(actual, ref, rtol=1e-14, atol=1e-14)
        elif op == "transpose":
            np.testing.assert_allclose(actual, ref, rtol=0.0, atol=0.0)
        else:
            np.testing.assert_allclose(actual, ref, rtol=1e-12, atol=1e-12)
        return

    raise TypeError(f"Unsupported dtype for CUDA ops tests: {dtype}")
