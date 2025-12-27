from __future__ import annotations

import ctypes
from ctypes import wintypes
import numpy as np

from ..avgpool2d_ctypes import load_keydnn_cuda_native

_GetProcAddress = ctypes.windll.kernel32.GetProcAddress
_GetProcAddress.argtypes = [wintypes.HMODULE, wintypes.LPCSTR]
_GetProcAddress.restype = wintypes.LPVOID


def _get_proc_addr(lib: ctypes.CDLL, sym: str) -> int:
    addr = _GetProcAddress(lib._handle, sym.encode("ascii"))
    v = int(ctypes.cast(addr, ctypes.c_void_p).value or 0)
    if v == 0:
        raise RuntimeError(f"Native DLL missing symbol: {sym}")
    return v


# int keydnn_cuda_memcpy_h2d(void* dst, const void* src, size_t nbytes)
def cuda_memcpy_h2d(lib, dst_dev: int, src_host, nbytes: int) -> None:
    addr = _get_proc_addr(lib, "keydnn_cuda_memcpy_h2d")
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    )
    fn = FN(addr)

    # src_host is a numpy array; get pointer to its data
    if not isinstance(src_host, np.ndarray):
        raise TypeError("src_host must be a numpy.ndarray")
    if int(nbytes) != int(src_host.nbytes):
        # allow exact byte copy only; caller can pass view if needed
        raise ValueError(
            f"nbytes mismatch: nbytes={nbytes}, src_host.nbytes={src_host.nbytes}"
        )

    st = int(
        fn(
            ctypes.c_void_p(int(dst_dev)),
            ctypes.c_void_p(int(src_host.ctypes.data)),
            ctypes.c_size_t(int(nbytes)),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")


# int keydnn_cuda_memcpy_d2h(void* dst, const void* src, size_t nbytes)
def cuda_memcpy_d2h(lib, dst_host, src_dev: int, nbytes: int) -> None:
    addr = _get_proc_addr(lib, "keydnn_cuda_memcpy_d2h")
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    )
    fn = FN(addr)

    if not isinstance(dst_host, np.ndarray):
        raise TypeError("dst_host must be a numpy.ndarray")
    if int(nbytes) != int(dst_host.nbytes):
        raise ValueError(
            f"nbytes mismatch: nbytes={nbytes}, dst_host.nbytes={dst_host.nbytes}"
        )

    st = int(
        fn(
            ctypes.c_void_p(int(dst_host.ctypes.data)),
            ctypes.c_void_p(int(src_dev)),
            ctypes.c_size_t(int(nbytes)),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")


# int keydnn_cuda_memcpy_d2d(void* dst, const void* src, size_t nbytes)
def cuda_memcpy_d2d(lib, dst_dev: int, src_dev: int, nbytes: int) -> None:
    addr = _get_proc_addr(lib, "keydnn_cuda_memcpy_d2d")
    FN = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.c_void_p(int(dst_dev)),
            ctypes.c_void_p(int(src_dev)),
            ctypes.c_size_t(int(nbytes)),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2d failed with status={st}")


def memcpy_dtod(
    lib, *, dst_dev: int, src_dev: int, nbytes: int, sync: bool = True
) -> None:
    cuda_memcpy_d2d(lib, dst_dev, src_dev, nbytes)


def memcpy_htod(
    lib, *, dst_dev: int, src_host: np.ndarray, nbytes: int, sync: bool = True
) -> None:
    cuda_memcpy_h2d(lib, dst_dev, src_host, nbytes)


def memcpy_dtoh(
    lib, *, dst_host: np.ndarray, src_dev: int, nbytes: int, sync: bool = True
) -> None:
    cuda_memcpy_d2h(lib, dst_host, src_dev, nbytes)


# ---------------------------------------------------------------------
# Compatibility aliases (preferred public API used by Tensor ops/tests)
# ---------------------------------------------------------------------

from ..global_avgpool2d_ctypes import cuda_synchronize  # type: ignore


def memcpy_htod(
    lib,
    *,
    dst_dev: int,
    src_host: np.ndarray,
    nbytes: int,
    sync: bool = True,
) -> None:
    """
    Host-to-device memcpy (compat alias).

    Matches the keyword-only signature used across tests and Tensor ops.
    """
    cuda_memcpy_h2d(lib, dst_dev=dst_dev, src_host=src_host, nbytes=nbytes)
    if sync:
        cuda_synchronize(lib)


def memcpy_dtoh(
    lib,
    *,
    dst_host: np.ndarray,
    src_dev: int,
    nbytes: int,
    sync: bool = True,
) -> None:
    """
    Device-to-host memcpy (compat alias).
    """
    cuda_memcpy_d2h(lib, dst_host=dst_host, src_dev=src_dev, nbytes=nbytes)
    if sync:
        cuda_synchronize(lib)


def memcpy_dtod(
    lib,
    *,
    dst_dev: int,
    src_dev: int,
    nbytes: int,
    sync: bool = True,
) -> None:
    """
    Device-to-device memcpy (compat alias).
    """
    cuda_memcpy_d2d(lib, dst_dev=dst_dev, src_dev=src_dev, nbytes=nbytes)
    if sync:
        cuda_synchronize(lib)
