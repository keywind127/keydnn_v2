from __future__ import annotations

import ctypes
from typing import Any, Tuple

import numpy as np


def _is_cdll(obj: object) -> bool:
    return isinstance(obj, ctypes.CDLL)


def _bind_memcpy(lib: ctypes.CDLL) -> Tuple[Any, Any, Any]:
    """
    Bind H2D / D2H / D2D memcpy entrypoints on the native DLL.

    Uses c_void_p for pointers (device + host) and c_size_t for nbytes.
    """
    if not hasattr(lib, "keydnn_cuda_memcpy_h2d"):
        raise RuntimeError("Native DLL missing symbol: keydnn_cuda_memcpy_h2d")
    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        raise RuntimeError("Native DLL missing symbol: keydnn_cuda_memcpy_d2h")
    if not hasattr(lib, "keydnn_cuda_memcpy_d2d"):
        raise RuntimeError("Native DLL missing symbol: keydnn_cuda_memcpy_d2d")

    h2d = lib.keydnn_cuda_memcpy_h2d
    d2h = lib.keydnn_cuda_memcpy_d2h
    d2d = lib.keydnn_cuda_memcpy_d2d

    # int keydnn_cuda_memcpy_h2d(void* dst, const void* src, size_t nbytes)
    h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    h2d.restype = ctypes.c_int

    # int keydnn_cuda_memcpy_d2h(void* dst, const void* src, size_t nbytes)
    d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    d2h.restype = ctypes.c_int

    # int keydnn_cuda_memcpy_d2d(void* dst, const void* src, size_t nbytes)
    d2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    d2d.restype = ctypes.c_int

    return h2d, d2h, d2d


def memcpy_htod(*args: Any, **kwargs: Any) -> None:
    """
    Host-to-Device memcpy.

    Expected by tests:
        memcpy_htod(lib, dst_dev=..., src_host=np.ndarray, nbytes=..., sync=True)

    Notes
    -----
    Tests store this as a class attribute and call via `self.memcpy_htod(...)`,
    which would bind `self` into the first positional argument. We therefore parse
    *args/**kwargs manually and locate the ctypes.CDLL.
    """
    if not args:
        raise TypeError("memcpy_htod expected at least a lib argument")

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "memcpy_htod expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]

    dst_dev = kwargs.pop("dst_dev", None)
    src_host = kwargs.pop("src_host", None)
    nbytes = kwargs.pop("nbytes", None)
    sync = bool(kwargs.pop("sync", True))

    # Accept older aliases
    if dst_dev is None:
        dst_dev = kwargs.pop("dst", None)
    if src_host is None:
        src_host = kwargs.pop("src", None)

    # Optional positional (dst_dev, src_host, nbytes)
    if dst_dev is None and len(rest) >= 1:
        dst_dev = rest[0]
    if src_host is None and len(rest) >= 2:
        src_host = rest[1]
    if nbytes is None and len(rest) >= 3:
        nbytes = rest[2]
    if len(rest) > 3:
        raise TypeError(
            "memcpy_htod accepts at most 3 positional args: dst_dev, src_host, nbytes"
        )

    if dst_dev is None or src_host is None:
        raise TypeError("memcpy_htod requires dst_dev and src_host")

    if not isinstance(src_host, np.ndarray):
        raise TypeError(f"src_host must be np.ndarray, got {type(src_host)!r}")

    if not src_host.flags["C_CONTIGUOUS"]:
        src_host = np.ascontiguousarray(src_host)

    if nbytes is None:
        nbytes_i = int(src_host.nbytes)
    else:
        nbytes_i = int(nbytes)

    if nbytes_i < 0:
        raise ValueError("nbytes must be >= 0")
    if nbytes_i != int(src_host.nbytes):
        raise ValueError(
            f"nbytes mismatch: {nbytes_i} vs src_host.nbytes={src_host.nbytes}"
        )

    h2d, _, _ = _bind_memcpy(lib)
    st = int(
        h2d(
            ctypes.c_void_p(int(dst_dev)),
            ctypes.c_void_p(int(src_host.ctypes.data)),
            ctypes.c_size_t(nbytes_i),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")

    if sync and hasattr(lib, "keydnn_cuda_synchronize"):
        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        _ = int(fn())


def memcpy_dtoh(*args: Any, **kwargs: Any) -> None:
    """
    Device-to-Host memcpy.

    Expected by tests:
        memcpy_dtoh(lib, dst_host=np.ndarray, src_dev=..., nbytes=..., sync=True)
    """
    if not args:
        raise TypeError("memcpy_dtoh expected at least a lib argument")

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "memcpy_dtoh expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]

    dst_host = kwargs.pop("dst_host", None)
    src_dev = kwargs.pop("src_dev", None)
    nbytes = kwargs.pop("nbytes", None)
    sync = bool(kwargs.pop("sync", True))

    # Accept older aliases
    if dst_host is None:
        dst_host = kwargs.pop("dst", None)
    if src_dev is None:
        src_dev = kwargs.pop("src", None)

    # Optional positional (dst_host, src_dev, nbytes)
    if dst_host is None and len(rest) >= 1:
        dst_host = rest[0]
    if src_dev is None and len(rest) >= 2:
        src_dev = rest[1]
    if nbytes is None and len(rest) >= 3:
        nbytes = rest[2]
    if len(rest) > 3:
        raise TypeError(
            "memcpy_dtoh accepts at most 3 positional args: dst_host, src_dev, nbytes"
        )

    if dst_host is None or src_dev is None:
        raise TypeError("memcpy_dtoh requires dst_host and src_dev")

    if not isinstance(dst_host, np.ndarray):
        raise TypeError(f"dst_host must be np.ndarray, got {type(dst_host)!r}")
    if not dst_host.flags["C_CONTIGUOUS"]:
        raise ValueError("dst_host must be C-contiguous")

    if nbytes is None:
        nbytes_i = int(dst_host.nbytes)
    else:
        nbytes_i = int(nbytes)

    if nbytes_i < 0:
        raise ValueError("nbytes must be >= 0")
    if nbytes_i != int(dst_host.nbytes):
        raise ValueError(
            f"nbytes mismatch: {nbytes_i} vs dst_host.nbytes={dst_host.nbytes}"
        )

    _, d2h, _ = _bind_memcpy(lib)
    st = int(
        d2h(
            ctypes.c_void_p(int(dst_host.ctypes.data)),
            ctypes.c_void_p(int(src_dev)),
            ctypes.c_size_t(nbytes_i),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")

    if sync and hasattr(lib, "keydnn_cuda_synchronize"):
        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        _ = int(fn())


def memcpy_dtod(*args: Any, **kwargs: Any) -> None:
    """
    Device-to-Device memcpy.

    Expected by tests:
        memcpy_dtod(lib, dst_dev=..., src_dev=..., nbytes=..., sync=True)
    """
    if not args:
        raise TypeError("memcpy_dtod expected at least a lib argument")

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "memcpy_dtod expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]

    dst_dev = kwargs.pop("dst_dev", None)
    src_dev = kwargs.pop("src_dev", None)
    nbytes = kwargs.pop("nbytes", None)
    sync = bool(kwargs.pop("sync", True))

    # Accept older aliases
    if dst_dev is None:
        dst_dev = kwargs.pop("dst", None)
    if src_dev is None:
        src_dev = kwargs.pop("src", None)

    # Optional positional (dst_dev, src_dev, nbytes)
    if dst_dev is None and len(rest) >= 1:
        dst_dev = rest[0]
    if src_dev is None and len(rest) >= 2:
        src_dev = rest[1]
    if nbytes is None and len(rest) >= 3:
        nbytes = rest[2]
    if len(rest) > 3:
        raise TypeError(
            "memcpy_dtod accepts at most 3 positional args: dst_dev, src_dev, nbytes"
        )

    if dst_dev is None or src_dev is None:
        raise TypeError("memcpy_dtod requires dst_dev and src_dev")
    if nbytes is None:
        raise TypeError("memcpy_dtod requires nbytes")

    nbytes_i = int(nbytes)
    if nbytes_i < 0:
        raise ValueError("nbytes must be >= 0")

    _, _, d2d = _bind_memcpy(lib)
    st = int(
        d2d(
            ctypes.c_void_p(int(dst_dev)),
            ctypes.c_void_p(int(src_dev)),
            ctypes.c_size_t(nbytes_i),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2d failed with status={st}")

    if sync and hasattr(lib, "keydnn_cuda_synchronize"):
        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        _ = int(fn())


# Aliases for resolve_func convenience
cuda_memcpy_htod = memcpy_htod
copy_htod = memcpy_htod
htod = memcpy_htod

cuda_memcpy_dtoh = memcpy_dtoh
copy_dtoh = memcpy_dtoh
dtoh = memcpy_dtoh

cuda_memcpy_dtod = memcpy_dtod
copy_dtod = memcpy_dtod
dtod = memcpy_dtod

__all__ = [
    "memcpy_htod",
    "memcpy_dtoh",
    "memcpy_dtod",
    "cuda_memcpy_htod",
    "copy_htod",
    "htod",
    "cuda_memcpy_dtoh",
    "copy_dtoh",
    "dtoh",
    "cuda_memcpy_dtod",
    "copy_dtod",
    "dtod",
]
