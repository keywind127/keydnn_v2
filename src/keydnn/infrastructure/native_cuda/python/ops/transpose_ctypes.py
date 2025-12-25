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


def transpose2d_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    rows: int,
    cols: int,
    dtype: np.dtype,
) -> None:
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sym = "keydnn_cuda_transpose2d_f32"
        arg_t = ctypes.c_float
    elif dtype == np.float64:
        sym = "keydnn_cuda_transpose2d_f64"
        arg_t = ctypes.c_double
    else:
        raise TypeError(f"transpose2d_cuda supports float32/float64 only, got {dtype}")

    addr = _get_proc_addr(lib, sym)

    FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(arg_t),
        ctypes.POINTER(arg_t),
        ctypes.c_int,
        ctypes.c_int,
    )
    fn = FN(addr)

    st = int(
        fn(
            ctypes.cast(ctypes.c_void_p(int(x_dev)), ctypes.POINTER(arg_t)),
            ctypes.cast(ctypes.c_void_p(int(y_dev)), ctypes.POINTER(arg_t)),
            ctypes.c_int(int(rows)),
            ctypes.c_int(int(cols)),
        )
    )
    if st != 0:
        raise RuntimeError(f"{sym} failed with status={st}")
