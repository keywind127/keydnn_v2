"""
ctypes bindings for KeyDNN v2 CUDA GlobalAvgPool2D kernels.

This module provides low-level Python bindings to the CUDA implementation of
GlobalAvgPool2D forward/backward via `ctypes`.

Assumptions
-----------
- The CUDA DLL exports C ABI functions:
    - keydnn_cuda_set_device
    - keydnn_cuda_malloc / keydnn_cuda_free
    - keydnn_cuda_memcpy_h2d / keydnn_cuda_memcpy_d2h
    - keydnn_cuda_memset (optional)
    - keydnn_cuda_synchronize
    - keydnn_cuda_global_avgpool2d_forward_f32 / _f64
    - keydnn_cuda_global_avgpool2d_backward_f32 / _f64

- Device pointers are represented as uintptr_t handles (Python int).
- Tensors are contiguous NCHW.
- Forward maps (N,C,H,W) -> (N,C,1,1).
- Backward maps grad_out (N,C,1,1) -> grad_x (N,C,H,W), distributing evenly.

Design notes
------------
- Strict dtype/contiguity validation to avoid undefined behavior across the
  Python ↔ native boundary.
- No implicit caching: caller owns device pointers and frees them.
- Intended for infrastructure-layer use; higher-level ops should wrap this.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    c_double,
    c_float,
    c_int,
    c_size_t,
    c_uint64,
    c_void_p,
)
from pathlib import Path
from typing import Tuple

import numpy as np

from .maxpool2d_ctypes import (
    cudaMemcpyDtoH,
    cudaMemcpyHtoD,
    cuda_memcpy_dtoh,
    cuda_memcpy_htod,
)  # do not remove, dynamic import dependencies

# Device pointer handle type: uintptr_t stored as Python int
DevPtr = int

_DEFAULT_DLL_PATH = Path(
    r"D:\keydnn_v2\src\keydnn\infrastructure\native_cuda\keydnn_v2_cuda_native\x64\Debug\KeyDNNV2CudaNative.dll"
)


def load_keydnn_cuda_native(dll_path: str | Path = _DEFAULT_DLL_PATH) -> ctypes.CDLL:
    """
    Load the KeyDNN v2 CUDA native DLL.

    Parameters
    ----------
    dll_path : str | Path
        Path to the CUDA native DLL (KeyDNNV2CudaNative.dll).

    Returns
    -------
    ctypes.CDLL
        Loaded library handle.

    Raises
    ------
    FileNotFoundError
        If the DLL file does not exist.
    OSError
        If the DLL fails to load (wrong arch, missing deps, etc.).
    """
    p = Path(dll_path)
    if not p.exists():
        raise FileNotFoundError(f"CUDA native DLL not found: {p}")
    return ctypes.CDLL(str(p))


def _bind_cuda_utils(lib: ctypes.CDLL) -> None:
    """
    Bind argtypes/restype for CUDA utility exports.

    Notes
    -----
    This function is idempotent and may be called multiple times.
    """
    lib.keydnn_cuda_set_device.argtypes = [c_int]
    lib.keydnn_cuda_set_device.restype = c_int

    lib.keydnn_cuda_malloc.argtypes = [POINTER(c_uint64), c_size_t]
    lib.keydnn_cuda_malloc.restype = c_int

    lib.keydnn_cuda_free.argtypes = [c_uint64]
    lib.keydnn_cuda_free.restype = c_int

    lib.keydnn_cuda_memcpy_h2d.argtypes = [c_uint64, c_void_p, c_size_t]
    lib.keydnn_cuda_memcpy_h2d.restype = c_int

    lib.keydnn_cuda_memcpy_d2h.argtypes = [c_void_p, c_uint64, c_size_t]
    lib.keydnn_cuda_memcpy_d2h.restype = c_int

    lib.keydnn_cuda_memset.argtypes = [c_uint64, c_int, c_size_t]
    lib.keydnn_cuda_memset.restype = c_int

    lib.keydnn_cuda_synchronize.argtypes = []
    lib.keydnn_cuda_synchronize.restype = c_int


def cuda_set_device(lib: ctypes.CDLL, device: int = 0) -> None:
    """
    Set current CUDA device.

    Raises
    ------
    RuntimeError
        If the native call returns non-zero.
    """
    _bind_cuda_utils(lib)
    st = lib.keydnn_cuda_set_device(int(device))
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_set_device failed with status={st}")


def cuda_malloc(lib: ctypes.CDLL, nbytes: int) -> DevPtr:
    """
    Allocate a device buffer.

    Parameters
    ----------
    nbytes : int
        Number of bytes to allocate.

    Returns
    -------
    DevPtr
        Device pointer handle (uintptr_t) as Python int.

    Raises
    ------
    RuntimeError
        If allocation fails.
    """
    _bind_cuda_utils(lib)
    out = c_uint64(0)
    st = lib.keydnn_cuda_malloc(ctypes.byref(out), c_size_t(int(nbytes)))
    if st != 0 or out.value == 0:
        raise RuntimeError(
            f"keydnn_cuda_malloc failed with status={st}, nbytes={nbytes}"
        )
    return int(out.value)


def cuda_free(lib: ctypes.CDLL, dev_ptr: DevPtr) -> None:
    """
    Free a device buffer.

    Notes
    -----
    Safe to call with 0.
    """
    _bind_cuda_utils(lib)
    st = lib.keydnn_cuda_free(c_uint64(int(dev_ptr)))
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_free failed with status={st}")


def cuda_memcpy_h2d(lib: ctypes.CDLL, dst_dev: DevPtr, src_host: np.ndarray) -> None:
    """
    Copy a contiguous NumPy array from host to device.
    """
    _bind_cuda_utils(lib)
    if not src_host.flags["C_CONTIGUOUS"]:
        src_host = np.ascontiguousarray(src_host)
    st = lib.keydnn_cuda_memcpy_h2d(
        c_uint64(int(dst_dev)),
        c_void_p(int(src_host.ctypes.data)),
        c_size_t(int(src_host.nbytes)),
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")


def cuda_memcpy_d2h(lib: ctypes.CDLL, dst_host: np.ndarray, src_dev: DevPtr) -> None:
    """
    Copy into a contiguous NumPy array from device to host.
    """
    _bind_cuda_utils(lib)
    if not dst_host.flags["C_CONTIGUOUS"]:
        raise ValueError("dst_host must be C-contiguous")
    st = lib.keydnn_cuda_memcpy_d2h(
        c_void_p(int(dst_host.ctypes.data)),
        c_uint64(int(src_dev)),
        c_size_t(int(dst_host.nbytes)),
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")


def cuda_memset(lib: ctypes.CDLL, dev_ptr: DevPtr, value: int, nbytes: int) -> None:
    """
    Memset a device buffer to a byte value.
    """
    _bind_cuda_utils(lib)
    st = lib.keydnn_cuda_memset(
        c_uint64(int(dev_ptr)), c_int(int(value)), c_size_t(int(nbytes))
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memset failed with status={st}")


def cuda_synchronize(lib: ctypes.CDLL) -> None:
    """
    Synchronize device.
    """
    _bind_cuda_utils(lib)
    st = lib.keydnn_cuda_synchronize()
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_synchronize failed with status={st}")


def cuda_from_host(lib: ctypes.CDLL, x: np.ndarray) -> DevPtr:
    """
    Convenience: allocate device buffer and copy a NumPy CPU array to GPU.

    Parameters
    ----------
    x : np.ndarray
        Host array. Must be float32 or float64.

    Returns
    -------
    DevPtr
        Device pointer handle.
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"cuda_from_host only supports float32/float64, got {x.dtype}")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    dev = cuda_malloc(lib, x.nbytes)
    try:
        cuda_memcpy_h2d(lib, dev, x)
    except Exception:
        cuda_free(lib, dev)
        raise
    return dev


def _bind_global_avgpool2d(lib: ctypes.CDLL) -> None:
    """
    Bind argtypes/restype for GlobalAvgPool2D CUDA exports.
    """
    lib.keydnn_cuda_global_avgpool2d_forward_f32.argtypes = [
        POINTER(c_float),  # x (device)
        POINTER(c_float),  # y (device)
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H, W
    ]
    lib.keydnn_cuda_global_avgpool2d_forward_f32.restype = c_int

    lib.keydnn_cuda_global_avgpool2d_forward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_global_avgpool2d_forward_f64.restype = c_int

    lib.keydnn_cuda_global_avgpool2d_backward_f32.argtypes = [
        POINTER(c_float),  # grad_out (device)
        POINTER(c_float),  # grad_x (device)
        c_int,
        c_int,  # N, C
        c_int,
        c_int,  # H, W
    ]
    lib.keydnn_cuda_global_avgpool2d_backward_f32.restype = c_int

    lib.keydnn_cuda_global_avgpool2d_backward_f64.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_int,
        c_int,
        c_int,
    ]
    lib.keydnn_cuda_global_avgpool2d_backward_f64.restype = c_int


def _as_dev_ptr_float(dev_ptr: DevPtr):
    """
    Cast a device pointer handle to float* for ctypes calls.

    Notes
    -----
    We cast through c_void_p to avoid 'wrong type' errors on some platforms.
    """
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_float))


def _as_dev_ptr_double(dev_ptr: DevPtr):
    """
    Cast a device pointer handle to double* for ctypes calls.
    """
    return ctypes.cast(c_void_p(int(dev_ptr)), POINTER(c_double))


def global_avgpool2d_forward_cuda(
    lib: ctypes.CDLL,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: np.dtype,
) -> None:
    """
    Run CUDA GlobalAvgPool2D forward on device buffers.

    Parameters
    ----------
    x_dev : DevPtr
        Device pointer to input x (N,C,H,W).
    y_dev : DevPtr
        Device pointer to output y (N,C,1,1), stored as N*C contiguous values.
    dtype : np.dtype
        np.float32 or np.float64 selects the kernel.

    Raises
    ------
    RuntimeError
        If the CUDA kernel reports failure.
    """
    _bind_global_avgpool2d(lib)

    if dtype == np.float32:
        st = lib.keydnn_cuda_global_avgpool2d_forward_f32(
            _as_dev_ptr_float(x_dev),
            _as_dev_ptr_float(y_dev),
            int(N),
            int(C),
            int(H),
            int(W),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_global_avgpool2d_forward_f64(
            _as_dev_ptr_double(x_dev),
            _as_dev_ptr_double(y_dev),
            int(N),
            int(C),
            int(H),
            int(W),
        )
    else:
        raise TypeError(f"Unsupported dtype for global_avgpool2d_forward_cuda: {dtype}")

    if st != 0:
        raise RuntimeError(
            f"keydnn_cuda_global_avgpool2d_forward failed with status={st}"
        )


def global_avgpool2d_backward_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,
    grad_x_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: np.dtype,
) -> None:
    """
    Run CUDA GlobalAvgPool2D backward on device buffers.

    Parameters
    ----------
    grad_out_dev : DevPtr
        Device pointer to grad_out (N,C,1,1), stored as N*C values.
    grad_x_dev : DevPtr
        Device pointer to grad_x (N,C,H,W).

    Notes
    -----
    This kernel overwrites all grad_x elements, so zero-initialization is not required.
    """
    _bind_global_avgpool2d(lib)

    if dtype == np.float32:
        st = lib.keydnn_cuda_global_avgpool2d_backward_f32(
            _as_dev_ptr_float(grad_out_dev),
            _as_dev_ptr_float(grad_x_dev),
            int(N),
            int(C),
            int(H),
            int(W),
        )
    elif dtype == np.float64:
        st = lib.keydnn_cuda_global_avgpool2d_backward_f64(
            _as_dev_ptr_double(grad_out_dev),
            _as_dev_ptr_double(grad_x_dev),
            int(N),
            int(C),
            int(H),
            int(W),
        )
    else:
        raise TypeError(
            f"Unsupported dtype for global_avgpool2d_backward_cuda: {dtype}"
        )

    if st != 0:
        raise RuntimeError(
            f"keydnn_cuda_global_avgpool2d_backward failed with status={st}"
        )


def global_avgpool2d_forward_cuda_from_numpy(
    lib: ctypes.CDLL,
    *,
    x: np.ndarray,
    device: int = 0,
) -> np.ndarray:
    """
    Convenience wrapper for correctness testing:
    - copy x (CPU) -> GPU
    - run forward
    - copy y (GPU) -> CPU
    - free GPU buffers

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N,C,H,W), float32/float64.
    device : int
        CUDA device index.

    Returns
    -------
    np.ndarray
        Output tensor of shape (N,C,1,1) on CPU.
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"x must be float32/float64, got {x.dtype}")
    if x.ndim != 4:
        raise ValueError(f"x must be 4D (N,C,H,W), got shape={x.shape}")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)

    N, C, H, W = map(int, x.shape)

    cuda_set_device(lib, device)

    # output is stored as N*C contiguous values on device; we reshape to (N,C,1,1) on host
    y = np.empty((N, C, 1, 1), dtype=x.dtype)

    x_dev = cuda_from_host(lib, x)
    y_dev = cuda_malloc(lib, (N * C) * x.dtype.itemsize)

    try:
        global_avgpool2d_forward_cuda(
            lib,
            x_dev=x_dev,
            y_dev=y_dev,
            N=N,
            C=C,
            H=H,
            W=W,
            dtype=x.dtype,
        )
        # copy raw N*C into y view
        y_flat = y.reshape(N * C)
        cuda_memcpy_d2h(lib, y_flat, y_dev)
    finally:
        cuda_free(lib, x_dev)
        cuda_free(lib, y_dev)

    return y


def global_avgpool2d_backward_cuda_from_numpy(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    device: int = 0,
) -> np.ndarray:
    """
    Convenience wrapper for correctness testing:
    - copy grad_out (CPU) -> GPU
    - run backward
    - copy grad_x (GPU) -> CPU
    - free GPU buffers

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient w.r.t output, shape (N,C,1,1), float32/float64.
    x_shape : (N,C,H,W)
        Original input shape.
    device : int
        CUDA device index.

    Returns
    -------
    np.ndarray
        grad_x on CPU, shape (N,C,H,W).
    """
    if grad_out.dtype not in (np.float32, np.float64):
        raise TypeError(f"grad_out must be float32/float64, got {grad_out.dtype}")
    if grad_out.ndim != 4 or grad_out.shape[2:] != (1, 1):
        raise ValueError(f"grad_out must have shape (N,C,1,1), got {grad_out.shape}")
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    N, C, H, W = map(int, x_shape)
    if grad_out.shape[0] != N or grad_out.shape[1] != C:
        raise ValueError(
            f"grad_out shape {grad_out.shape} incompatible with x_shape {x_shape}"
        )

    cuda_set_device(lib, device)

    grad_x = np.empty((N, C, H, W), dtype=grad_out.dtype)

    # device buffers: grad_out stored as N*C contiguous values
    go_flat = grad_out.reshape(N * C)
    go_dev = cuda_from_host(lib, go_flat)
    gx_dev = cuda_malloc(lib, grad_x.nbytes)

    try:
        global_avgpool2d_backward_cuda(
            lib,
            grad_out_dev=go_dev,
            grad_x_dev=gx_dev,
            N=N,
            C=C,
            H=H,
            W=W,
            dtype=grad_out.dtype,
        )
        cuda_memcpy_d2h(lib, grad_x, gx_dev)
    finally:
        cuda_free(lib, go_dev)
        cuda_free(lib, gx_dev)

    return grad_x
