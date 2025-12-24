"""
ctypes bindings for KeyDNN v2 CUDA AvgPool2D kernels.

This module provides low-level Python bindings to the CUDA implementation of
AvgPool2D forward/backward via `ctypes`. It is backend-specific and assumes:

- The CUDA DLL exports C ABI functions:
    - keydnn_cuda_set_device
    - keydnn_cuda_malloc / keydnn_cuda_free
    - keydnn_cuda_memcpy_h2d / keydnn_cuda_memcpy_d2h
    - keydnn_cuda_memset
    - keydnn_cuda_synchronize
    - keydnn_cuda_avgpool2d_forward_f32 / _f64
    - keydnn_cuda_avgpool2d_backward_f32 / _f64

- Device pointers are represented as uintptr_t handles (Python int).
- Tensors are NCHW contiguous.
- Forward expects x_pad already padded (typically zero-padded for avg pooling).
- Backward expects grad_x_pad device buffer to be zero-initialized before calling.

Platform notes
--------------
This wrapper currently targets Windows and loads the CUDA DLL from the build
output path used by your Visual Studio project.

Design notes
------------
- Strict dtype and contiguity validation to avoid undefined behavior.
- No implicit device memory caching: caller owns device pointers and frees them.
- Uses `c_void_p` for device pointers in function signatures to avoid fragile
  ctypes pointer-casting issues on Windows.
- These functions are infrastructure-layer utilities; higher-level Tensor/Module
  code should wrap them with safe abstractions.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    c_int,
    c_size_t,
    c_void_p,
    c_uint64,
)
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------
# DLL loading
# ---------------------------------------------------------------------

_DEFAULT_DLL_PATH = Path(
    r"D:\keydnn_v2\src\keydnn\infrastructure\native_cuda\keydnn_v2_cuda_native\x64\Debug\KeyDNNV2CudaNative.dll"
)


def load_keydnn_cuda_native(dll_path: str | Path = _DEFAULT_DLL_PATH) -> ctypes.CDLL:
    """
    Load the KeyDNN v2 CUDA native DLL.

    Parameters
    ----------
    dll_path : str | Path
        Path to KeyDNNV2CudaNative.dll.

    Returns
    -------
    ctypes.CDLL
        Loaded DLL handle.

    Raises
    ------
    FileNotFoundError
        If the DLL path does not exist.
    OSError
        If the DLL cannot be loaded (missing dependencies, wrong arch, etc.).
    """
    p = Path(dll_path)
    if not p.exists():
        raise FileNotFoundError(f"CUDA native DLL not found: {p}")
    return ctypes.CDLL(str(p))


# ---------------------------------------------------------------------
# Low-level CUDA utility bindings (same style as maxpool2d_cuda_ctypes)
# ---------------------------------------------------------------------

DevPtr = int


class CudaLib:
    """
    Thin binding layer around the KeyDNN CUDA native DLL.

    This class performs one-time `argtypes`/`restype` binding for exported symbols
    and exposes helpers for common CUDA operations and AvgPool2D dispatch.

    Notes
    -----
    - This class does not manage device pointers automatically; callers must
      free device allocations using `cuda_free`.
    - Pointers passed between Python and the DLL are treated as raw addresses
      (`void*`) on the Python side.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        """
        Create a CUDA binding wrapper around an already-loaded DLL.

        Parameters
        ----------
        lib : ctypes.CDLL
            Loaded KeyDNNV2CudaNative.dll handle.
        """
        self.lib = lib
        self._cuda_utils_bound = False
        self._avgpool_bound = False

    def _bind_cuda_utils(self) -> None:
        """Bind argtypes/restype for CUDA utility exports (idempotent)."""
        if self._cuda_utils_bound:
            return

        lib = self.lib
        lib.keydnn_cuda_set_device.argtypes = [c_int]
        lib.keydnn_cuda_set_device.restype = c_int

        lib.keydnn_cuda_malloc.argtypes = [ctypes.POINTER(c_uint64), c_size_t]
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

        self._cuda_utils_bound = True

    def _bind_avgpool2d(self) -> None:
        """
        Bind argtypes/restype for AvgPool2D CUDA exports (idempotent).

        We bind device pointers as `c_void_p` to keep pointer passing robust.
        """
        if self._avgpool_bound:
            return

        lib = self.lib

        # forward f32
        lib.keydnn_cuda_avgpool2d_forward_f32.argtypes = [
            c_void_p,  # x_pad (device)
            c_void_p,  # y (device)
            c_int,
            c_int,  # N, C
            c_int,
            c_int,  # H_pad, W_pad
            c_int,
            c_int,  # H_out, W_out
            c_int,
            c_int,  # k_h, k_w
            c_int,
            c_int,  # s_h, s_w
        ]
        lib.keydnn_cuda_avgpool2d_forward_f32.restype = c_int

        # forward f64
        lib.keydnn_cuda_avgpool2d_forward_f64.argtypes = [
            c_void_p,
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        lib.keydnn_cuda_avgpool2d_forward_f64.restype = c_int

        # backward f32
        lib.keydnn_cuda_avgpool2d_backward_f32.argtypes = [
            c_void_p,  # grad_out (device)
            c_void_p,  # grad_x_pad (device)
            c_int,
            c_int,  # N, C
            c_int,
            c_int,  # H_out, W_out
            c_int,
            c_int,  # H_pad, W_pad
            c_int,
            c_int,  # k_h, k_w
            c_int,
            c_int,  # s_h, s_w
        ]
        lib.keydnn_cuda_avgpool2d_backward_f32.restype = c_int

        # backward f64
        lib.keydnn_cuda_avgpool2d_backward_f64.argtypes = [
            c_void_p,
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        lib.keydnn_cuda_avgpool2d_backward_f64.restype = c_int

        self._avgpool_bound = True

    @staticmethod
    def _as_dev_ptr(dev_ptr: DevPtr) -> c_void_p:
        """
        Convert a device pointer handle (Python int) into a ctypes void*.

        Parameters
        ----------
        dev_ptr : DevPtr
            Device address stored as an integer (uintptr_t).

        Returns
        -------
        ctypes.c_void_p
            Raw pointer value usable in ctypes calls.
        """
        return c_void_p(int(dev_ptr))

    # ----------------------------
    # CUDA utils
    # ----------------------------

    def cuda_set_device(self, device: int = 0) -> None:
        """
        Set the current CUDA device.

        Raises
        ------
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_set_device(int(device))
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_set_device failed with status={st}")

    def cuda_malloc(self, nbytes: int) -> DevPtr:
        """
        Allocate device memory.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.

        Returns
        -------
        DevPtr
            Device pointer handle (uintptr_t as Python int).

        Raises
        ------
        RuntimeError
            If allocation fails.
        """
        self._bind_cuda_utils()
        out = c_uint64(0)
        st = self.lib.keydnn_cuda_malloc(ctypes.byref(out), c_size_t(int(nbytes)))
        if st != 0 or out.value == 0:
            raise RuntimeError(
                f"keydnn_cuda_malloc failed with status={st}, nbytes={nbytes}"
            )
        return int(out.value)

    def cuda_free(self, dev_ptr: DevPtr) -> None:
        """
        Free device memory.

        Parameters
        ----------
        dev_ptr : DevPtr
            Device pointer handle. Safe to pass 0.

        Raises
        ------
        RuntimeError
            If the native free call fails.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_free(c_uint64(int(dev_ptr)))
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_free failed with status={st}")

    def cuda_memcpy_h2d(self, dst_dev: DevPtr, src_host: np.ndarray) -> None:
        """
        Copy a NumPy array from host to device.

        Parameters
        ----------
        dst_dev : DevPtr
            Destination device buffer pointer.
        src_host : np.ndarray
            Source host array. If not C-contiguous, it will be copied into a
            contiguous buffer before transfer.
        """
        self._bind_cuda_utils()
        if not src_host.flags["C_CONTIGUOUS"]:
            src_host = np.ascontiguousarray(src_host)

        st = self.lib.keydnn_cuda_memcpy_h2d(
            c_uint64(int(dst_dev)),
            c_void_p(int(src_host.ctypes.data)),
            c_size_t(int(src_host.nbytes)),
        )
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")

    def cuda_memcpy_d2h(self, dst_host: np.ndarray, src_dev: DevPtr) -> None:
        """
        Copy from device to a NumPy host array.

        Parameters
        ----------
        dst_host : np.ndarray
            Destination host array. Must be C-contiguous.
        src_dev : DevPtr
            Source device pointer.

        Raises
        ------
        ValueError
            If dst_host is not C-contiguous.
        """
        self._bind_cuda_utils()
        if not dst_host.flags["C_CONTIGUOUS"]:
            raise ValueError("dst_host must be C-contiguous")

        st = self.lib.keydnn_cuda_memcpy_d2h(
            c_void_p(int(dst_host.ctypes.data)),
            c_uint64(int(src_dev)),
            c_size_t(int(dst_host.nbytes)),
        )
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")

    def cuda_memset(self, dev_ptr: DevPtr, value: int, nbytes: int) -> None:
        """
        Set a device buffer to a byte value.

        Parameters
        ----------
        dev_ptr : DevPtr
            Device pointer handle.
        value : int
            Byte value [0,255] used by cudaMemset.
        nbytes : int
            Number of bytes to set.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_memset(
            c_uint64(int(dev_ptr)), c_int(int(value)), c_size_t(int(nbytes))
        )
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_memset failed with status={st}")

    def cuda_synchronize(self) -> None:
        """
        Synchronize the device (blocking).

        Raises
        ------
        RuntimeError
            If device synchronization fails.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_synchronize()
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_synchronize failed with status={st}")

    def cuda_from_host(self, x: np.ndarray) -> DevPtr:
        """
        Convenience helper: allocate a device buffer and copy a host array to GPU.

        Supported dtypes
        ----------------
        - np.float32
        - np.float64

        Parameters
        ----------
        x : np.ndarray
            Host array to upload. If not C-contiguous, it will be made contiguous.

        Returns
        -------
        DevPtr
            Device pointer handle.

        Raises
        ------
        TypeError
            If dtype is unsupported.
        """
        if x.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"cuda_from_host only supports float32/float64, got {x.dtype}"
            )
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        dev = self.cuda_malloc(x.nbytes)
        try:
            self.cuda_memcpy_h2d(dev, x)
        except Exception:
            self.cuda_free(dev)
            raise
        return dev

    # ----------------------------
    # AvgPool2D
    # ----------------------------

    def avgpool2d_forward_cuda(
        self,
        *,
        x_pad_dev: DevPtr,
        y_dev: DevPtr,
        N: int,
        C: int,
        H_pad: int,
        W_pad: int,
        H_out: int,
        W_out: int,
        k_h: int,
        k_w: int,
        s_h: int,
        s_w: int,
        dtype: np.dtype,
        sync: bool = True,
    ) -> None:
        """
        Run CUDA AvgPool2D forward on device buffers.

        Parameters
        ----------
        x_pad_dev, y_dev : DevPtr
            Device pointer handles (raw addresses).
        N, C : int
            Batch size and channels.
        H_pad, W_pad : int
            Spatial dimensions of padded input (zero padded for avg pooling).
        H_out, W_out : int
            Output spatial dimensions.
        k_h, k_w : int
            Pooling kernel size.
        s_h, s_w : int
            Strides.
        dtype : np.dtype
            np.float32 or np.float64 selects the kernel variant.
        sync : bool
            If True, call `cuda_synchronize` after kernel launch.
        """
        self._bind_avgpool2d()

        if dtype == np.float32:
            st = self.lib.keydnn_cuda_avgpool2d_forward_f32(
                self._as_dev_ptr(x_pad_dev),
                self._as_dev_ptr(y_dev),
                int(N),
                int(C),
                int(H_pad),
                int(W_pad),
                int(H_out),
                int(W_out),
                int(k_h),
                int(k_w),
                int(s_h),
                int(s_w),
            )
        elif dtype == np.float64:
            st = self.lib.keydnn_cuda_avgpool2d_forward_f64(
                self._as_dev_ptr(x_pad_dev),
                self._as_dev_ptr(y_dev),
                int(N),
                int(C),
                int(H_pad),
                int(W_pad),
                int(H_out),
                int(W_out),
                int(k_h),
                int(k_w),
                int(s_h),
                int(s_w),
            )
        else:
            raise TypeError(f"Unsupported dtype for avgpool2d_forward_cuda: {dtype}")

        if st != 0:
            raise RuntimeError(f"keydnn_cuda_avgpool2d_forward failed with status={st}")

        if sync:
            self.cuda_synchronize()

    def avgpool2d_backward_cuda(
        self,
        *,
        grad_out_dev: DevPtr,
        grad_x_pad_dev: DevPtr,
        N: int,
        C: int,
        H_out: int,
        W_out: int,
        H_pad: int,
        W_pad: int,
        k_h: int,
        k_w: int,
        s_h: int,
        s_w: int,
        dtype: np.dtype,
        sync: bool = True,
    ) -> None:
        """
        Run CUDA AvgPool2D backward on device buffers.

        Parameters
        ----------
        grad_out_dev : DevPtr
            Device pointer to grad_out, shape (N, C, H_out, W_out).
        grad_x_pad_dev : DevPtr
            Device pointer to grad_x_pad, shape (N, C, H_pad, W_pad).
            Must be zero-initialized before calling.
        dtype : np.dtype
            np.float32 or np.float64 selects the kernel variant.
        sync : bool
            If True, call `cuda_synchronize` after kernel launch.

        Notes
        -----
        This backward implementation scatters each output gradient uniformly
        over the k_h x k_w receptive field (using atomic adds). Callers must
        ensure `grad_x_pad_dev` is zeroed before invoking.
        """
        self._bind_avgpool2d()

        if dtype == np.float32:
            st = self.lib.keydnn_cuda_avgpool2d_backward_f32(
                self._as_dev_ptr(grad_out_dev),
                self._as_dev_ptr(grad_x_pad_dev),
                int(N),
                int(C),
                int(H_out),
                int(W_out),
                int(H_pad),
                int(W_pad),
                int(k_h),
                int(k_w),
                int(s_h),
                int(s_w),
            )
        elif dtype == np.float64:
            st = self.lib.keydnn_cuda_avgpool2d_backward_f64(
                self._as_dev_ptr(grad_out_dev),
                self._as_dev_ptr(grad_x_pad_dev),
                int(N),
                int(C),
                int(H_out),
                int(W_out),
                int(H_pad),
                int(W_pad),
                int(k_h),
                int(k_w),
                int(s_h),
                int(s_w),
            )
        else:
            raise TypeError(f"Unsupported dtype for avgpool2d_backward_cuda: {dtype}")

        if st != 0:
            raise RuntimeError(
                f"keydnn_cuda_avgpool2d_backward failed with status={st}"
            )

        if sync:
            self.cuda_synchronize()


# ---------------------------------------------------------------------
# Functional API (same pattern as maxpool2d wrapper)
# ---------------------------------------------------------------------

_cuda_singleton: CudaLib | None = None


def _get_cuda(lib: ctypes.CDLL) -> CudaLib:
    """
    Return a cached `CudaLib` wrapper for a given `ctypes.CDLL`.

    This avoids repeating argtype binding on every call and keeps the public API
    of this module function-based.
    """
    global _cuda_singleton
    if _cuda_singleton is None or _cuda_singleton.lib is not lib:
        _cuda_singleton = CudaLib(lib)
    return _cuda_singleton


def cuda_set_device(lib: ctypes.CDLL, device: int = 0) -> None:
    """Set the active CUDA device (module-level convenience wrapper)."""
    _get_cuda(lib).cuda_set_device(device)


def cuda_malloc(lib: ctypes.CDLL, nbytes: int) -> DevPtr:
    """Allocate device memory and return a device pointer handle."""
    return _get_cuda(lib).cuda_malloc(nbytes)


def cuda_free(lib: ctypes.CDLL, dev_ptr: DevPtr) -> None:
    """Free device memory for a given device pointer handle."""
    _get_cuda(lib).cuda_free(dev_ptr)


def cuda_memcpy_h2d(lib: ctypes.CDLL, dst_dev: DevPtr, src_host: np.ndarray) -> None:
    """Copy a NumPy host array to a device buffer."""
    _get_cuda(lib).cuda_memcpy_h2d(dst_dev, src_host)


def cuda_memcpy_d2h(lib: ctypes.CDLL, dst_host: np.ndarray, src_dev: DevPtr) -> None:
    """Copy a device buffer into a NumPy host array."""
    _get_cuda(lib).cuda_memcpy_d2h(dst_host, src_dev)


def cuda_memset(lib: ctypes.CDLL, dev_ptr: DevPtr, value: int, nbytes: int) -> None:
    """Set device memory to a byte value."""
    _get_cuda(lib).cuda_memset(dev_ptr, value, nbytes)


def cuda_synchronize(lib: ctypes.CDLL) -> None:
    """Synchronize CUDA device execution."""
    _get_cuda(lib).cuda_synchronize()


def cuda_from_host(lib: ctypes.CDLL, x: np.ndarray) -> DevPtr:
    """Allocate a device buffer and upload a NumPy array (float32/float64 only)."""
    return _get_cuda(lib).cuda_from_host(x)


def avgpool2d_forward_cuda(
    lib: ctypes.CDLL,
    *,
    x_pad_dev: DevPtr,
    y_dev: DevPtr,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Module-level convenience wrapper for CUDA AvgPool2D forward.

    See `CudaLib.avgpool2d_forward_cuda` for full parameter details.
    """
    _get_cuda(lib).avgpool2d_forward_cuda(
        x_pad_dev=x_pad_dev,
        y_dev=y_dev,
        N=N,
        C=C,
        H_pad=H_pad,
        W_pad=W_pad,
        H_out=H_out,
        W_out=W_out,
        k_h=k_h,
        k_w=k_w,
        s_h=s_h,
        s_w=s_w,
        dtype=dtype,
        sync=sync,
    )


def avgpool2d_backward_cuda(
    lib: ctypes.CDLL,
    *,
    grad_out_dev: DevPtr,
    grad_x_pad_dev: DevPtr,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Module-level convenience wrapper for CUDA AvgPool2D backward.

    See `CudaLib.avgpool2d_backward_cuda` for full parameter details.
    """
    _get_cuda(lib).avgpool2d_backward_cuda(
        grad_out_dev=grad_out_dev,
        grad_x_pad_dev=grad_x_pad_dev,
        N=N,
        C=C,
        H_out=H_out,
        W_out=W_out,
        H_pad=H_pad,
        W_pad=W_pad,
        k_h=k_h,
        k_w=k_w,
        s_h=s_h,
        s_w=s_w,
        dtype=dtype,
        sync=sync,
    )


# ---------------------------------------------------------------------
# Optional convenience: end-to-end CPU NumPy -> GPU -> CPU for testing
# ---------------------------------------------------------------------


def avgpool2d_forward_cuda_from_numpy(
    lib: ctypes.CDLL,
    *,
    x_pad: np.ndarray,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    device: int = 0,
) -> np.ndarray:
    """
    Convenience forward wrapper for quick correctness testing:
    - copies x_pad (CPU) -> GPU
    - runs avgpool2d forward
    - copies y back to CPU
    - frees GPU buffers

    Parameters
    ----------
    x_pad : np.ndarray
        Padded input tensor on CPU, dtype float32/float64, shape (N,C,H_pad,W_pad).
    device : int
        CUDA device ordinal to use.

    Returns
    -------
    np.ndarray
        Output tensor y, dtype float32/float64, shape (N,C,H_out,W_out).
    """
    if x_pad.dtype not in (np.float32, np.float64):
        raise TypeError(f"x_pad must be float32/float64, got {x_pad.dtype}")
    if not x_pad.flags["C_CONTIGUOUS"]:
        x_pad = np.ascontiguousarray(x_pad)

    cuda = _get_cuda(lib)
    cuda.cuda_set_device(device)

    y = np.empty((N, C, H_out, W_out), dtype=x_pad.dtype)

    x_dev = cuda.cuda_from_host(x_pad)
    y_dev = cuda.cuda_malloc(y.nbytes)

    try:
        cuda.avgpool2d_forward_cuda(
            x_pad_dev=x_dev,
            y_dev=y_dev,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
            dtype=x_pad.dtype,
            sync=True,
        )
        cuda.cuda_memcpy_d2h(y, y_dev)
    finally:
        cuda.cuda_free(x_dev)
        cuda.cuda_free(y_dev)

    return y


def avgpool2d_backward_cuda_from_numpy(
    lib: ctypes.CDLL,
    *,
    grad_out: np.ndarray,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    device: int = 0,
) -> np.ndarray:
    """
    Convenience backward wrapper for quick correctness testing:
    - copies grad_out (CPU) -> GPU
    - allocates grad_x_pad on GPU and zeros it
    - runs avgpool2d backward
    - copies grad_x_pad back to CPU
    - frees GPU buffers

    Parameters
    ----------
    grad_out : np.ndarray
        Output gradient on CPU, dtype float32/float64, shape (N,C,H_out,W_out).
    device : int
        CUDA device ordinal to use.

    Returns
    -------
    np.ndarray
        grad_x_pad, dtype float32/float64, shape (N,C,H_pad,W_pad).
    """
    if grad_out.dtype not in (np.float32, np.float64):
        raise TypeError(f"grad_out must be float32/float64, got {grad_out.dtype}")
    if not grad_out.flags["C_CONTIGUOUS"]:
        grad_out = np.ascontiguousarray(grad_out)

    cuda = _get_cuda(lib)
    cuda.cuda_set_device(device)

    grad_x_pad = np.zeros((N, C, H_pad, W_pad), dtype=grad_out.dtype)

    go_dev = cuda.cuda_from_host(grad_out)
    gx_dev = cuda.cuda_malloc(grad_x_pad.nbytes)

    try:
        cuda.cuda_memset(gx_dev, 0, grad_x_pad.nbytes)
        cuda.avgpool2d_backward_cuda(
            grad_out_dev=go_dev,
            grad_x_pad_dev=gx_dev,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            H_pad=H_pad,
            W_pad=W_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
            dtype=grad_out.dtype,
            sync=True,
        )
        cuda.cuda_memcpy_d2h(grad_x_pad, gx_dev)
    finally:
        cuda.cuda_free(go_dev)
        cuda.cuda_free(gx_dev)

    return grad_x_pad
