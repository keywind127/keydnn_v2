"""
CUDA pooling kernels for KeyDNN (device-pointer first).

This module provides infrastructure-level pooling kernels that execute on CUDA
by calling the KeyDNN v2 CUDA native DLL via `ctypes`.

Key design
----------
These functions operate on **device pointers** directly and do not convert CUDA
tensors to NumPy. They are intended to be called by Tensor-boundary wrappers
(e.g. `pool2d_cuda_ext.py`).

Utilities included
------------------
- pad2d_cuda:  build padded tensors entirely on GPU (pad with -inf or 0)
- crop2d_cuda: extract unpadded regions entirely on GPU

Implemented pooling variants
----------------------------
- MaxPool2D (forward + backward)
- AvgPool2D (forward + backward)
- GlobalAvgPool2D (forward + backward)
"""

from __future__ import annotations

import numpy as np

DevPtr = int


def _load_cuda_lib():
    """
    Load the KeyDNN CUDA native DLL.

    Returns
    -------
    ctypes.CDLL
        Loaded CUDA DLL handle.
    """
    from ..native_cuda.python.maxpool2d_ctypes import load_keydnn_cuda_native

    return load_keydnn_cuda_native()


def cuda_set_device(lib, device: int) -> None:
    """Set current CUDA device (thin wrapper)."""
    from ..native_cuda.python.maxpool2d_ctypes import cuda_set_device as _set

    _set(lib, int(device))


def cuda_malloc(lib, nbytes: int) -> DevPtr:
    """Allocate device memory and return DevPtr."""
    from ..native_cuda.python.maxpool2d_ctypes import cuda_malloc as _malloc

    return int(_malloc(lib, int(nbytes)))


def cuda_free(lib, dev_ptr: DevPtr) -> None:
    """Free device memory."""
    from ..native_cuda.python.maxpool2d_ctypes import cuda_free as _free

    _free(lib, int(dev_ptr))


def cuda_memset(lib, dev_ptr: DevPtr, value: int, nbytes: int) -> None:
    """Byte-wise memset of device memory."""
    from ..native_cuda.python.maxpool2d_ctypes import cuda_memset as _memset

    _memset(lib, int(dev_ptr), int(value), int(nbytes))


def pad2d_cuda(
    lib,
    *,
    x_dev: DevPtr,
    y_pad_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    p_h: int,
    p_w: int,
    pad_value: float,
    dtype: np.dtype,
    device: int = 0,
    sync: bool = True,
) -> None:
    """
    GPU pad2d helper: create padded tensor on device.

    For MaxPool2D use pad_value=-inf; for AvgPool2D use pad_value=0.
    """
    from ..native_cuda.python.pad2d_cuda_ctypes import pad2d_cuda as _pad

    _pad(
        lib,
        x_dev=int(x_dev),
        y_pad_dev=int(y_pad_dev),
        N=int(N),
        C=int(C),
        H=int(H),
        W=int(W),
        p_h=int(p_h),
        p_w=int(p_w),
        pad_value=float(pad_value),
        dtype=dtype,
        device=int(device),
        sync=bool(sync),
    )


def crop2d_cuda(
    lib,
    *,
    x_pad_dev: DevPtr,
    y_dev: DevPtr,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    p_h: int,
    p_w: int,
    H: int,
    W: int,
    dtype: np.dtype,
    device: int = 0,
    sync: bool = True,
) -> None:
    """GPU crop2d helper: extract unpadded region on device."""
    from ..native_cuda.python.pad2d_cuda_ctypes import crop2d_cuda as _crop

    _crop(
        lib,
        x_pad_dev=int(x_pad_dev),
        y_dev=int(y_dev),
        N=int(N),
        C=int(C),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        p_h=int(p_h),
        p_w=int(p_w),
        H=int(H),
        W=int(W),
        dtype=dtype,
        device=int(device),
        sync=bool(sync),
    )


def maxpool2d_forward_cuda(
    lib,
    *,
    x_pad_dev: DevPtr,
    y_dev: DevPtr,
    argmax_idx_dev: DevPtr,
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
    """Device-pointer MaxPool2D forward."""
    from ..native_cuda.python.maxpool2d_ctypes import maxpool2d_forward_cuda as _fwd

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"maxpool2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    _fwd(
        lib,
        x_pad_dev=int(x_pad_dev),
        y_dev=int(y_dev),
        argmax_idx_dev=int(argmax_idx_dev),
        N=int(N),
        C=int(C),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        H_out=int(H_out),
        W_out=int(W_out),
        k_h=int(k_h),
        k_w=int(k_w),
        s_h=int(s_h),
        s_w=int(s_w),
        dtype=dtype,
        sync=bool(sync),
    )


def maxpool2d_backward_cuda(
    lib,
    *,
    grad_out_dev: DevPtr,
    argmax_idx_dev: DevPtr,
    grad_x_pad_dev: DevPtr,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """Device-pointer MaxPool2D backward."""
    from ..native_cuda.python.maxpool2d_ctypes import maxpool2d_backward_cuda as _bwd

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"maxpool2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    _bwd(
        lib,
        grad_out_dev=int(grad_out_dev),
        argmax_idx_dev=int(argmax_idx_dev),
        grad_x_pad_dev=int(grad_x_pad_dev),
        N=int(N),
        C=int(C),
        H_out=int(H_out),
        W_out=int(W_out),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        dtype=dtype,
        sync=bool(sync),
    )


def avgpool2d_forward_cuda(
    lib,
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
    """Device-pointer AvgPool2D forward."""
    from ..native_cuda.python.avgpool2d_ctypes import avgpool2d_forward_cuda as _fwd

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"avgpool2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    _fwd(
        lib,
        x_pad_dev=int(x_pad_dev),
        y_dev=int(y_dev),
        N=int(N),
        C=int(C),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        H_out=int(H_out),
        W_out=int(W_out),
        k_h=int(k_h),
        k_w=int(k_w),
        s_h=int(s_h),
        s_w=int(s_w),
        dtype=dtype,
        sync=bool(sync),
    )


def avgpool2d_backward_cuda(
    lib,
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
    """Device-pointer AvgPool2D backward."""
    from ..native_cuda.python.avgpool2d_ctypes import avgpool2d_backward_cuda as _bwd

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"avgpool2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    _bwd(
        lib,
        grad_out_dev=int(grad_out_dev),
        grad_x_pad_dev=int(grad_x_pad_dev),
        N=int(N),
        C=int(C),
        H_out=int(H_out),
        W_out=int(W_out),
        H_pad=int(H_pad),
        W_pad=int(W_pad),
        k_h=int(k_h),
        k_w=int(k_w),
        s_h=int(s_h),
        s_w=int(s_w),
        dtype=dtype,
        sync=bool(sync),
    )


def global_avgpool2d_forward_cuda(
    lib,
    *,
    x_dev: DevPtr,
    y_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """Device-pointer GlobalAvgPool2D forward."""
    from ..native_cuda.python.global_avgpool2d_ctypes import (
        global_avgpool2d_forward_cuda as _fwd,
    )

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"global_avgpool2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    _fwd(
        lib,
        x_dev=int(x_dev),
        y_dev=int(y_dev),
        N=int(N),
        C=int(C),
        H=int(H),
        W=int(W),
        dtype=dtype,
        sync=bool(sync),
    )


def global_avgpool2d_backward_cuda(
    lib,
    *,
    grad_out_dev: DevPtr,
    grad_x_dev: DevPtr,
    N: int,
    C: int,
    H: int,
    W: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """Device-pointer GlobalAvgPool2D backward."""
    from ..native_cuda.python.global_avgpool2d_ctypes import (
        global_avgpool2d_backward_cuda as _bwd,
    )

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"global_avgpool2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    _bwd(
        lib,
        grad_out_dev=int(grad_out_dev),
        grad_x_dev=int(grad_x_dev),
        N=int(N),
        C=int(C),
        H=int(H),
        W=int(W),
        dtype=dtype,
        sync=bool(sync),
    )
