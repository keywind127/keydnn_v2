# infrastructure/ops/pool2d_cuda_ext.py
"""
CUDA pooling primitives with Tensor boundaries (device-pointer based).

This module is the CUDA counterpart of `pool2d_cpu_ext.py`. Unlike the CPU
version, it does not call `to_numpy()` for CUDA tensors. Instead it:

- Reads `Tensor.data` as a device pointer (DevPtr) when `x.device.is_cuda()`
- Allocates output device buffers
- Calls CUDA kernels via ctypes wrappers
- Returns CUDA tensors via `Tensor._from_devptr`

Padding and cropping are performed entirely on GPU using `pad2d_cuda` and
`crop2d_cuda` helper kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .._tensor import Tensor
from .pool2d_cpu import _pair, _out_hw
from .pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memset,
    pad2d_cuda,
    crop2d_cuda,
    maxpool2d_forward_cuda,
    maxpool2d_backward_cuda,
    avgpool2d_forward_cuda,
    avgpool2d_backward_cuda,
    global_avgpool2d_forward_cuda,
    global_avgpool2d_backward_cuda,
)


def _require_cuda(x: Tensor, name: str) -> None:
    """Validate that a tensor is on CUDA."""
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """Validate dtype is float32/float64 and return dtype."""
    dt = x.dtype
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def maxpool2d_forward(
    x: Tensor,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
    device: int = 0,
) -> tuple[Tensor, object]:
    """
    MaxPool2D forward (CUDA, NCHW).

    Returns
    -------
    (y, argmax_idx_dev)
        y : Tensor on CUDA
        argmax_idx_dev : DevPtr (int) for int64 indices on CUDA
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x.shape
    k_h, k_w = k
    s_h, s_w = s
    p_h, p_w = p

    H_out, W_out = _out_hw(H, W, k, s, p)
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # Allocate x_pad and pad with -inf
    nbytes_x_pad = int(N * C * H_pad * W_pad * np.dtype(dt).itemsize)
    x_pad_dev = cuda_malloc(lib, nbytes_x_pad)

    try:
        pad2d_cuda(
            lib,
            x_dev=int(x.data),
            y_pad_dev=int(x_pad_dev),
            N=N,
            C=C,
            H=H,
            W=W,
            p_h=p_h,
            p_w=p_w,
            pad_value=float(-np.inf),
            dtype=dt,
            device=device,
            sync=True,
        )

        # Allocate y and argmax
        nbytes_y = int(N * C * H_out * W_out * np.dtype(dt).itemsize)
        nbytes_idx = int(N * C * H_out * W_out * np.dtype(np.int64).itemsize)

        y_dev = cuda_malloc(lib, nbytes_y)
        argmax_idx_dev = cuda_malloc(lib, nbytes_idx)

        try:
            maxpool2d_forward_cuda(
                lib,
                x_pad_dev=int(x_pad_dev),
                y_dev=int(y_dev),
                argmax_idx_dev=int(argmax_idx_dev),
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
                dtype=dt,
                sync=True,
            )

            y = Tensor._from_devptr(
                int(y_dev),
                shape=(N, C, H_out, W_out),
                dtype=dt,
                device=x.device,
                requires_grad=False,
            )
            return y, int(argmax_idx_dev)

        except Exception:
            cuda_free(lib, y_dev)
            cuda_free(lib, argmax_idx_dev)
            raise

    finally:
        cuda_free(lib, x_pad_dev)


def maxpool2d_backward(
    grad_out: Tensor,
    *,
    argmax_idx: object,
    x_shape,
    kernel_size,
    stride,
    padding,
    device: int = 0,
) -> Tensor:
    """
    MaxPool2D backward (CUDA, NCHW).

    Notes
    -----
    - Uses CUDA backward into grad_x_pad then crops back to (N,C,H,W) on device.
    """
    _require_cuda(grad_out, "grad_out")
    dt = _require_f32_f64(grad_out, "grad_out")

    k = _pair(kernel_size)
    s = _pair(stride)
    p = _pair(padding)

    N, C, H, W = x_shape
    p_h, p_w = p
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w
    H_out, W_out = grad_out.shape[2], grad_out.shape[3]

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # grad_x_pad
    nbytes_gx_pad = int(N * C * H_pad * W_pad * np.dtype(dt).itemsize)
    grad_x_pad_dev = cuda_malloc(lib, nbytes_gx_pad)
    cuda_memset(lib, grad_x_pad_dev, 0, nbytes_gx_pad)

    # grad_x (cropped)
    nbytes_gx = int(N * C * H * W * np.dtype(dt).itemsize)
    grad_x_dev = cuda_malloc(lib, nbytes_gx)

    try:
        maxpool2d_backward_cuda(
            lib,
            grad_out_dev=int(grad_out.data),
            argmax_idx_dev=int(argmax_idx),
            grad_x_pad_dev=int(grad_x_pad_dev),
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            H_pad=H_pad,
            W_pad=W_pad,
            dtype=dt,
            sync=True,
        )

        crop2d_cuda(
            lib,
            x_pad_dev=int(grad_x_pad_dev),
            y_dev=int(grad_x_dev),
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            p_h=p_h,
            p_w=p_w,
            H=H,
            W=W,
            dtype=dt,
            device=device,
            sync=True,
        )

        return Tensor._from_devptr(
            int(grad_x_dev),
            shape=(N, C, H, W),
            dtype=dt,
            device=grad_out.device,
            requires_grad=False,
        )

    except Exception:
        cuda_free(lib, grad_x_dev)
        raise

    finally:
        cuda_free(lib, grad_x_pad_dev)


def avgpool2d_forward(
    x: Tensor,
    *,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
    device: int = 0,
) -> Tensor:
    """
    AvgPool2D forward (CUDA, NCHW).

    Notes
    -----
    - Pads with 0 on device (same semantics as CPU reference).
    """
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x.shape
    k_h, k_w = k
    s_h, s_w = s
    p_h, p_w = p

    H_out, W_out = _out_hw(H, W, k, s, p)
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_x_pad = int(N * C * H_pad * W_pad * np.dtype(dt).itemsize)
    x_pad_dev = cuda_malloc(lib, nbytes_x_pad)

    try:
        pad2d_cuda(
            lib,
            x_dev=int(x.data),
            y_pad_dev=int(x_pad_dev),
            N=N,
            C=C,
            H=H,
            W=W,
            p_h=p_h,
            p_w=p_w,
            pad_value=0.0,
            dtype=dt,
            device=device,
            sync=True,
        )

        nbytes_y = int(N * C * H_out * W_out * np.dtype(dt).itemsize)
        y_dev = cuda_malloc(lib, nbytes_y)

        try:
            avgpool2d_forward_cuda(
                lib,
                x_pad_dev=int(x_pad_dev),
                y_dev=int(y_dev),
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
                dtype=dt,
                sync=True,
            )

            return Tensor._from_devptr(
                int(y_dev),
                shape=(N, C, H_out, W_out),
                dtype=dt,
                device=x.device,
                requires_grad=False,
            )

        except Exception:
            cuda_free(lib, y_dev)
            raise

    finally:
        cuda_free(lib, x_pad_dev)


def avgpool2d_backward(
    grad_out: Tensor,
    *,
    x_shape,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    device: int = 0,
) -> Tensor:
    """
    AvgPool2D backward (CUDA, NCHW).

    Notes
    -----
    - Accumulates into grad_x_pad then crops to (N,C,H,W) on device.
    """
    _require_cuda(grad_out, "grad_out")
    dt = _require_f32_f64(grad_out, "grad_out")

    N, C, H, W = x_shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding

    H_out, W_out = grad_out.shape[2], grad_out.shape[3]
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_gx_pad = int(N * C * H_pad * W_pad * np.dtype(dt).itemsize)
    grad_x_pad_dev = cuda_malloc(lib, nbytes_gx_pad)
    cuda_memset(lib, grad_x_pad_dev, 0, nbytes_gx_pad)

    nbytes_gx = int(N * C * H * W * np.dtype(dt).itemsize)
    grad_x_dev = cuda_malloc(lib, nbytes_gx)

    try:
        avgpool2d_backward_cuda(
            lib,
            grad_out_dev=int(grad_out.data),
            grad_x_pad_dev=int(grad_x_pad_dev),
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
            dtype=dt,
            sync=True,
        )

        crop2d_cuda(
            lib,
            x_pad_dev=int(grad_x_pad_dev),
            y_dev=int(grad_x_dev),
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            p_h=p_h,
            p_w=p_w,
            H=H,
            W=W,
            dtype=dt,
            device=device,
            sync=True,
        )

        return Tensor._from_devptr(
            int(grad_x_dev),
            shape=(N, C, H, W),
            dtype=dt,
            device=grad_out.device,
            requires_grad=False,
        )

    except Exception:
        cuda_free(lib, grad_x_dev)
        raise

    finally:
        cuda_free(lib, grad_x_pad_dev)


def global_avgpool2d_forward(x: Tensor, *, device: int = 0) -> Tensor:
    """GlobalAvgPool2D forward (CUDA)."""
    _require_cuda(x, "x")
    dt = _require_f32_f64(x, "x")

    N, C, H, W = x.shape

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_y = int(N * C * 1 * 1 * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    try:
        global_avgpool2d_forward_cuda(
            lib,
            x_dev=int(x.data),
            y_dev=int(y_dev),
            N=N,
            C=C,
            H=H,
            W=W,
            dtype=dt,
            sync=True,
        )

        return Tensor._from_devptr(
            int(y_dev),
            shape=(N, C, 1, 1),
            dtype=dt,
            device=x.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, y_dev)
        raise


def global_avgpool2d_backward(grad_out: Tensor, *, x_shape, device: int = 0) -> Tensor:
    """GlobalAvgPool2D backward (CUDA)."""
    _require_cuda(grad_out, "grad_out")
    dt = _require_f32_f64(grad_out, "grad_out")

    N, C, H, W = x_shape

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    nbytes_gx = int(N * C * H * W * np.dtype(dt).itemsize)
    gx_dev = cuda_malloc(lib, nbytes_gx)
    cuda_memset(lib, gx_dev, 0, nbytes_gx)

    try:
        global_avgpool2d_backward_cuda(
            lib,
            grad_out_dev=int(grad_out.data),
            grad_x_dev=int(gx_dev),
            N=N,
            C=C,
            H=H,
            W=W,
            dtype=dt,
            sync=True,
        )

        return Tensor._from_devptr(
            int(gx_dev),
            shape=(N, C, H, W),
            dtype=dt,
            device=grad_out.device,
            requires_grad=False,
        )
    except Exception:
        cuda_free(lib, gx_dev)
        raise
