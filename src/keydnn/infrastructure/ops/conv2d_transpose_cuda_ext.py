"""
infrastructure/ops/conv2d_transpose_cuda_ext.py

ConvTranspose2D (Conv2D Transpose) CUDA backend boundary helpers.

This module defines *boundary-layer* functions that bridge KeyDNN infrastructure
`Tensor` objects to the NumPy-facing CUDA transpose-conv ops wrapper and back.

Purpose
-------
The CUDA transpose-conv ops wrapper (`conv2d_transpose_forward_cuda`,
`conv2d_transpose_backward_cuda`) is currently NumPy-facing and manages CUDA
device buffers internally. To keep higher-level autograd / operator code
NumPy-free, all Tensor <-> NumPy and CUDA memcpy conversion is localized here.

Design notes
------------
- These helpers perform *no* autograd graph wiring. They only:
  1) validate CUDA tensors and dtype consistency,
  2) copy CUDA `Tensor` inputs to host NumPy arrays (D2H),
  3) call the NumPy-facing CUDA ops wrapper (which launches native kernels),
  4) copy outputs back to device (H2D) and wrap them as new CUDA `Tensor`s.
- Output tensors are created with explicit `requires_grad` flags supplied by the
  caller (typically the differentiable conv2d-transpose operation).

Backend constraints (current)
-----------------------------
- Host round-trip: D2H/H2D is required because the ops wrapper is NumPy-facing.
- A future "true CUDA boundary" can avoid host copies by calling ctypes kernels
  directly with device pointers once Tensor exposes stable raw devptr access for
  transpose-conv.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..tensor._tensor import Tensor
from ..native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cudaMemcpyDtoH,
    cudaMemcpyHtoD,
    cuda_synchronize,
)

from .conv2d_transpose_cuda import (
    conv2d_transpose_forward_cuda as _conv2d_t_fwd_ops,
    conv2d_transpose_backward_cuda as _conv2d_t_bwd_ops,
)


# -----------------------------------------------------------------------------
# Shared helpers (mirrors conv2d_cuda_ext.py)
# -----------------------------------------------------------------------------
def _ensure_cuda_tensor(t: Tensor, name: str) -> None:
    """
    Validate that a `Tensor` is CUDA-backed.

    Raises
    ------
    TypeError
        If `t` is not a CUDA tensor.
    """
    if not t.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA tensor, got device={t.device}")


def _pair(v: Tuple[int, int] | int) -> Tuple[int, int]:
    """Normalize an int-or-pair into a (h, w) tuple of ints."""
    return v if isinstance(v, tuple) else (int(v), int(v))


def _get_tensor_dev_ptr(t: Tensor) -> int:
    """
    Best-effort extraction of the underlying CUDA device pointer from a `Tensor`.

    Returns 0 if no non-zero pointer can be found (allowed for empty tensors).
    """
    if hasattr(t, "data"):
        try:
            v = int(getattr(t, "data"))
            if v != 0:
                return v
        except Exception:
            pass

    for name in ("_cuda_dev_ptr", "cuda_dev_ptr", "_dev_ptr", "dev_ptr", "_data_ptr"):
        if hasattr(t, name):
            try:
                v = int(getattr(t, name))
                if v != 0:
                    return v
            except Exception:
                continue

    return 0


def _to_numpy_via_d2h(t: Tensor, *, lib, device_index: int) -> np.ndarray:
    """
    Copy a CUDA `Tensor` to a host NumPy array (D2H) with matching dtype/shape.
    """
    _ensure_cuda_tensor(t, "tensor")
    dt = np.dtype(t.dtype)
    out = np.empty(t.shape, dtype=dt)

    dev_ptr = _get_tensor_dev_ptr(t)
    nbytes = int(out.nbytes)

    if nbytes > 0 and dev_ptr == 0:
        raise RuntimeError("CUDA tensor has no device pointer (allocation missing)")

    if nbytes > 0:
        cuda_set_device(lib, int(device_index))
        cudaMemcpyDtoH(lib, out, int(dev_ptr), nbytes)
        cuda_synchronize(lib)

    return out


def _from_numpy_to_cuda_tensor(
    arr: np.ndarray, *, device, requires_grad: bool, lib, device_index: int
) -> Tensor:
    """
    Allocate device memory, H2D copy from NumPy, and wrap as a CUDA `Tensor`.
    """
    arr_c = np.ascontiguousarray(arr)
    dt = np.dtype(arr_c.dtype)
    nbytes = int(arr_c.nbytes)

    cuda_set_device(lib, int(device_index))

    if nbytes == 0:
        return Tensor._from_devptr(
            dev_ptr=0,
            shape=tuple(int(d) for d in arr_c.shape),
            device=device,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dt,
        )

    dev_ptr = int(cuda_malloc(lib, nbytes))
    if dev_ptr == 0:
        raise RuntimeError("cuda_malloc returned null device pointer (0)")

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=dev_ptr,
        nbytes=nbytes,
        dtype=dt,
    )

    try:
        cudaMemcpyHtoD(lib, int(dev_ptr), arr_c, nbytes)
        cuda_synchronize(lib)
    except Exception:
        try:
            cuda_free(lib, int(dev_ptr))
        except Exception:
            pass
        raise

    return Tensor._from_storage(
        storage,
        shape=tuple(int(d) for d in arr_c.shape),
        device=device,
        requires_grad=requires_grad,
        ctx=None,
        dtype=dt,
    )


# -----------------------------------------------------------------------------
# Public Tensor-facing CUDA transpose-conv boundary
# -----------------------------------------------------------------------------
def conv2d_transpose_forward_cuda_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    out_requires_grad: bool,
    device_index: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Run CUDA ConvTranspose2D forward using `Tensor` inputs and return a CUDA `Tensor` output.

    Parameters
    ----------
    x : Tensor
        CUDA input tensor, shape (N, C_in, H_in, W_in), NCHW.
    w : Tensor
        CUDA weight tensor, shape (C_in, C_out, K_h, K_w), IOHW (transpose-conv layout).
    b : Optional[Tensor]
        Optional CUDA bias tensor, shape (C_out,).
    stride, padding, output_padding : (int, int)
        Transpose-conv hyperparameters.
    out_requires_grad : bool
        `requires_grad` for the returned output tensor.
    device_index : int
        CUDA device index to execute on.
    sync : bool
        Forwarded to the underlying ops wrapper.

    Returns
    -------
    Tensor
        CUDA output tensor, shape (N, C_out, H_out, W_out).
    """
    _ensure_cuda_tensor(x, "x")
    _ensure_cuda_tensor(w, "w")
    if b is not None:
        _ensure_cuda_tensor(b, "b")

    # Dtype consistency (match conv2d_cuda_ext behavior)
    x_dt = np.dtype(x.dtype)
    w_dt = np.dtype(w.dtype)
    if x_dt != w_dt:
        raise TypeError(f"conv2d_transpose dtype mismatch: x={x_dt} vs w={w_dt}")
    if b is not None:
        b_dt = np.dtype(b.dtype)
        if b_dt != x_dt:
            raise TypeError(f"conv2d_transpose dtype mismatch: b={b_dt} vs x={x_dt}")

    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device_index))

    # D2H for NumPy-facing ops wrapper
    x_np = _to_numpy_via_d2h(x, lib=lib, device_index=device_index)
    w_np = _to_numpy_via_d2h(w, lib=lib, device_index=device_index)
    b_np = (
        None if b is None else _to_numpy_via_d2h(b, lib=lib, device_index=device_index)
    )

    y_np = _conv2d_t_fwd_ops(
        lib,
        x=x_np,
        w=w_np,
        b=b_np,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dtype=np.dtype(x_np.dtype),
        sync=sync,
        device_index=device_index,
    )

    out = _from_numpy_to_cuda_tensor(
        y_np,
        device=x.device,
        requires_grad=out_requires_grad,
        lib=lib,
        device_index=device_index,
    )
    return out


def conv2d_transpose_backward_cuda_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    grad_out: Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    device_index: int = 0,
    sync: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Run CUDA ConvTranspose2D backward using `Tensor` inputs and return CUDA gradients.

    Returns
    -------
    (grad_x, grad_w, grad_b)
        grad_x : Tensor
            CUDA Tensor with shape matching `x`.
        grad_w : Tensor
            CUDA Tensor with shape matching `w` (IOHW).
        grad_b : Optional[Tensor]
            CUDA Tensor with shape (C_out,) if `b` is not None, else None.
    """
    _ensure_cuda_tensor(x, "x")
    _ensure_cuda_tensor(w, "w")
    _ensure_cuda_tensor(grad_out, "grad_out")
    if b is not None:
        _ensure_cuda_tensor(b, "b")

    # Dtype consistency
    x_dt = np.dtype(x.dtype)
    w_dt = np.dtype(w.dtype)
    go_dt = np.dtype(grad_out.dtype)
    if x_dt != w_dt or x_dt != go_dt:
        raise TypeError(
            f"conv2d_transpose dtype mismatch: x={x_dt}, w={w_dt}, grad_out={go_dt}"
        )
    if b is not None:
        b_dt = np.dtype(b.dtype)
        if b_dt != x_dt:
            raise TypeError(f"conv2d_transpose dtype mismatch: b={b_dt} vs x={x_dt}")

    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device_index))

    # D2H for NumPy-facing ops wrapper
    x_np = _to_numpy_via_d2h(x, lib=lib, device_index=device_index)
    w_np = _to_numpy_via_d2h(w, lib=lib, device_index=device_index)
    b_np = (
        None if b is None else _to_numpy_via_d2h(b, lib=lib, device_index=device_index)
    )
    go_np = _to_numpy_via_d2h(grad_out, lib=lib, device_index=device_index)

    gx_np, gw_np, gb_np = _conv2d_t_bwd_ops(
        lib,
        x=x_np,
        w=w_np,
        b=b_np,
        grad_out=go_np,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dtype=np.dtype(x_np.dtype),
        sync=sync,
        device_index=device_index,
    )

    grad_x = _from_numpy_to_cuda_tensor(
        gx_np,
        device=x.device,
        requires_grad=False,
        lib=lib,
        device_index=device_index,
    )
    grad_w = _from_numpy_to_cuda_tensor(
        gw_np,
        device=w.device,
        requires_grad=False,
        lib=lib,
        device_index=device_index,
    )

    grad_b = None
    if gb_np is not None:
        grad_b = _from_numpy_to_cuda_tensor(
            gb_np,
            device=b.device if b is not None else x.device,
            requires_grad=False,
            lib=lib,
            device_index=device_index,
        )

    return grad_x, grad_w, grad_b


class _Conv2dTransposeCudaTensorBoundaryExports:
    """
    Documentation-only summary of the public boundary-layer entry points.
    """

    conv2d_transpose_forward_cuda_tensor = conv2d_transpose_forward_cuda_tensor
    conv2d_transpose_backward_cuda_tensor = conv2d_transpose_backward_cuda_tensor


__all__ = [
    "conv2d_transpose_forward_cuda_tensor",
    "conv2d_transpose_backward_cuda_tensor",
]
