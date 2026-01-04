"""
infrastructure/ops/conv2d_cuda_ext.py

Conv2D CUDA backend boundary helpers.

This module defines *boundary-layer* functions that bridge KeyDNN infrastructure
`Tensor` objects to CUDA conv2d ops wrappers and back.

Purpose
-------
The CUDA conv2d ops wrapper (`conv2d_forward_cuda`, `conv2d_backward_cuda`) is
currently NumPy-facing and manages temporary CUDA buffers internally. To keep
higher-level autograd / operator code NumPy-free, all Tensor <-> NumPy and CUDA
memcpy conversion is localized to this module.

Design notes
------------
- These helpers perform *no* shape inference beyond what is necessary to move
  bytes correctly. They:
  1) bring CUDA `Tensor` inputs to host NumPy arrays (D2H),
  2) call the CUDA ops wrapper (which pads on CPU and launches CUDA kernels),
  3) wrap returned arrays into new CUDA `Tensor` objects (H2D).
- Output tensors are created with explicit `requires_grad` flags supplied by the
  caller (typically the differentiable conv2d operation).

Backend constraints (current)
-----------------------------
- This boundary currently performs D2H/H2D copies because the conv2d CUDA ops
  wrapper is NumPy-facing.
- A future "true CUDA boundary" can avoid host copies by calling the ctypes
  wrapper directly with device pointers (x_dev/w_dev/b_dev/y_dev), once your
  Tensor exposes/stabilizes raw device pointers for conv2d without padding on CPU.
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

from .conv2d_cuda import (
    conv2d_forward_cuda as _conv2d_forward_ops,
    conv2d_backward_cuda as _conv2d_backward_ops,
)


def _ensure_cuda_tensor(t: Tensor, name: str) -> None:
    """
    Validate that a `Tensor` is CUDA-backed.

    Parameters
    ----------
    t : Tensor
        Tensor to validate.
    name : str
        Human-readable argument name used in error messages.

    Raises
    ------
    TypeError
        If `t` is not a CUDA tensor.
    """
    if not t.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA tensor, got device={t.device}")


def _pair(v: Tuple[int, int] | int) -> Tuple[int, int]:
    """
    Normalize an int-or-pair into a (h, w) tuple of ints.

    Parameters
    ----------
    v : int | (int, int)
        Scalar or pair value.

    Returns
    -------
    (int, int)
        A 2-tuple of ints.
    """
    return v if isinstance(v, tuple) else (int(v), int(v))


def _get_tensor_dev_ptr(t: Tensor) -> int:
    """
    Best-effort extraction of the underlying CUDA device pointer from a `Tensor`.

    In this repository, CUDA tensors created via `Tensor._from_devptr(...)` store
    the raw device pointer in `t.data` (as an int-like value). This helper
    attempts to read that field first, then checks several common fallback field
    names used by older/internal layouts.

    Parameters
    ----------
    t : Tensor
        CUDA tensor to inspect.

    Returns
    -------
    int
        Device pointer address as an integer. Returns 0 if no non-zero pointer
        can be found.

    Notes
    -----
    Returning 0 is allowed for empty tensors (nbytes == 0). Callers should
    validate pointer presence when a non-empty buffer is required.
    """
    # Primary: `data` field used by Tensor._from_devptr in your tests
    if hasattr(t, "data"):
        try:
            v = int(getattr(t, "data"))
            if v != 0:
                return v
        except Exception:
            pass

    # Fallbacks (older/alternate layouts)
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

    Parameters
    ----------
    t : Tensor
        CUDA tensor to copy from.
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library handle.
    device_index : int
        CUDA device index to select before performing the copy.

    Returns
    -------
    np.ndarray
        Host array with the same shape and dtype as `t`.

    Raises
    ------
    TypeError
        If `t` is not a CUDA tensor.
    RuntimeError
        If `t` has non-zero size but no valid device pointer can be found.
    """
    _ensure_cuda_tensor(t, "tensor")
    dt = np.dtype(t.dtype)
    out = np.empty(t.shape, dtype=dt)

    dev_ptr = _get_tensor_dev_ptr(t)

    # If tensor is empty, allow dev_ptr==0
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

    This function explicitly uses `cuda_malloc` and `cudaMemcpyHtoD` and then
    wraps the resulting device pointer using `Tensor._from_devptr(...)`.

    Parameters
    ----------
    arr : np.ndarray
        Host array to transfer to device. Converted to contiguous form.
    device : Device
        Target KeyDNN device object (typically `x.device`).
    requires_grad : bool
        `requires_grad` flag for the resulting Tensor.
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library handle.
    device_index : int
        CUDA device index to select before allocating/copying.

    Returns
    -------
    Tensor
        CUDA Tensor wrapping the newly allocated device buffer.

    Raises
    ------
    RuntimeError
        If allocation fails or returns a null device pointer.

    Important
    ---------
    Do NOT rely on `Tensor._ensure_cuda_alloc()` here, because:
    - its internal devptr field name may differ across implementations
    - it may use a different cached CUDA lib handle / device state
    - tests construct CUDA tensors via `Tensor._from_devptr`, so we mirror that contract
    """
    arr_c = np.ascontiguousarray(arr)
    dt = np.dtype(arr_c.dtype)

    nbytes = int(arr_c.nbytes)

    # Ensure correct device before allocating/copying
    cuda_set_device(lib, int(device_index))

    if nbytes == 0:
        # Allow empty tensors to have dev_ptr == 0
        return Tensor._from_devptr(
            dev_ptr=0,
            shape=tuple(int(d) for d in arr_c.shape),
            device=device,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dt,
        )

    dev_ptr = int(cuda_malloc(lib, nbytes))

    from ..tensor._cuda_storage import _CudaStorage

    storage = _CudaStorage(
        lib=lib,
        device_index=device_index,
        dev_ptr=dev_ptr,
        nbytes=nbytes,
        dtype=dt,
    )
    if dev_ptr == 0:
        raise RuntimeError("cuda_malloc returned null device pointer (0)")

    try:
        cudaMemcpyHtoD(lib, int(dev_ptr), arr_c, nbytes)
        cuda_synchronize(lib)
    except Exception:
        # Avoid leaking device memory if copy fails
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


def conv2d_forward_cuda_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    out_requires_grad: bool,
    device_index: int = 0,
    sync: bool = True,
) -> Tensor:
    """
    Run CUDA conv2d forward using `Tensor` inputs and return a CUDA `Tensor` output.

    This boundary helper:
    1) validates CUDA devices and dtype consistency,
    2) copies `Tensor` inputs from device to host NumPy (D2H),
    3) calls the NumPy-facing CUDA ops wrapper,
    4) copies the resulting NumPy output back to device (H2D) and wraps it as a
       new CUDA `Tensor`.

    Parameters
    ----------
    x : Tensor
        CUDA input tensor of shape (N, C_in, H, W), NCHW.
    w : Tensor
        CUDA weight tensor of shape (C_out, C_in, K_h, K_w), OIHW.
    b : Optional[Tensor]
        Optional CUDA bias tensor of shape (C_out,). If None, forward runs without bias.
    stride : (int, int)
        Stride along (height, width).
    padding : (int, int)
        Zero-padding along (height, width). Padding is applied symmetrically.
    out_requires_grad : bool
        `requires_grad` flag for the returned output tensor.
    device_index : int, default 0
        CUDA device index to execute on.
    sync : bool, default True
        Whether to synchronize in the underlying ops wrapper.

    Returns
    -------
    Tensor
        CUDA output tensor of shape (N, C_out, H_out, W_out).

    Raises
    ------
    TypeError
        If any input is not CUDA or if dtypes are inconsistent across x/w/b.
    RuntimeError
        If an input has data but lacks a valid device pointer.
    """
    _ensure_cuda_tensor(x, "x")
    _ensure_cuda_tensor(w, "w")
    if b is not None:
        _ensure_cuda_tensor(b, "b")

    # Enforce dtype consistency across inputs (tests expect TypeError)
    x_dt = np.dtype(x.dtype)
    w_dt = np.dtype(w.dtype)
    if x_dt != w_dt:
        raise TypeError(f"conv2d dtype mismatch: x={x_dt} vs w={w_dt}")
    if b is not None:
        b_dt = np.dtype(b.dtype)
        if b_dt != x_dt:
            raise TypeError(f"conv2d dtype mismatch: b={b_dt} vs x={x_dt}")

    stride = _pair(stride)
    padding = _pair(padding)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device_index))

    # D2H: bring tensors to NumPy for ops wrapper
    x_np = _to_numpy_via_d2h(x, lib=lib, device_index=device_index)
    w_np = _to_numpy_via_d2h(w, lib=lib, device_index=device_index)
    b_np = (
        None if b is None else _to_numpy_via_d2h(b, lib=lib, device_index=device_index)
    )

    # Call ops wrapper (pads on CPU, launches CUDA kernels internally)
    y_np = _conv2d_forward_ops(
        lib,
        x=x_np,
        w=w_np,
        b=b_np,
        stride=stride,
        padding=padding,
        dtype=np.dtype(x_np.dtype),
        sync=sync,
        device_index=device_index,
    )

    # H2D: wrap result back into a CUDA Tensor
    out = _from_numpy_to_cuda_tensor(
        y_np,
        device=x.device,
        requires_grad=out_requires_grad,
        lib=lib,
        device_index=device_index,
    )
    return out


def conv2d_backward_cuda_tensor(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    grad_out: Tensor,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    device_index: int = 0,
    sync: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Run CUDA conv2d backward using `Tensor` inputs and return CUDA gradients.

    This boundary helper:
    1) validates CUDA devices and dtype consistency,
    2) copies `Tensor` inputs from device to host NumPy (D2H),
    3) calls the NumPy-facing CUDA ops wrapper to compute gradients,
    4) copies gradient arrays back to device (H2D) and wraps them as CUDA `Tensor`s.

    Bias gradient behavior
    ----------------------
    The ops wrapper computes `grad_b` on CPU via reduction over grad_out, matching
    the CPU conv2d semantics. If `b` is None, `grad_b` will be None.

    Parameters
    ----------
    x : Tensor
        CUDA input tensor from forward.
    w : Tensor
        CUDA weight tensor from forward.
    b : Optional[Tensor]
        Optional CUDA bias tensor from forward.
    grad_out : Tensor
        CUDA gradient dL/dY with shape (N, C_out, H_out, W_out), NCHW.
    stride : (int, int)
        Stride along (height, width).
    padding : (int, int)
        Zero-padding along (height, width).
    device_index : int, default 0
        CUDA device index to execute on.
    sync : bool, default True
        Whether to synchronize in the underlying ops wrapper.

    Returns
    -------
    (grad_x, grad_w, grad_b)
        grad_x : Tensor
            CUDA Tensor of shape (N, C_in, H, W).
        grad_w : Tensor
            CUDA Tensor of shape (C_out, C_in, K_h, K_w).
        grad_b : Optional[Tensor]
            CUDA Tensor of shape (C_out,) if `b` is not None, else None.

    Raises
    ------
    TypeError
        If any input is not CUDA or if dtypes are inconsistent across x/w/grad_out/b.
    RuntimeError
        If an input has data but lacks a valid device pointer.
    """
    _ensure_cuda_tensor(x, "x")
    _ensure_cuda_tensor(w, "w")
    _ensure_cuda_tensor(grad_out, "grad_out")
    if b is not None:
        _ensure_cuda_tensor(b, "b")

        # Enforce dtype consistency (tests expect TypeError)
    x_dt = np.dtype(x.dtype)
    w_dt = np.dtype(w.dtype)
    go_dt = np.dtype(grad_out.dtype)
    if x_dt != w_dt or x_dt != go_dt:
        raise TypeError(f"conv2d dtype mismatch: x={x_dt}, w={w_dt}, grad_out={go_dt}")
    if b is not None:
        b_dt = np.dtype(b.dtype)
        if b_dt != x_dt:
            raise TypeError(f"conv2d dtype mismatch: b={b_dt} vs x={x_dt}")

    stride = _pair(stride)
    padding = _pair(padding)

    lib = load_keydnn_cuda_native()
    cuda_set_device(lib, int(device_index))

    # D2H: bring tensors to NumPy for ops wrapper
    x_np = _to_numpy_via_d2h(x, lib=lib, device_index=device_index)
    w_np = _to_numpy_via_d2h(w, lib=lib, device_index=device_index)
    b_np = (
        None if b is None else _to_numpy_via_d2h(b, lib=lib, device_index=device_index)
    )
    go_np = _to_numpy_via_d2h(grad_out, lib=lib, device_index=device_index)

    gx_np, gw_np, gb_np = _conv2d_backward_ops(
        lib,
        x=x_np,
        w=w_np,
        b=b_np,
        grad_out=go_np,
        stride=stride,
        padding=padding,
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


class _Conv2dCudaTensorBoundaryExports:
    """
    Documentation-only summary of the public boundary-layer entry points.

    This class is not used at runtime. It exists to clarify which functions are
    intended as the external boundary surface for Tensor-based CUDA conv2d.

    Attributes
    ----------
    conv2d_forward_cuda_tensor : callable
        Tensor-facing CUDA conv2d forward helper.
    conv2d_backward_cuda_tensor : callable
        Tensor-facing CUDA conv2d backward helper.
    """

    conv2d_forward_cuda_tensor = conv2d_forward_cuda_tensor
    conv2d_backward_cuda_tensor = conv2d_backward_cuda_tensor


__all__ = [
    "conv2d_forward_cuda_tensor",
    "conv2d_backward_cuda_tensor",
]
