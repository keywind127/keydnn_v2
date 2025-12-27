"""
CUDA stack primitives with Tensor boundaries (device-pointer based).

This module implements the CUDA backend for `Tensor.stack` by operating purely
on device pointers (no host round-trips such as `to_numpy()`), and returning
a CUDA `Tensor` created from an output device pointer.

High-level flow
---------------
Forward:
- Validate that all inputs are CUDA tensors with identical shape, dtype, and
  device placement.
- Normalize `axis` (supporting negative axes for "insert new dim").
- Compute `(pre, post)` products used by the CUDA kernel launch.
- Allocate the output buffer on device.
- Call the CUDA forward kernel via a ctypes wrapper. The wrapper additionally
  allocates a temporary device-side pointer-array of input pointers.
- Construct and return the output tensor via `Tensor._from_devptr`.

Backward:
- Validate that `grad_out` is a CUDA tensor of the expected stacked shape.
- Normalize `axis` and compute `(pre, post)`.
- Allocate `K` device buffers for per-input gradients.
- Call the CUDA backward kernel via a ctypes wrapper (overwrite semantics).
- Return `K` gradient tensors via `Tensor._from_devptr`.

Notes
-----
- This module assumes input tensors are contiguous CUDA tensors.
- The underlying backward kernel writes into `dx` buffers without accumulation
  (overwrite semantics). If you need accumulation, do it outside this module.
- Temporary device allocations made by the ctypes wrappers (pointer arrays) are
  explicitly freed here.

Performance notes
-----------------
This module is written to avoid implicit global synchronizations.

- Both forward and backward calls accept `sync` (default False). When False, the
  CUDA work is enqueued on the current stream and may complete asynchronously.
- For benchmarking correctness or latency-sensitive boundaries, pass `sync=True`
  (or synchronize in your benchmark harness around timing windows).
"""

from __future__ import annotations

from typing import Sequence, List, Tuple
import numpy as np

from ..tensor._tensor import Tensor
from .pool2d_cuda import _load_cuda_lib, cuda_set_device, cuda_malloc, cuda_free
from ..native_cuda.python.stack_ctypes import (
    stack_forward_cuda as _stack_fwd,
    stack_backward_cuda as _stack_bwd,
)

DevPtr = int


def _require_cuda(x: Tensor, name: str) -> None:
    """
    Assert that a tensor is placed on a CUDA device.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Human-readable name used in error messages.

    Raises
    ------
    TypeError
        If `x.device.is_cuda()` is False.
    """
    if not x.device.is_cuda():
        raise TypeError(f"{name} must be a CUDA Tensor; got device={x.device}")


def _require_f32_f64(x: Tensor, name: str) -> np.dtype:
    """
    Assert that a tensor dtype is float32 or float64.

    Parameters
    ----------
    x : Tensor
        Tensor to validate.
    name : str
        Human-readable name used in error messages.

    Returns
    -------
    np.dtype
        The tensor dtype (np.float32 or np.float64).

    Raises
    ------
    TypeError
        If dtype is not float32/float64.
    """
    dt = x.dtype
    if dt not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32/float64; got dtype={dt}")
    return dt


def _normalize_axis(axis: int, ndim: int) -> int:
    """
    Normalize a stack axis into the inclusive range [0, ndim].

    For stack, the new dimension is inserted, so valid axes are:
        axis ∈ [-ndim-1, ..., ndim]

    Parameters
    ----------
    axis : int
        User-specified axis (may be negative).
    ndim : int
        Number of dimensions of the input tensor(s).

    Returns
    -------
    int
        Normalized axis in [0, ndim].

    Raises
    ------
    ValueError
        If the normalized axis is out of bounds for stack insertion.
    """
    # axis is in [-ndim-1, ndim]
    if axis < 0:
        axis = axis + (ndim + 1)
    if axis < 0 or axis > ndim:
        raise ValueError(f"axis {axis} out of bounds for stack with input ndim {ndim}")
    return int(axis)


def _prod(xs: Sequence[int]) -> int:
    """
    Compute the product of an integer sequence.

    Parameters
    ----------
    xs : Sequence[int]
        Integer values to multiply.

    Returns
    -------
    int
        Product of all values in `xs` (1 for an empty sequence).
    """
    out = 1
    for v in xs:
        out *= int(v)
    return int(out)


def _pre_post(in_shape: Tuple[int, ...], axis: int) -> Tuple[int, int]:
    """
    Compute `(pre, post)` products used by the CUDA stack kernels.

    For an input shape `in_shape` and an insertion axis `axis` (normalized),
    the flattened layout can be viewed as:

        [pre, post] where:
            pre  = prod(in_shape[:axis])
            post = prod(in_shape[axis:])

    Parameters
    ----------
    in_shape : tuple[int, ...]
        Input tensor shape (without the stacked dimension).
    axis : int
        Axis at which stack inserts the new dimension. May be negative; will be
        normalized relative to `len(in_shape)`.

    Returns
    -------
    (int, int)
        The pair `(pre, post)`.

    Notes
    -----
    This function normalizes `axis` internally, so callers may pass a negative
    axis.
    """
    ndim = len(in_shape)
    axis = _normalize_axis(axis, ndim)
    pre = _prod(in_shape[:axis])
    post = _prod(in_shape[axis:])
    return int(pre), int(post)


def stack_forward(
    tensors: Sequence[Tensor],
    *,
    axis: int = 0,
    device: int = 0,
    sync: bool = False,
    debug_verify_ptrs: bool = False,
) -> Tensor:
    """
    Stack CUDA tensors along a new axis (forward pass).

    This function stacks `K = len(tensors)` inputs that all share:
    - CUDA device placement
    - identical shape
    - identical dtype (float32 or float64)

    The stacking operation inserts a new dimension of size `K` at `axis`:

        out.shape = in.shape[:axis] + (K,) + in.shape[axis:]

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Non-empty sequence of CUDA tensors with identical shape/dtype/device.
    axis : int, optional
        Axis at which the new dimension is inserted. Supports negative axes in
        the range [-ndim-1, ndim]. Defaults to 0.
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, synchronizes after kernel launch in the ctypes layer. Defaults
        to False for performance (async execution).
    debug_verify_ptrs : bool, optional
        If True, ctypes layer may read back and validate device pointer arrays.
        Defaults to False for performance.

    Returns
    -------
    Tensor
        A CUDA tensor whose device buffer is newly allocated and filled by the
        CUDA stack kernel.

    Raises
    ------
    ValueError
        If `tensors` is empty, shapes mismatch, dtype mismatch, device mismatch,
        or `axis` is out of bounds.
    TypeError
        If any tensor is not CUDA or has unsupported dtype.

    Resource management
    -------------------
    - Allocates `y_dev` for the output.
    - The ctypes wrapper allocates a temporary device pointer-array for the
      input pointers and returns its device address; this function frees it in
      `finally`.
    - On exceptions, `y_dev` is freed before re-raising.
    """
    if len(tensors) == 0:
        raise ValueError("stack_forward requires a non-empty sequence")

    first = tensors[0]
    _require_cuda(first, "tensors[0]")
    dt = _require_f32_f64(first, "tensors[0]")

    dev = first.device
    in_shape = tuple(first.shape)

    for i, t in enumerate(tensors):
        _require_cuda(t, f"tensors[{i}]")
        _require_f32_f64(t, f"tensors[{i}]")
        if str(t.device) != str(dev):
            raise ValueError(
                f"All tensors must be on the same device; tensors[0]={dev!r}, tensors[{i}]={t.device!r}"
            )
        if tuple(t.shape) != in_shape:
            raise ValueError(
                f"All tensors must have the same shape; expected {in_shape}, got {tuple(t.shape)} at index {i}"
            )
        if t.dtype != dt:
            raise ValueError(
                f"All tensors must have the same dtype; expected {dt}, got {t.dtype} at index {i}"
            )

    K = int(len(tensors))
    axis_n = _normalize_axis(axis, len(in_shape))
    out_shape = tuple(in_shape[:axis_n]) + (K,) + tuple(in_shape[axis_n:])
    pre, post = _pre_post(in_shape, axis_n)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # allocate y
    nbytes_y = int(_prod(out_shape) * np.dtype(dt).itemsize)
    y_dev = cuda_malloc(lib, nbytes_y)

    xs_ptrs_dev = 0
    try:
        xs_dev_ptrs = [int(t.data) for t in tensors]

        # The ctypes wrapper allocates the pointer-array on device and returns it.
        # IMPORTANT: default sync=False to avoid sync-storms in callers like Linear(bias=True).
        xs_ptrs_dev = _stack_fwd(
            lib,
            xs_dev_ptrs=xs_dev_ptrs,
            y_dev=int(y_dev),
            pre=int(pre),
            post=int(post),
            dtype=dt,
            sync=bool(sync),
            debug_verify_ptrs=bool(debug_verify_ptrs),
        )

        out = Tensor._from_devptr(
            int(y_dev),
            shape=out_shape,
            dtype=dt,
            device=dev,
            requires_grad=False,
        )
        return out

    except Exception:
        cuda_free(lib, y_dev)
        raise

    finally:
        # free temporary device pointer array
        if xs_ptrs_dev:
            cuda_free(lib, int(xs_ptrs_dev))


def stack_backward(
    grad_out: Tensor,
    *,
    x_shape: Tuple[int, ...],
    axis: int,
    K: int,
    device: int = 0,
    sync: bool = False,
    debug_verify_ptrs: bool = False,
) -> List[Tensor]:
    """
    Compute input gradients for CUDA stack (backward pass).

    Given `grad_out` of shape:

        x_shape[:axis] + (K,) + x_shape[axis:]

    this function produces `K` gradient tensors, each of shape `x_shape`,
    corresponding to gradients w.r.t. each input tensor that was stacked.

    Parameters
    ----------
    grad_out : Tensor
        CUDA tensor of stacked gradients.
    x_shape : tuple[int, ...]
        Original per-input shape (before stacking).
    axis : int
        Axis at which the new dimension was inserted during forward.
        Supports negative axes in the range [-ndim-1, ndim].
    K : int
        Number of stacked tensors (size of the inserted dimension).
    device : int, optional
        CUDA device ordinal passed to `cuda_set_device`. Defaults to 0.
    sync : bool, optional
        If True, synchronizes after kernel launch in the ctypes layer. Defaults
        to False for performance (async execution).
    debug_verify_ptrs : bool, optional
        If True, ctypes layer may read back and validate device pointer arrays.
        Defaults to False for performance.

    Returns
    -------
    list[Tensor]
        List of length `K`, each a CUDA tensor of shape `x_shape`.

    Raises
    ------
    ValueError
        If `grad_out.shape` does not match the expected stacked shape or if
        `axis` is out of bounds.
    TypeError
        If `grad_out` is not CUDA or has unsupported dtype.

    Notes
    -----
    The underlying kernel overwrites each `dx` buffer (no accumulation). If your
    autograd requires accumulation, accumulate at a higher level.
    """
    _require_cuda(grad_out, "grad_out")
    dt = _require_f32_f64(grad_out, "grad_out")

    x_shape = tuple(int(d) for d in x_shape)
    ndim = len(x_shape)
    axis_n = _normalize_axis(axis, ndim)

    expected_out_shape = tuple(x_shape[:axis_n]) + (int(K),) + tuple(x_shape[axis_n:])
    if tuple(grad_out.shape) != expected_out_shape:
        raise ValueError(
            f"grad_out shape mismatch: expected {expected_out_shape}, got {tuple(grad_out.shape)}"
        )

    pre, post = _pre_post(x_shape, axis_n)

    lib = _load_cuda_lib()
    cuda_set_device(lib, int(device))

    # allocate each dx
    nbytes_dx = int(_prod(x_shape) * np.dtype(dt).itemsize)
    dx_devs: List[DevPtr] = []
    try:
        for _ in range(int(K)):
            dx_devs.append(cuda_malloc(lib, nbytes_dx))

        # IMPORTANT: default sync=False to avoid sync-storms.
        dxs_ptrs_dev = _stack_bwd(
            lib,
            dy_dev=int(grad_out.data),
            dxs_dev_ptrs=[int(p) for p in dx_devs],
            pre=int(pre),
            post=int(post),
            dtype=dt,
            sync=bool(sync),
            debug_verify_ptrs=bool(debug_verify_ptrs),
        )

        grads: List[Tensor] = [
            Tensor._from_devptr(
                int(dx_devs[i]),
                shape=x_shape,
                dtype=dt,
                device=grad_out.device,
                requires_grad=False,
            )
            for i in range(int(K))
        ]
        return grads

    except Exception:
        # if failure, free allocated dx buffers
        for p in dx_devs:
            cuda_free(lib, int(p))
        raise

    finally:
        # free temporary pointer-array allocation (if created)
        if "dxs_ptrs_dev" in locals() and dxs_ptrs_dev:
            cuda_free(lib, int(dxs_ptrs_dev))
