"""
Device-specific implementations of Tensor addition via control-path dispatch.

This module registers CPU and CUDA implementations of elementwise addition
for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public operator entrypoint is `TensorMixinArithmetic.__add__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations preserve the same high-level semantics:
- elementwise addition with no broadcasting (shape must match),
- scalar inputs are promoted to tensor-like operands,
- autograd support via a Context storing a backward function.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinArithmetic as TMA
from typing import Union


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMA, TMA.__add__, Device("cuda:0"))
def tensor_add_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise tensor addition.

    This implementation performs elementwise addition on CUDA tensors using
    a native CUDA extension wrapper. It validates shape compatibility and
    dtype equality (required by the current CUDA kernels), then delegates the
    actual computation to the CUDA backend.

    Parameters
    ----------
    self : ITensor
        Left-hand operand tensor. Must reside on a CUDA device.
    other : Union[ITensor, Number]
        Right-hand operand. If a scalar, it is promoted to a tensor compatible
        with `self` (same shape/device) via `_as_tensor_like`.

    Returns
    -------
    ITensor
        A CUDA tensor containing the elementwise sum.

    Raises
    ------
    TypeError
        If `self` and `other` have mismatched dtypes (CUDA kernels require
        identical dtypes).
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Autograd backward for addition is:
        d(a + b)/da = 1, d(a + b)/db = 1
      so the upstream gradient is returned for both parents.
    - The output device index is taken from `self.device.index` when available.
    """
    import numpy as np

    other_t = self._as_tensor_like(other, self)

    # -----------------------------
    # CUDA path (device-pointer elementwise)
    # -----------------------------
    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        # dtype must match for our CUDA kernels
        if np.dtype(self.dtype) != np.dtype(other_t.dtype):
            raise TypeError(
                f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
            )

        req = self._result_requires_grad(self, other_t)

        # Use your CUDA ext wrapper (allocates output and returns CUDA Tensor)
        from ....ops.tensor_arithmetic_cuda_ext import add as _cuda_add

        # Prefer the tensor's device index if available; otherwise default 0
        device_index = int(getattr(self.device, "index", 0) or 0)

        out = _cuda_add(self, other_t, device=device_index)
        out.requires_grad = bool(req)  # ensure flag matches CPU behavior

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out, grad_out),
            )
            out._set_ctx(ctx)

        return out

    self._raise_device_not_supported("add")


@tensor_control_path_manager(TMA, TMA.__add__, Device("cpu"))
def tensor_add_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise tensor addition.

    This implementation performs elementwise addition by converting both
    operands to NumPy arrays, adding them, and copying the result into a new
    tensor on the CPU.

    Parameters
    ----------
    self : ITensor
        Left-hand operand tensor residing on the CPU.
    other : Union[ITensor, Number]
        Right-hand operand. If a scalar, it is promoted to a tensor compatible
        with `self` (same shape/device) via `_as_tensor_like`.

    Returns
    -------
    ITensor
        A CPU tensor containing the elementwise sum.

    Raises
    ------
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Autograd backward for addition returns the upstream gradient for both
      parents, consistent with elementwise addition semantics.
    """
    Tensor = type(self)
    other_t = self._as_tensor_like(other, self)

    # -----------------------------
    # CPU path (existing)
    # -----------------------------
    if other_t.device.is_cpu():
        self._binary_op_shape_check(self, other_t)

        req = self._result_requires_grad(self, other_t)
        out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
        out.copy_from_numpy(self.to_numpy() + other_t.to_numpy())

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out, grad_out),
            )
            out._set_ctx(ctx)
        return out

    self._raise_device_not_supported("add")
