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
    native CUDA extension wrappers.

    Fast-path:
    - If `other` is a Python scalar, dispatch to `add_scalar(self, alpha)` to
      avoid scalar-lifting into a full device tensor.

    Tensor-path:
    - If `other` is a CUDA tensor, dispatch to `add(self, other)`.

    See original docstring for full semantics/notes.
    """
    import numpy as np

    # Prefer the tensor's device index if available; otherwise default 0
    device_index = int(getattr(self.device, "index", 0) or 0)

    # -----------------------------
    # CUDA scalar fast-path
    # -----------------------------
    if isinstance(other, (int, float)):
        # dtype gate (matches CUDA kernels / ext wrapper constraints)
        dt = np.dtype(self.dtype)
        if dt not in (np.float32, np.float64):
            raise TypeError(f"add scalar requires float32/float64, got dtype={dt}")

        req = bool(getattr(self, "requires_grad", False))

        from ....ops.tensor_arithmetic_cuda_ext import add_scalar as _cuda_add_scalar

        out = _cuda_add_scalar(self, float(other), device=device_index)
        out.requires_grad = bool(req)

        if req:
            # y = x + alpha => dy/dx = 1, scalar has no grad
            ctx = Context(
                parents=(self,),
                backward_fn=lambda grad_out: (grad_out,),
            )
            out._set_ctx(ctx)

        return out

    # -----------------------------
    # CUDA tensor path (existing behavior)
    # -----------------------------
    other_t = self._as_tensor_like(other, self)

    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        # dtype must match for our CUDA kernels
        if np.dtype(self.dtype) != np.dtype(other_t.dtype):
            raise TypeError(
                f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
            )

        req = self._result_requires_grad(self, other_t)

        from ....ops.tensor_arithmetic_cuda_ext import add as _cuda_add

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
