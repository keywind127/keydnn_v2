"""
Device-specific implementations of Tensor subtraction via control-path dispatch.

This module registers CPU and CUDA implementations of elementwise subtraction
for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public operator entrypoint is `TensorMixinArithmetic.__sub__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations preserve the same high-level semantics:
- elementwise subtraction with no broadcasting (shape must match),
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


@tensor_control_path_manager(TMA, TMA.__sub__, Device("cuda:0"))
def tensor_sub_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise tensor subtraction.

    (Docstring unchanged; implementation includes a scalar fast-path to avoid
    scalar-lifting into a full CUDA tensor.)
    """
    import numpy as np

    # Prefer the tensor's device index if available; otherwise default 0
    device_index = int(getattr(self.device, "index", 0) or 0)

    # -----------------------------
    # CUDA scalar fast-path: y = self - alpha
    # -----------------------------
    if isinstance(other, (int, float)):
        dt = np.dtype(self.dtype)
        if dt not in (np.float32, np.float64):
            raise TypeError(
                f"CUDA sub scalar supports float32/float64 only; got dtype={dt}"
            )

        alpha = float(other)
        req = bool(getattr(self, "requires_grad", False))

        from ....ops.tensor_arithmetic_cuda_ext import sub_scalar as _cuda_sub_scalar

        out = _cuda_sub_scalar(self, alpha, device=device_index)
        out.requires_grad = bool(req)

        if req:
            # y = a - c  => dy/da = 1 ; scalar has no grad
            ctx = Context(
                parents=(self,),
                backward_fn=lambda grad_out: (grad_out,),
            )
            out._set_ctx(ctx)

        return out

    # -----------------------------
    # CUDA tensor path (device-pointer elementwise)
    # -----------------------------
    other_t = self._as_tensor_like(other, self)

    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        if np.dtype(self.dtype) != np.dtype(other_t.dtype):
            raise TypeError(
                f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
            )

        req = self._result_requires_grad(self, other_t)

        from ....ops.tensor_arithmetic_cuda_ext import sub as _cuda_sub

        out = _cuda_sub(self, other_t, device=device_index)
        out.requires_grad = bool(req)

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out, -grad_out),
            )
            out._set_ctx(ctx)

        return out

    self._raise_device_not_supported("sub")


@tensor_control_path_manager(TMA, TMA.__sub__, Device("cpu"))
def tensor_sub_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise tensor subtraction.

    This implementation performs elementwise subtraction by converting both
    operands to NumPy arrays, subtracting them, and copying the result into a
    new tensor on the CPU.

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
        A CPU tensor containing the elementwise result of ``self - other``.

    Raises
    ------
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Backward propagation follows subtraction rules:

        * grad_a = grad_out
        * grad_b = -grad_out
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
        out.copy_from_numpy(self.to_numpy() - other_t.to_numpy())

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out, -grad_out),
            )
            out._set_ctx(ctx)
        return out
    self._raise_device_not_supported("sub")
