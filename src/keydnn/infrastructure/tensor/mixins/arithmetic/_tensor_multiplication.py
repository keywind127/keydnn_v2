"""
Device-specific implementations of Tensor multiplication via control-path dispatch.

This module registers CPU and CUDA implementations of elementwise multiplication
for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public operator entrypoint is `TensorMixinArithmetic.__mul__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations preserve the same high-level semantics:
- elementwise multiplication with no broadcasting (shape must match),
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


@tensor_control_path_manager(TMA, TMA.__mul__, Device("cuda:0"))
def tensor_mul_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise tensor multiplication.

    This implementation performs elementwise multiplication on CUDA tensors
    using a native CUDA extension wrapper. It validates shape compatibility
    (no broadcasting) and enforces dtype equality, consistent with the current
    CUDA arithmetic kernels.

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
        A CUDA tensor containing the elementwise result of ``self * other``.

    Raises
    ------
    TypeError
        If operand dtypes do not match.
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Backward propagation follows elementwise multiplication rules:

        * grad_a = grad_out * b
        * grad_b = grad_out * a

      The backward implementation uses existing Tensor operators; for CUDA
      tensors these operators are expected to route to CUDA-compatible paths.
    """
    Tensor = type(self)
    other_t = self._as_tensor_like(other, self)

    # -----------------------------
    # CUDA path
    # -----------------------------
    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        # Enforce same dtype semantics as other CUDA arithmetic ops ext
        import numpy as np

        dt_self = np.dtype(self.dtype)
        dt_other = np.dtype(other_t.dtype)
        if dt_self != dt_other:
            raise TypeError(
                f"dtype mismatch: self.dtype={dt_self} vs other.dtype={dt_other}"
            )

        from ....ops.mul_cuda_ext import mul_forward as _cuda_mul

        device_index = int(getattr(self.device, "index", 0) or 0)
        out = _cuda_mul(self, other_t, device=device_index, sync=True)

        # Autograd (same rule as CPU)
        req = self._result_requires_grad(self, other_t)
        out.requires_grad = bool(req)  # ensure flag matches graph needs

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out * other_t, grad_out * self),
            )
            out._set_ctx(ctx)

        return out

    self._raise_device_not_supported("mul")


@tensor_control_path_manager(TMA, TMA.__mul__, Device("cpu"))
def tensor_mul_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise tensor multiplication.

    This implementation performs elementwise multiplication by converting both
    operands to NumPy arrays, multiplying them, and copying the result into a
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
        A CPU tensor containing the elementwise result of ``self * other``.

    Raises
    ------
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Backward propagation follows elementwise multiplication rules:

        * grad_a = grad_out * b
        * grad_b = grad_out * a
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
        out.copy_from_numpy(self.to_numpy() * other_t.to_numpy())

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (grad_out * other_t, grad_out * self),
            )
            out._set_ctx(ctx)
        return out
    self._raise_device_not_supported("mul")
