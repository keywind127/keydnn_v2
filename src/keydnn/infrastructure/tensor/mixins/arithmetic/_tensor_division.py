"""
Device-specific implementations of Tensor true division via control-path dispatch.

This module registers CPU and CUDA implementations of elementwise true division
for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public operator entrypoint is `TensorMixinArithmetic.__truediv__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations enforce elementwise semantics (no broadcasting) and
support scalar inputs by promoting them to tensor-like operands.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinArithmetic as TMA
from typing import Union


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMA, TMA.__truediv__, Device("cuda:0"))
def tensor_truediv_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise true division.

    (Docstring unchanged; implementation includes a scalar fast-path to avoid
    scalar-lifting into a full CUDA tensor.)
    """
    import numpy as np

    # Prefer the tensor's device index if available; otherwise default 0
    device_index = int(getattr(self.device, "index", 0) or 0)

    # -----------------------------
    # CUDA scalar fast-path: y = self / alpha
    # -----------------------------
    if isinstance(other, (int, float)):
        dt = np.dtype(self.dtype)
        if dt not in (np.float32, np.float64):
            raise TypeError(
                f"CUDA truediv scalar supports float32/float64 only; got dtype={dt}"
            )

        alpha = float(other)
        req = bool(getattr(self, "requires_grad", False))

        from ....ops.tensor_arithmetic_cuda_ext import div_scalar as _cuda_div_scalar

        out = _cuda_div_scalar(self, alpha, device=device_index)
        out.requires_grad = bool(req)

        if req:
            # y = a / c  => dy/da = 1/c ; scalar has no grad
            ctx = Context(
                parents=(self,),
                backward_fn=lambda grad_out: (grad_out / alpha,),
            )
            out._set_ctx(ctx)

        return out

    # ----------------------------
    # CUDA tensor path (device-pointer, no to_numpy)
    # ----------------------------
    other_t = self._as_tensor_like(other, self)

    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        # Enforce dtype policy consistent with CUDA kernels (f32/f64 only, same dtype)
        dt_a = np.dtype(self.dtype)
        dt_b = np.dtype(other_t.dtype)
        if dt_a not in (np.float32, np.float64) or dt_b not in (np.float32, np.float64):
            raise TypeError(
                f"CUDA truediv supports float32/float64 only; got self.dtype={dt_a}, other.dtype={dt_b}"
            )
        if dt_a != dt_b:
            raise TypeError(
                f"CUDA truediv requires matching dtypes; got {dt_a} vs {dt_b}"
            )

        # Require same CUDA device placement
        if self.device != other_t.device and str(self.device) != str(other_t.device):
            raise ValueError(
                f"device mismatch: self.device={self.device} vs other.device={other_t.device}"
            )

        from ....ops.tensor_arithmetic_cuda_ext import div as _cuda_div

        req = self._result_requires_grad(self, other_t)

        out = _cuda_div(self, other_t, device=device_index)
        out.requires_grad = bool(req)

        if req:
            # grad_a = grad_out / b
            # grad_b = -(grad_out * a) / (b*b)
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (
                    grad_out / other_t,
                    -(grad_out * self) / (other_t * other_t),
                ),
            )
            out._set_ctx(ctx)

        return out

    self._raise_device_not_supported("truediv")


@tensor_control_path_manager(TMA, TMA.__truediv__, Device("cpu"))
def tensor_truediv_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise true division.

    This implementation performs elementwise division by converting both
    operands to NumPy arrays, dividing them, and copying the result into a new
    tensor on the CPU. It mirrors the same mathematical behavior as the CUDA
    path, but uses NumPy for computation.

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
        A CPU tensor containing the elementwise result of ``self / other``.

    Raises
    ------
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - Backward propagation follows elementwise division rules:

        * grad_a = grad_out / b
        * grad_b = -(grad_out * a) / (b * b)

      The backward implementation uses existing Tensor operators, consistent
      with the CUDA path.
    """
    other_t = self._as_tensor_like(other, self)

    Tensor = type(self)

    # ----------------------------
    # CPU path (backward compatible)
    # ----------------------------
    if other_t.device.is_cpu():
        self._binary_op_shape_check(self, other_t)

        req = self._result_requires_grad(self, other_t)
        out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
        out.copy_from_numpy(self.to_numpy() / other_t.to_numpy())

        if req:
            ctx = Context(
                parents=(self, other_t),
                backward_fn=lambda grad_out: (
                    grad_out / other_t,
                    -(grad_out * self) / (other_t * other_t),
                ),
            )
            out._set_ctx(ctx)
        return out

    self._raise_device_not_supported("truediv")
