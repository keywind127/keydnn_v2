"""
Device-specific implementations of Tensor less-than comparison via control-path dispatch.

This module registers CPU and CUDA implementations of the elementwise
less-than comparison operator (``__lt__``) for tensors. Implementations are
selected at runtime using the `tensor_control_path_manager`, which dispatches
based on the tensor's device.

The public operator entrypoint is `TensorMixinComparison.__lt__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Semantics
---------
- Performs an elementwise comparison with no broadcasting (shapes must match).
- Scalars are supported:
  - CPU: scalar is promoted to tensor-like via `_as_tensor_like`.
  - CUDA: dispatches to a scalar CUDA kernel variant without promoting the
    scalar to a tensor (to avoid temporary allocations).
- The result is a float32 mask tensor: ``1.0`` where ``self < other``,
  and ``0.0`` elsewhere.
- Comparison operations do not participate in autograd; outputs always have
  ``requires_grad=False``.
"""

from __future__ import annotations

from typing import Union

from ..._tensor_builder import tensor_control_path_manager
from .....domain.device._device import Device
from .....domain._tensor import ITensor
from ._base import TensorMixinComparison as TMA

Number = Union[int, float]
"""Scalar types accepted by Tensor comparison operators."""


@tensor_control_path_manager(TMA, TMA.__lt__, Device("cuda:0"))
def tensor_lt_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise less-than comparison (Tensor.__lt__).

    This implementation computes the elementwise mask ``self < other`` using
    the dedicated CUDA comparison extension wrapper.

    Parameters
    ----------
    self : ITensor
        Left-hand operand tensor residing on a CUDA device.
    other : Union[ITensor, Number]
        Right-hand operand. If a scalar, dispatches to the scalar CUDA kernel
        variant without promoting the scalar to a tensor.

    Returns
    -------
    ITensor
        A CUDA float32 mask tensor with ``1.0`` where ``self < other`` and
        ``0.0`` elsewhere.

    Raises
    ------
    TypeError
        If `other` is a CUDA tensor with a mismatched dtype, or if `other` is
        an unsupported type.
    ValueError
        If `other` is a CUDA tensor with a mismatched shape.
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; tensor-tensor shapes must match.
    - The returned tensor never requires gradients.
    """
    import numpy as np
    from numbers import Number as _Number

    # -----------------------------
    # CUDA path
    # -----------------------------
    if self.device.is_cuda():
        # Dedicated comparison ext wrapper (NOT arithmetic)
        from ....ops.tensor_comparison_cuda_ext import (
            lt as _cuda_lt,
            lt_scalar as _cuda_lt_scalar,
        )

        device_index = int(getattr(self.device, "index", 0) or 0)

        # Tensor-tensor
        if hasattr(other, "device"):
            other_t = other  # type: ignore[assignment]
            if not other_t.device.is_cuda():
                self._raise_device_not_supported("lt")

            self._binary_op_shape_check(self, other_t)

            # dtype must match for current CUDA comparison kernels
            if np.dtype(self.dtype) != np.dtype(other_t.dtype):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
                )

            out = _cuda_lt(self, other_t, device=device_index)
            out.requires_grad = False
            return out

        # Tensor-scalar (no scalar->tensor projection)
        if isinstance(other, _Number):
            out = _cuda_lt_scalar(self, float(other), device=device_index)
            out.requires_grad = False
            return out

        raise TypeError(
            f"unsupported operand type(s) for <: {type(self)!r} and {type(other)!r}"
        )

    self._raise_device_not_supported("lt")


@tensor_control_path_manager(TMA, TMA.__lt__, Device("cpu"))
def tensor_lt_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise less-than comparison (Tensor.__lt__).

    This implementation computes the elementwise mask ``self < other`` using
    NumPy and copies the result into a newly allocated output tensor on the CPU.

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
        A CPU float32 mask tensor with ``1.0`` where ``self < other`` and
        ``0.0`` elsewhere.

    Raises
    ------
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - The returned tensor never requires gradients.
    """
    import numpy as np

    Tensor = type(self)
    other_t = self._as_tensor_like(other, self)

    # -----------------------------
    # CPU path
    # -----------------------------
    if other_t.device.is_cpu():
        self._binary_op_shape_check(self, other_t)

        out = Tensor(shape=self.shape, device=self.device, requires_grad=False)
        out.copy_from_numpy((self.to_numpy() < other_t.to_numpy()).astype(np.float32))
        return out

    self._raise_device_not_supported("lt")
