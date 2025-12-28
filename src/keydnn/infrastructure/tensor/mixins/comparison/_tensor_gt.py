"""
Device-specific implementations of Tensor greater-than comparison via control-path dispatch.

This module registers CPU and CUDA implementations of the elementwise
greater-than comparison operator (``__gt__``) for tensors. Implementations are
selected at runtime using the `tensor_control_path_manager`, which dispatches
based on the tensor's device.

The public operator entrypoint is `TensorMixinComparison.__gt__`, and this
module provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Semantics
---------
- Performs an elementwise comparison with no broadcasting (shapes must match).
- Scalars are promoted to tensor-like operands via `_as_tensor_like`.
- The result is a float32 mask tensor: ``1.0`` where ``self > other``,
  and ``0.0`` elsewhere.
- Comparison operations do not participate in autograd; outputs always have
  ``requires_grad=False``.
"""

from ..._tensor_builder import tensor_control_path_manager

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinComparison as TMA
from typing import Union


Number = Union[int, float]
"""Scalar types accepted by Tensor comparison operators."""


@tensor_control_path_manager(TMA, TMA.__gt__, Device("cuda:0"))
def tensor_gt_gpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CUDA control path for elementwise greater-than comparison (Tensor.__gt__).

    This implementation computes the elementwise mask ``self > other`` using
    a native CUDA extension wrapper. It validates shape compatibility and
    enforces dtype equality, which is required by the current CUDA comparison
    kernels.

    Parameters
    ----------
    self : ITensor
        Left-hand operand tensor residing on a CUDA device.
    other : Union[ITensor, Number]
        Right-hand operand. If a scalar, it is promoted to a tensor compatible
        with `self` (same shape/device) via `_as_tensor_like`.

    Returns
    -------
    ITensor
        A CUDA float32 mask tensor with ``1.0`` where ``self > other`` and
        ``0.0`` elsewhere.

    Raises
    ------
    TypeError
        If operand dtypes do not match (CUDA kernels require identical dtypes).
    NotImplementedError
        If the device combination is not supported by this control path.

    Notes
    -----
    - Broadcasting is not supported; shapes must match.
    - The returned tensor never requires gradients.
    """
    import numpy as np

    other_t = self._as_tensor_like(other, self)

    # -----------------------------
    # CUDA path
    # -----------------------------

    if other_t.device.is_cuda():
        self._binary_op_shape_check(self, other_t)

        # dtype must match for CUDA comparison kernels
        if np.dtype(self.dtype) != np.dtype(other_t.dtype):
            raise TypeError(
                f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
            )

        from ....ops.tensor_arithmetic_cuda_ext import gt as _cuda_gt

        device_index = int(getattr(self.device, "index", 0) or 0)
        out = _cuda_gt(self, other_t, device=device_index)

        # Explicitly ensure no gradients
        out.requires_grad = False
        return out

    self._raise_device_not_supported("gt")


@tensor_control_path_manager(TMA, TMA.__gt__, Device("cpu"))
def tensor_gt_cpu(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
    """
    CPU control path for elementwise greater-than comparison (Tensor.__gt__).

    This implementation computes the elementwise mask ``self > other`` using
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
        A CPU float32 mask tensor with ``1.0`` where ``self > other`` and
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
        out.copy_from_numpy((self.to_numpy() > other_t.to_numpy()).astype(np.float32))
        return out

    self._raise_device_not_supported("gt")
