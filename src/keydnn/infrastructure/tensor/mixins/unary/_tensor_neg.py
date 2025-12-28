"""
Device-specific implementations of Tensor negation via control-path dispatch.

This module registers CPU and CUDA implementations of elementwise negation
for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public operator entrypoint is `TensorMixinUnary.__neg__`, and this module
provides concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations preserve the same semantics:
- elementwise negation with output shape equal to input shape,
- autograd support via the backward rule:
    d(-x)/dx = -1
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinUnary as TMU


@tensor_control_path_manager(TMU, TMU.__neg__, Device("cuda:0"))
def tensor_neg_gpu(self: ITensor) -> "ITensor":
    """
    CUDA control path for elementwise negation (unary minus).

    This implementation computes `-self` directly on the GPU by delegating to
    a native CUDA extension wrapper. The forward pass operates on device memory
    without a NumPy round-trip.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.

    Returns
    -------
    ITensor
        A CUDA tensor of the same shape as `self`, containing `-self`
        elementwise.

    Notes
    -----
    - Backward rule:
        d(-x)/dx = -1
      so the upstream gradient is negated and returned for the single parent.
    - The output tensor's `requires_grad` flag is set to mirror the input.
    """
    # -----------------------------
    # CUDA path
    # -----------------------------
    from ....ops.tensor_arithmetic_cuda_ext import neg as _cuda_neg

    device_index = int(getattr(self.device, "index", 0) or 0)
    out = _cuda_neg(self, device=device_index)

    # Preserve autograd semantics
    out.requires_grad = bool(self.requires_grad)

    if self.requires_grad:
        ctx = Context(
            parents=(self,),
            backward_fn=lambda grad_out: (-(grad_out),),
        )
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMU, TMU.__neg__, Device("cpu"))
def tensor_neg_cpu(self: ITensor) -> "ITensor":
    """
    CPU control path for elementwise negation (unary minus).

    This implementation computes `-self` on the CPU by converting the tensor to
    a NumPy array, applying NumPy negation, and copying the result into a newly
    allocated output tensor.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.

    Returns
    -------
    ITensor
        A CPU tensor of the same shape as `self`, containing `-self`
        elementwise.

    Notes
    -----
    - Backward rule:
        d(-x)/dx = -1
      so the upstream gradient is negated and returned for the single parent.
    - The output tensor's `requires_grad` flag mirrors the input.
    """
    Tensor = type(self)
    # -----------------------------
    # CPU path
    # -----------------------------
    out = Tensor(
        shape=self.shape,
        device=self.device,
        requires_grad=self.requires_grad,
    )
    out.copy_from_numpy(-self.to_numpy())

    if self.requires_grad:
        ctx = Context(
            parents=(self,),
            backward_fn=lambda grad_out: (-(grad_out),),
        )
        out._set_ctx(ctx)

    return out
