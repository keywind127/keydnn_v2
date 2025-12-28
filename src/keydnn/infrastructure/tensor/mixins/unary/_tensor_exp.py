"""
Device-specific implementations of Tensor.exp via control-path dispatch.

This module registers CPU and CUDA implementations of the elementwise
exponential operation for tensors. Implementations are selected at runtime
using the `tensor_control_path_manager`, which dispatches based on the tensor's
device.

The public API entrypoint is `TensorMixinUnary.exp`, and this module provides
concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Both implementations preserve the same semantics:
- elementwise exp with output shape equal to input shape,
- autograd support via a Context storing the backward rule:
    d(exp(x))/dx = exp(x)
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinUnary as TMU


@tensor_control_path_manager(TMU, TMU.exp, Device("cuda:0"))
def tensor_exp_gpu(self: ITensor) -> "ITensor":
    """
    CUDA control path for elementwise exponential (Tensor.exp).

    This implementation computes the elementwise exponential of a CUDA tensor
    by delegating to a native CUDA kernel wrapper. The forward kernel operates
    directly on device memory (no host round-trip).

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.

    Returns
    -------
    ITensor
        A CUDA tensor of the same shape as `self`, containing `exp(self)`
        elementwise.

    Notes
    -----
    - The underlying CUDA wrapper `exp_forward` returns a tensor with
      `requires_grad=False` by design; this function restores autograd
      participation by setting `out.requires_grad` and attaching a Context.
    - Backward rule:
        d(exp(x))/dx = exp(x)
      The implementation reuses the forward output `out` to avoid recomputing
      exp during backpropagation.
    """
    # -------------------------
    # CUDA path (native kernel)
    # -------------------------

    # exp_forward returns requires_grad=False by design; we attach ctx below if needed.
    from ....ops.unary_cuda_ext import exp_forward as _exp_forward

    out = _exp_forward(
        self, device=self.device.index if hasattr(self.device, "index") else 0
    )

    # Preserve autograd participation
    out.requires_grad = self.requires_grad  # if your Tensor allows attribute set

    if self.requires_grad:
        ctx = Context(
            parents=(self,),
            backward_fn=lambda grad_out: (grad_out * out,),
        )
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMU, TMU.exp, Device("cpu"))
def tensor_exp_cpu(self: ITensor) -> "ITensor":
    """
    CPU control path for elementwise exponential (Tensor.exp).

    This implementation computes the elementwise exponential of a CPU tensor
    using NumPy and copies the result into a newly allocated output tensor.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.

    Returns
    -------
    ITensor
        A CPU tensor of the same shape as `self`, containing `exp(self)`
        elementwise.

    Notes
    -----
    - Forward pass uses `np.exp` on the NumPy representation of the tensor.
    - Backward rule:
        d(exp(x))/dx = exp(x)
      The implementation reuses the forward output `out` in the backward pass.
    """
    import numpy as np

    Tensor = type(self)

    # -------------------------
    # CPU path (NumPy)
    # -------------------------
    out = Tensor(shape=self.shape, device=self.device, requires_grad=self.requires_grad)
    out.copy_from_numpy(np.exp(self.to_numpy()).astype(np.float32, copy=False))

    if self.requires_grad:
        # Save output to reuse in backward
        ctx = Context(
            parents=(self,),
            backward_fn=lambda grad_out: (grad_out * out,),
        )
        out._set_ctx(ctx)

    return out
