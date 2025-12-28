"""
Device-specific implementations of Tensor.log via control-path dispatch.

This module registers CPU and CUDA implementations of the elementwise natural
logarithm operation for tensors. Implementations are selected at runtime using
the `tensor_control_path_manager`, which dispatches based on the tensor's device.

The public API entrypoint is `TensorMixinUnary.log`, and this module provides
concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Notes on CUDA behavior
----------------------
The CUDA control path currently uses a correctness-first CPU workaround:
- forward pass performs a device-to-host copy (to_numpy), applies NumPy log,
  then copies results back to device (copy_from_numpy).
- backward pass runs on CUDA and uses existing Tensor operators for division.

This preserves correctness and autograd semantics, but is not optimized.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinUnary as TMU


@tensor_control_path_manager(TMU, TMU.log, Device("cuda:0"))
def tensor_log_gpu(self: ITensor) -> "ITensor":
    """
    CUDA control path for elementwise natural logarithm (Tensor.log).

    This implementation currently uses a CPU round-trip workaround to compute
    the forward pass:
        device → host (`to_numpy`) → NumPy `log` → device (`copy_from_numpy`)

    The backward pass remains device-aware and computes gradients on CUDA using
    the rule:

        d(log(x)) / dx = 1 / x

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.

    Returns
    -------
    ITensor
        A CUDA tensor of the same shape as `self`, containing `log(self)`
        elementwise.

    Raises
    ------
    RuntimeError
        If the upstream gradient tensor for the backward pass is not CUDA, or
        if it is on a different CUDA device than `self`.

    Notes
    -----
    - Forward semantics follow NumPy behavior for non-positive values (e.g.,
      producing `-inf` or `nan`).
    - The output tensor explicitly allocates a CUDA buffer before copying the
      computed values back to device memory.
    - Metadata is recorded in the autograd Context to indicate that a CUDA
      workaround was used.
    """
    import numpy as np

    Tensor = type(self)

    # ============================================================
    # CUDA path (CPU workaround)
    # ============================================================
    # Forward: D2H -> NumPy -> H2D
    x_np = self.to_numpy()  # should be a CPU ndarray even if self is CUDA
    y_np = np.log(x_np).astype(np.float32, copy=False)

    out = Tensor(
        shape=self.shape,
        device=self.device,
        requires_grad=self.requires_grad,
        ctx=None,
    )
    # Ensure output has a device buffer; dtype should match your tensor dtype.
    out._ensure_cuda_alloc(dtype=np.dtype(getattr(self, "dtype", np.float32)))
    out.copy_from_numpy(y_np)

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor"):
            """
            Backward function for CUDA Tensor.log.

            Computes the gradient with respect to the input tensor using:

                d(log(x))/dx = 1/x

            Parameters
            ----------
            grad_out : ITensor
                Upstream gradient. Must be a CUDA tensor on the same device
                as `self`.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect
                to the input tensor.
            """
            if not grad_out.device.is_cuda():
                raise RuntimeError("grad_out must be CUDA for CUDA log backward")
            if str(grad_out.device) != str(self.device):
                raise RuntimeError(
                    f"grad_out device mismatch: expected {self.device!r}, got {grad_out.device!r}"
                )
            # d/dx log(x) = 1/x
            return (grad_out / self,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["cuda_workaround"] = True
        ctx.saved_meta["op"] = "log"
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMU, TMU.log, Device("cpu"))
def tensor_log_cpu(self: ITensor) -> "ITensor":
    """
    CPU control path for elementwise natural logarithm (Tensor.log).

    This implementation computes the forward pass using NumPy and copies the
    result into a newly allocated output tensor on the CPU. Autograd is
    supported via the standard derivative:

        d(log(x)) / dx = 1 / x

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.

    Returns
    -------
    ITensor
        A CPU tensor of the same shape as `self`, containing `log(self)`
        elementwise.

    Notes
    -----
    - Forward semantics follow NumPy behavior for non-positive values (e.g.,
      producing `-inf` or `nan`).
    - Backward propagation is implemented using Tensor division to stay
      consistent with the rest of the autograd system.
    """
    import numpy as np

    Tensor = type(self)
    # ============================================================
    # CPU path (KEEP EXACT SEMANTICS)
    # ============================================================
    out = Tensor(
        shape=self.shape,
        device=self.device,
        requires_grad=self.requires_grad,
    )

    out.copy_from_numpy(np.log(self.to_numpy()))

    if self.requires_grad:
        ctx = Context(
            parents=(self,),
            backward_fn=lambda grad_out: (grad_out / self,),
        )
        out._set_ctx(ctx)

    return out
