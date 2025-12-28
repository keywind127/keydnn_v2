"""
Device-specific implementations of Tensor.sqrt via control-path dispatch.

This module registers CPU and CUDA implementations of the elementwise square
root operation for tensors. Implementations are selected at runtime using the
`tensor_control_path_manager`, which dispatches based on the tensor's device.

The public API entrypoint is `TensorMixinUnary.sqrt`, and this module provides
concrete control paths for:
- Device("cpu")
- Device("cuda:0")

Notes on CUDA behavior
----------------------
The CUDA control path currently uses a correctness-first CPU workaround:
- forward pass performs a device-to-host copy (to_numpy), applies NumPy sqrt,
  then copies results back to device (copy_from_numpy).
- backward pass runs on CUDA and uses Tensor arithmetic to compute the local
  derivative against the forward output.

This preserves correctness and autograd semantics, but is not optimized.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinUnary as TMU


@tensor_control_path_manager(TMU, TMU.sqrt, Device("cuda:0"))
def tensor_sqrt_gpu(self: ITensor) -> "ITensor":
    """
    CUDA control path for elementwise square root (Tensor.sqrt).

    This implementation currently uses a CPU round-trip workaround for the
    forward pass:
        device → host (`to_numpy`) → NumPy `sqrt` → device (`copy_from_numpy`)

    The backward pass is performed on CUDA using the derivative:

        d(sqrt(x)) / dx = 0.5 / sqrt(x)

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.

    Returns
    -------
    ITensor
        A CUDA tensor of the same shape as `self`, containing `sqrt(self)`
        elementwise.

    Raises
    ------
    RuntimeError
        If the upstream gradient is not a CUDA tensor, or if it is on a
        different CUDA device than `self`.

    Notes
    -----
    - Forward behavior follows NumPy semantics for negative inputs (e.g.,
      producing `nan`).
    - The output tensor explicitly allocates a CUDA buffer before copying
      results back to device memory.
    - Metadata is recorded in the autograd Context to indicate that a CUDA
      workaround was used.
    """
    from typing import Sequence, Optional

    Tensor = type(self)
    import numpy as np

    # ============================================================
    # CUDA path (CPU workaround)
    # ============================================================
    # Forward: D2H -> NumPy -> H2D
    x_np = self.to_numpy()
    y_np = np.sqrt(x_np).astype(np.float32, copy=False)

    out = Tensor(
        shape=self.shape,
        device=self.device,
        requires_grad=self.requires_grad,
        ctx=None,
    )
    out._ensure_cuda_alloc(dtype=np.dtype(getattr(self, "dtype", np.float32)))
    out.copy_from_numpy(y_np)

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor") -> Sequence[Optional["ITensor"]]:
            """
            Backward function for CUDA Tensor.sqrt.

            Computes the gradient with respect to the input using:

                d(sqrt(x))/dx = 0.5 / sqrt(x)

            Parameters
            ----------
            grad_out : ITensor
                Upstream gradient. Must be a CUDA tensor on the same device
                as `self`.

            Returns
            -------
            Sequence[Optional[ITensor]]
                A single-element sequence containing the gradient with respect
                to the input tensor.
            """
            if not grad_out.device.is_cuda():
                raise RuntimeError("grad_out must be CUDA for CUDA sqrt backward")
            if str(grad_out.device) != str(self.device):
                raise RuntimeError(
                    f"grad_out device mismatch: expected {self.device!r}, got {grad_out.device!r}"
                )

            # d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / out
            return (grad_out * (0.5 / out),)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["cuda_workaround"] = True
        ctx.saved_meta["op"] = "sqrt"
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMU, TMU.sqrt, Device("cpu"))
def tensor_sqrt_cpu(self: ITensor) -> "ITensor":
    """
    CPU control path for elementwise square root (Tensor.sqrt).

    This implementation computes the forward pass using NumPy and copies the
    result into a newly allocated output tensor. Backward propagation is
    implemented explicitly using NumPy to preserve current CPU semantics.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.

    Returns
    -------
    ITensor
        A CPU tensor of the same shape as `self`, containing `sqrt(self)`
        elementwise.

    Notes
    -----
    - Forward behavior follows NumPy semantics for negative inputs (e.g.,
      producing `nan`).
    - Backward rule:
        d(sqrt(x))/dx = 0.5 / sqrt(x)
      The backward implementation uses the saved forward output `out`.
    - The forward output is saved via `ctx.save_for_backward(out)` for reuse.
    """
    from typing import Sequence, Optional
    import numpy as np

    Tensor = type(self)

    # ============================================================
    # CPU path (KEEP EXACT SEMANTICS)
    # ============================================================
    x_np = self.to_numpy()
    y_np = np.sqrt(x_np).astype(np.float32, copy=False)

    out = Tensor(shape=self.shape, device=self.device, requires_grad=self.requires_grad)
    out.copy_from_numpy(y_np)

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor") -> Sequence[Optional["ITensor"]]:
            """
            Backward function for CPU Tensor.sqrt.

            Computes the gradient with respect to the input using:

                d(sqrt(x))/dx = 0.5 / sqrt(x)

            Parameters
            ----------
            grad_out : ITensor
                Upstream gradient. Must be a CPU tensor.

            Returns
            -------
            Sequence[Optional[ITensor]]
                A single-element sequence containing the gradient with respect
                to the input tensor.
            """
            if not grad_out.device.is_cpu():
                grad_out._raise_device_not_supported("sqrt_backward")

            go = grad_out.to_numpy().astype(np.float32, copy=False)
            y = out.to_numpy().astype(np.float32, copy=False)

            gx_np = (go * (0.5 / y)).astype(np.float32, copy=False)

            gx = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            gx.copy_from_numpy(gx_np)
            return (gx,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.save_for_backward(out)
        out._set_ctx(ctx)

    return out
