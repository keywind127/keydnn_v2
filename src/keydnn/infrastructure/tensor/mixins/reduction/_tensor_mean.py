"""
Device-specific implementations of Tensor.mean using control-path dispatch.

This module defines CPU and CUDA implementations of the Tensor.mean reduction,
registered via the tensor_control_path_manager. The appropriate implementation
is selected at runtime based on the tensor's device.

Both implementations preserve the public Tensor.mean API while specializing
the execution strategy and backward propagation logic for each device.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor
from ._base import TensorMixinReduction as TMR


@tensor_control_path_manager(TMR, TMR.mean, Device("cpu"))
def tensor_mean_cpu(self: ITensor) -> ITensor:
    """
    CPU implementation of Tensor.mean.

    Computes the arithmetic mean of all elements in the tensor using NumPy,
    returning a scalar (0-dimensional) tensor. This function serves as the
    reference implementation of Tensor.mean for CPU tensors.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.

    Returns
    -------
    ITensor
        A scalar tensor (shape=()) containing the mean of all elements.

    Notes
    -----
    - The mean is computed as ``sum(x) / numel(x)``.
    - The output tensor is always a scalar, regardless of the input shape.
    - Gradients are propagated uniformly to all input elements during the
      backward pass.
    """
    import numpy as np

    Tensor = type(self)
    # -----------------------
    # CPU path (existing behavior)
    # -----------------------

    n = self.numel()
    value = float(np.sum(self.data) / n)
    out = Tensor(shape=(), device=self.device, requires_grad=self.requires_grad)
    out.copy_from_numpy(np.array(value, dtype=np.float32))

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor"):
            """
            Backward function for CPU Tensor.mean.

            Distributes the upstream scalar gradient evenly across all input
            elements, scaled by ``1 / numel``.

            Parameters
            ----------
            grad_out : ITensor
                Gradient of the output scalar tensor. Must reside on the CPU.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect to
                the input tensor.
            """
            grad = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            grad.copy_from_numpy(
                np.ones(self.shape, dtype=np.float32)
                * (float(np.asarray(grad_out.to_numpy())) / n)
            )
            return (grad,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMR, TMR.mean, Device("cuda:0"))
def tensor_mean_gpu(self: ITensor) -> ITensor:
    """
    CUDA implementation of Tensor.mean using all-reduce kernels.

    Computes the arithmetic mean of all elements in a CUDA tensor by delegating
    to native CUDA reduction kernels. The forward pass produces a scalar CUDA
    tensor, and the backward pass fills the gradient uniformly across all input
    elements.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.

    Returns
    -------
    ITensor
        A scalar CUDA tensor (shape=()) containing the mean of all elements.

    Raises
    ------
    RuntimeError
        If the upstream gradient for the backward pass is not a CUDA tensor.
    NotImplementedError
        If Tensor.reshape is unavailable during the backward pass.

    Notes
    -----
    - This implementation performs a full-tensor (all-reduce) mean; no axis
      argument is currently supported.
    - The output tensor's ``requires_grad`` flag mirrors that of the input.
    - Backward propagation relies on a CUDA kernel that generates a flat
      gradient tensor, which is reshaped to match the input tensor's shape.
    """
    import numpy as np

    # -----------------------
    # CUDA path (all-reduce)
    # -----------------------
    if self.device.is_cuda():
        # Keep current dtype policy (float32) until Tensor tracks dtype end-to-end
        dtype = np.float32
        n = int(self.numel())

        from ....ops.reduce_cuda_ext import (
            mean_all_forward as _mean_all_forward,
            mean_backward_fill_forward as _mean_backward_fill_forward,
        )

        # forward: returns CUDA scalar tensor (shape=())
        out = _mean_all_forward(self, device=0, sync=True)

        # preserve autograd contract: output requires_grad mirrors input
        out.requires_grad = bool(self.requires_grad)

        if self.requires_grad:

            def backward_fn(grad_out: "ITensor"):
                """
                Backward function for CUDA Tensor.mean.

                Generates a gradient tensor in which every input element
                receives the upstream gradient scaled by ``1 / numel``.

                Parameters
                ----------
                grad_out : ITensor
                    Gradient of the output scalar tensor. Must be a CUDA tensor.

                Returns
                -------
                tuple[ITensor]
                    A single-element tuple containing the gradient with respect
                    to the input tensor.
                """
                if not grad_out.device.is_cuda():
                    raise RuntimeError(
                        "grad_out must be CUDA for CUDA Tensor.mean backward"
                    )

                # mean backward fill returns a flat (n,) CUDA tensor
                gx_flat = _mean_backward_fill_forward(
                    grad_out, numel=n, device=0, sync=True
                )

                # reshape back to input shape
                if hasattr(gx_flat, "reshape") and callable(
                    getattr(gx_flat, "reshape")
                ):
                    gx = gx_flat.reshape(self.shape)  # type: ignore[call-arg]
                else:
                    raise NotImplementedError(
                        "CUDA mean backward requires Tensor.reshape support"
                    )

                return (gx,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            out._set_ctx(ctx)

        return out
