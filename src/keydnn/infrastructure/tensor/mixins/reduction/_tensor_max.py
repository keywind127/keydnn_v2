"""
Device-specific implementations of Tensor.max using control-path dispatch.

This module provides CPU and CUDA implementations of the Tensor.max operation,
registered via the tensor_control_path_manager. The correct implementation is
selected at runtime based on the tensor's device.

The implementations here preserve the public Tensor.max API while specializing
the execution strategy, numerical backend, and backward propagation logic for
each supported device.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinReduction as TMR


@tensor_control_path_manager(TMR, TMR.max, Device("cuda:0"))
def tensor_max_gpu(self: ITensor, axis: int = -1, keepdims: bool = False) -> ITensor:
    """
    CUDA implementation of Tensor.max for 2D tensors.

    This function implements the forward and backward logic for computing the
    maximum values of a 2D CUDA tensor along a specified axis, using native
    CUDA extension kernels. It is registered as a control path for
    `TensorMixinReduction.max` when the tensor resides on `cuda:0`.

    Parameters
    ----------
    self : ITensor
        Input tensor. Must be a CUDA tensor with exactly two dimensions.
    axis : int, optional
        Axis along which to compute the maximum. Supported values are
        {0, 1, -1, -2}. Defaults to -1.
    keepdims : bool, optional
        Whether to retain reduced dimensions in the output tensor.
        Defaults to False.

    Returns
    -------
    ITensor
        Output tensor containing the maximum values along the specified axis.

    Raises
    ------
    NotImplementedError
        If the input tensor is not 2D, or if required reshape functionality
        is unavailable when keepdims=True.
    ValueError
        If an unsupported axis value is provided.

    Notes
    -----
    - This CUDA path is currently limited to 2D tensors to match the available
      `max_axis2d` CUDA kernels.
    - The forward kernel returns both the reduced values and the argmax indices;
      the indices are stored for use during backward propagation.
    - The backward pass scatters gradients back to the input tensor using the
      saved argmax indices.
    """
    import numpy as np

    # -----------------------
    # CUDA path (2D only, axis 0/1)
    # -----------------------
    if len(self.shape) != 2:
        raise NotImplementedError(
            "CUDA Tensor.max currently supports 2D tensors only "
            "(to match max_axis2d kernels)."
        )

    rows, cols = int(self.shape[0]), int(self.shape[1])

    # normalize axis to 0 or 1 for 2D
    axis_ = axis
    if axis_ < 0:
        axis_ = 2 + axis_
    if axis_ not in (0, 1):
        raise ValueError(
            "CUDA Tensor.max only supports axis in {0,1,-1,-2} for 2D tensors."
        )

    dtype = np.float32  # keep current dtype policy

    from ....ops.reduce_cuda_ext import (
        max_axis2d_forward as _max_axis2d_forward,
        max_axis2d_backward as _max_axis2d_backward,
    )

    # forward returns (y, idx) where:
    # y: (out_len,), idx: (out_len,) int64 — both CUDA tensors
    y_base, idx = _max_axis2d_forward(self, axis=int(axis_), device=0, sync=True)

    # keepdims reshape for output only (idx stays 1D)
    if keepdims:
        out_shape = (1, cols) if axis_ == 0 else (rows, 1)
        if hasattr(y_base, "reshape") and callable(getattr(y_base, "reshape")):
            out = y_base.reshape(out_shape)  # type: ignore[call-arg]
        else:
            raise NotImplementedError(
                "keepdims=True for CUDA max(axis=...) requires Tensor.reshape support"
            )
    else:
        out = y_base

    # preserve autograd contract
    out.requires_grad = bool(self.requires_grad)

    if self.requires_grad:

        def backward_fn(grad_out: ITensor):
            """
            Backward function for CUDA Tensor.max.

            Computes the gradient with respect to the input tensor by scattering
            the upstream gradients to the positions of the maximum values
            recorded during the forward pass.

            Parameters
            ----------
            grad_out : ITensor
                Gradient of the output tensor. Must be a CUDA tensor.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect to
                the input tensor.
            """
            if not grad_out.device.is_cuda():
                raise RuntimeError("grad_out must be CUDA for CUDA Tensor.max backward")

            # For keepdims=True, grad_out may be (1, cols) or (rows, 1).
            # The ext wrapper expects 1D payload: (cols,) or (rows,).
            go = grad_out
            if keepdims:
                exp_shape = (cols,) if axis_ == 0 else (rows,)
                if tuple(int(d) for d in grad_out.shape) != exp_shape:
                    if hasattr(grad_out, "reshape") and callable(
                        getattr(grad_out, "reshape")
                    ):
                        go = grad_out.reshape(exp_shape)  # type: ignore[call-arg]
                    else:
                        raise NotImplementedError(
                            "CUDA max backward with keepdims=True requires Tensor.reshape support"
                        )

            gx = _max_axis2d_backward(
                go,
                idx,
                rows=rows,
                cols=cols,
                axis=int(axis_),
                device=0,
                zero_grad_x=True,
                sync=True,
            )
            return (gx,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["axis"] = axis_
        ctx.saved_meta["keepdims"] = keepdims
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMR, TMR.max, Device("cpu"))
def tensor_max_cpu(self: ITensor, axis: int = -1, keepdims: bool = False) -> ITensor:
    """
    CPU implementation of Tensor.max using NumPy.

    This function implements the forward and backward logic for computing the
    maximum values of a tensor on the CPU. It mirrors standard NumPy semantics
    and serves as the reference implementation for Tensor.max.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.
    axis : int, optional
        Axis along which to compute the maximum. Defaults to -1.
    keepdims : bool, optional
        Whether to retain reduced dimensions in the output tensor.
        Defaults to False.

    Returns
    -------
    ITensor
        Output tensor containing the maximum values along the specified axis.

    Raises
    ------
    ValueError
        If the specified axis is out of bounds for the input tensor.

    Notes
    -----
    - The forward pass delegates to NumPy's `np.max`.
    - The backward pass computes gradients by masking the input tensor with
      positions equal to the maximum values.
    - This implementation currently requires gradients to remain on the CPU.
    """
    import numpy as np

    Tensor = type(self)
    # -----------------------
    # CPU path (existing behavior)
    # -----------------------

    x_np = self.to_numpy()
    ndim = x_np.ndim
    axis_ = axis if axis >= 0 else ndim + axis
    if axis_ < 0 or axis_ >= ndim:
        raise ValueError(f"axis {axis} out of bounds for ndim {ndim}")

    m_np = np.max(x_np, axis=axis_, keepdims=keepdims).astype(np.float32, copy=False)
    out = Tensor(shape=m_np.shape, device=self.device, requires_grad=self.requires_grad)
    out.copy_from_numpy(m_np)

    if self.requires_grad:

        def backward_fn(grad_out: ITensor):
            """
            Backward function for CPU Tensor.max.

            Computes the gradient with respect to the input tensor by propagating
            upstream gradients only to elements equal to the maximum value along
            the reduced axis.

            Parameters
            ----------
            grad_out : ITensor
                Gradient of the output tensor. Must be a CPU tensor.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect to
                the input tensor.
            """
            if not grad_out.device.is_cpu():
                raise RuntimeError("grad_out must be CPU in current implementation")

            g = grad_out.to_numpy().astype(np.float32, copy=False)

            m = m_np
            g_aligned = g
            if not keepdims:
                m = np.expand_dims(m, axis=axis_)
                g_aligned = np.expand_dims(g_aligned, axis=axis_)

            mask = (x_np == m).astype(np.float32)
            grad_np = mask * g_aligned

            grad = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            grad.copy_from_numpy(grad_np.astype(np.float32, copy=False))
            return (grad,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["axis"] = axis_
        ctx.saved_meta["keepdims"] = keepdims
        out._set_ctx(ctx)

    return out
