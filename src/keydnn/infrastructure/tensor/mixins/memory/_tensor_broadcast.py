"""
CPU broadcast implementation for KeyDNN tensors.

This module defines a CPU-only broadcast helper that expands a tensor to a
target shape using NumPy broadcasting semantics, then materializes the result
as a new `Tensor`.

The function is registered through `tensor_control_path_manager`, which
dispatches calls for the (mixin, method, device) triple to this concrete
implementation.

Autograd
--------
If `self.requires_grad` is True, the broadcast operation records a backward
rule that reduces `grad_out` back to the original source shape by summing over
broadcasted axes (i.e., the inverse of broadcast expansion), matching standard
array autodiff behavior.
"""

from typing import Union

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.broadcast_to, Device("cpu"))
def tensor_broadcast_cpu(self: ITensor, shape: tuple[int, ...]) -> "ITensor":
    """
    Broadcast a tensor to `shape` on CPU by materializing a NumPy-broadcasted copy.

    This function expands `self` to `shape` using NumPy's broadcasting rules and
    stores the broadcasted result into a newly allocated tensor on CPU. The output
    is always materialized (not a view), and its dtype is normalized to float32 to
    match the framework's CPU tensor conventions.

    Parameters
    ----------
    shape : tuple[int, ...]
        Target shape to broadcast to. Must be compatible with `self.shape` under
        NumPy broadcasting rules.

    Returns
    -------
    ITensor
        A new tensor of shape `shape` on CPU containing the broadcasted values.

    Raises
    ------
    ValueError
        If `self.shape` cannot be broadcast to `shape`.

    Notes
    -----
    - This is an explicit broadcasting primitive used to keep higher-level binary
      operations strict unless broadcasting is requested.
    - The backward rule reduces gradients by summing over broadcasted dimensions.
    - Current backward implementation requires `grad_out` to be on CPU.
    """
    import numpy as np

    Tensor = type(self)

    src = self.to_numpy()
    try:
        out_np = np.broadcast_to(src, shape).astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"Cannot broadcast shape {self.shape} to {shape}") from e

    req = self.requires_grad
    out = Tensor(shape=shape, device=self.device, requires_grad=req, ctx=None)
    out.copy_from_numpy(out_np)

    if req:
        src_shape = self.shape

        def backward_fn(grad_out: "ITensor"):
            """
            Reduce broadcasted gradient back to the source tensor shape.

            Parameters
            ----------
            grad_out : ITensor
                Gradient w.r.t. the broadcasted output. Must be a CPU tensor in the
                current implementation.

            Returns
            -------
            tuple[ITensor]
                A one-tuple containing the gradient w.r.t. the input tensor `self`,
                with shape equal to `src_shape`.

            Raises
            ------
            RuntimeError
                If `grad_out` is not on CPU.
            """
            if not grad_out.device.is_cpu():
                raise RuntimeError("grad_out must be CPU in current implementation")

            g = grad_out.to_numpy().astype(np.float32, copy=False)

            # Align src_shape to target rank by left-padding with ones
            src_rank = len(src_shape)
            tgt_rank = len(shape)
            padded_src = (1,) * (tgt_rank - src_rank) + src_shape

            # Sum over axes that were broadcasted (src dim == 1 and target dim > 1)
            reduce_axes = []
            for i, (sd, td) in enumerate(zip(padded_src, shape)):
                if sd == 1 and td != 1:
                    reduce_axes.append(i)

            if reduce_axes:
                g = np.sum(g, axis=tuple(reduce_axes), keepdims=True)

            # Remove left padding dims to return to src_shape
            if tgt_rank != src_rank:
                for _ in range(tgt_rank - src_rank):
                    g = np.squeeze(g, axis=0)

            grad = Tensor(shape=src_shape, device=self.device, requires_grad=False)
            grad.copy_from_numpy(g.astype(np.float32, copy=False))
            return (grad,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["broadcast_from"] = src_shape
        ctx.saved_meta["broadcast_to"] = shape
        out._set_ctx(ctx)

    return out
