"""
Broadcast-to implementations for KeyDNN tensors (CPU and CUDA control paths).

This module registers concrete implementations of `Tensor.broadcast_to` via
`tensor_control_path_manager` for two device targets:

- CPU (`Device("cpu")`)
- CUDA (`Device("cuda:0")`)

Both implementations follow NumPy broadcasting semantics: a tensor is expanded to
a target shape (if compatible) and the result is *materialized* as a new tensor
(not a view).

Device behavior
---------------
CPU path
- Forward: uses NumPy `np.broadcast_to` on host and materializes a new CPU tensor.
- Backward: reduces `grad_out` back to the source shape by summing over the
  broadcasted axes (the inverse of broadcast expansion). The backward currently
  expects `grad_out` to be on CPU and performs the reduction with NumPy.

CUDA path
- Forward: performed fully on GPU using the Tensor-boundary CUDA ops wrapper
  (`broadcast_to_cuda_ext.broadcast_to_forward`), avoiding any NumPy round-trips.
- Backward: performed fully on GPU using the CUDA sum-to-shape reduction wrapper
  (`reduce_cuda_ext.sum_to_shape`) to reduce `grad_out` back to the source shape.
  This removes the previous CUDA->CPU->CUDA fallback and keeps CUDA graphs
  device-resident.

Autograd semantics
------------------
If `self.requires_grad` is True, the broadcast operation records a backward rule:

- Identify which axes were broadcasted (source dim == 1 while target dim > 1,
  after left-padding the source shape with ones to match rank).
- Sum-reduce `grad_out` over those axes and return a gradient tensor with the
  original source shape.

Notes
-----
- The CPU path currently normalizes to float32 to match existing CPU tensor
  conventions in this codebase.
- The CUDA path supports float32/float64 as defined by the underlying CUDA reduce
  kernels and wrappers.
"""

from typing import Union

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.broadcast_to, Device("cuda:0"))
def tensor_broadcast_gpu(self: ITensor, shape: tuple[int, ...]) -> "ITensor":
    """
    Broadcast a tensor to `shape` on CUDA by materializing an expanded copy on device.

    Forward is performed entirely on GPU using the Tensor-boundary CUDA ops wrapper
    (`broadcast_to_forward`), avoiding any NumPy round-trips.

    Autograd (CUDA)
    ---------------
    If `self.requires_grad` is True, we attach a backward rule that reduces
    `grad_out` back to the source shape by summing over broadcasted axes.

    This implementation uses the CUDA sum-to-shape reduction path (no NumPy):
      - sum_to_shape(grad_out, target_shape=src_shape)

    Parameters
    ----------
    shape : tuple[int, ...]
        Target shape to broadcast to.

    Returns
    -------
    ITensor
        A new CUDA tensor of shape `shape` containing broadcasted values.

    Raises
    ------
    ValueError
        If broadcasting is not possible.
    TypeError / RuntimeError
        If dtype/device constraints are violated (from the CUDA ext wrapper).
    """
    from .....infrastructure.ops.broadcast_to_cuda_ext import (
        broadcast_to_forward as _broadcast_to_forward,
    )
    from .....infrastructure.ops.reduce_cuda_ext import (
        sum_to_shape as _sum_to_shape,
    )

    # Forward (GPU)
    try:
        out = _broadcast_to_forward(self, shape, device=0, sync=True)  # type: ignore[arg-type]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot broadcast shape {self.shape} to {shape}") from e

    req = bool(getattr(self, "requires_grad", False))
    if not req:
        return out

    src_shape = tuple(int(d) for d in self.shape)
    tgt_shape = tuple(int(d) for d in shape)

    def backward_fn(grad_out: "ITensor"):
        """
        Reduce broadcasted gradient back to the source tensor shape (CUDA-native).

        Uses sum-to-shape reduction on CUDA (no NumPy / no host round-trips).
        """
        g = grad_out

        # Best-effort: ensure grad_out is on CUDA if some upstream produced CPU grads.
        try:
            if hasattr(g, "device") and (not g.device.is_cuda()):  # type: ignore[attr-defined]
                if hasattr(g, "to"):
                    g = g.to(self.device)  # type: ignore[call-arg]
                else:
                    raise TypeError(
                        f"broadcast_to backward requires CUDA grad_out; got device={g.device}"  # type: ignore[attr-defined]
                    )
        except AttributeError:
            # If grad_out doesn't expose `.device`, we assume it's already CUDA in this path.
            pass

        grad = _sum_to_shape(
            g,  # type: ignore[arg-type]
            target_shape=src_shape,
            device=0,
            sync=True,
        )
        return (grad,)

    ctx = Context(parents=(self,), backward_fn=backward_fn)
    ctx.saved_meta["broadcast_from"] = src_shape
    ctx.saved_meta["broadcast_to"] = tgt_shape
    out._set_ctx(ctx)
    return out


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
