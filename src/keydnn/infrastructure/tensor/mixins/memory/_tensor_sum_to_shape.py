"""
Tensor sum-to-shape (inverse-broadcast) implementations for CPU and CUDA.

This module implements the `sum_to_shape` primitive, which is commonly used in
autograd systems to reduce gradients back to an operand's original shape after a
broadcasted forward operation.

Conceptually, `sum_to_shape` is the inverse of `broadcast_to`:
- Forward: broadcast a smaller tensor to a larger shape for elementwise ops.
- Backward: sum-reduce the gradient over the broadcasted axes to recover the
  gradient in the original smaller shape.

Two device-specialized implementations are provided:
- `tensor_sum_to_shape_cpu`: NumPy-based reduction on CPU tensors.
- `tensor_sum_to_shape_gpu`: CUDA-native reduction that stays on device.

The dispatch to the appropriate implementation is handled by
`tensor_control_path_manager`, which registers each function as the control-path
implementation of `TensorMixinMemory.sum_to_shape` for a specific `Device`.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM


def _sum_to_shape_reduce_axes(
    src_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """
    Compute the padded target shape and reduction axes for `sum_to_shape`.

    Given a source shape `src_shape` (typically the broadcasted/result shape) and a
    desired `target_shape` (the original pre-broadcast shape), this helper:

    1) Left-pads `target_shape` with leading ones so it has the same rank as
       `src_shape`. This makes axis-wise comparison straightforward.
    2) Validates that `target_shape` could have been broadcast to `src_shape`
       under standard NumPy broadcasting rules.
    3) Determines which axes must be reduced (summed) to collapse broadcasted
       dimensions back to size 1.

    Parameters
    ----------
    src_shape:
        The source (larger) shape to reduce from.
    target_shape:
        The target (smaller or equal-rank) shape to reduce to.

    Returns
    -------
    padded_target:
        `target_shape` left-padded with ones to match `len(src_shape)`.
    reduce_axes:
        Axes in the source to sum over using `keepdims=True`.
        An axis is included if the padded target dimension is 1 while the source
        dimension is not 1.
    pad:
        The number of leading dimensions added to the target (i.e., rank difference).

    Raises
    ------
    ValueError
        If `target_shape` has higher rank than `src_shape`, or if any dimension is
        not broadcast-compatible (target dim must be 1 or equal to source dim).

    Notes
    -----
    The returned `reduce_axes` are designed to be used with `keepdims=True`, so
    the padded rank is preserved and the left-padding can later be removed cleanly.
    """
    src = tuple(int(d) for d in src_shape)
    tgt = tuple(int(d) for d in target_shape)

    if len(tgt) > len(src):
        raise ValueError(f"target_shape rank {len(tgt)} > src rank {len(src)}")

    pad = len(src) - len(tgt)
    padded_tgt = (1,) * pad + tgt

    # Validate broadcast-compatibility: padded_tgt must be broadcastable to src
    for i, (sd, td) in enumerate(zip(src, padded_tgt)):
        if td not in (1, sd):
            raise ValueError(
                f"Cannot sum_to_shape from {src_shape} to {target_shape}: "
                f"dim mismatch at axis {i}: src={sd}, target={td}"
            )

    # Reduce axes: any axis where target is 1 but src is >1
    reduce_axes = tuple(
        i for i, (sd, td) in enumerate(zip(src, padded_tgt)) if td == 1 and sd != 1
    )

    return padded_tgt, reduce_axes, pad


@tensor_control_path_manager(TMM, TMM.sum_to_shape, Device("cpu"))
def tensor_sum_to_shape_cpu(self: ITensor, target_shape: tuple[int, ...]) -> "ITensor":
    """
    Sum-reduce a CPU tensor to `target_shape` using NumPy.

    This operation is typically used during backpropagation of broadcasted ops.
    If a tensor was broadcast from `target_shape` to `self.shape` during the
    forward pass, the corresponding gradient must be reduced (summed) over the
    broadcasted axes to recover the gradient in the original shape.

    Implementation details
    ----------------------
    - Converts the tensor to a NumPy array.
    - Computes reduction axes via `_sum_to_shape_reduce_axes`.
    - Sums over broadcasted axes with `keepdims=True`.
    - Squeezes away any leading padding dimensions to restore the target rank.
    - Allocates a new tensor of `target_shape` on CPU and copies data back.
    - If `requires_grad` is set, attaches an autograd `Context` whose backward
      broadcasts `grad_out` back to the input shape.

    Parameters
    ----------
    self:
        Source tensor to be reduced (CPU device).
    target_shape:
        Desired output shape after reduction. Must be broadcast-compatible with
        `self.shape` (i.e., could have been broadcast to `self.shape`).

    Returns
    -------
    ITensor
        A new tensor with shape `target_shape` on the same device as `self`.

    Raises
    ------
    ValueError
        If `target_shape` is not compatible with `self.shape` under broadcasting.
    RuntimeError
        If an internal shape invariant is violated after reduction (defensive check).

    Autograd
    --------
    Backward rule: if forward reduces `self.shape -> target_shape`, then backward
    maps `grad_out` back to `self.shape` by broadcasting:
        dL/dx = broadcast_to(dL/dy, self.shape)
    """
    import numpy as np

    Tensor = type(self)

    src_shape = tuple(int(d) for d in self.shape)
    tgt_shape = tuple(int(d) for d in target_shape)

    padded_tgt, reduce_axes, pad = _sum_to_shape_reduce_axes(src_shape, tgt_shape)

    x = self.to_numpy()

    # Sum along broadcasted axes, keeping dims so we can drop padding cleanly
    if reduce_axes:
        x = np.sum(x, axis=reduce_axes, keepdims=True)

    # Drop the left padding dims to match target rank
    if pad:
        for _ in range(pad):
            x = np.squeeze(x, axis=0)

    # Now x should have shape == tgt_shape
    if tuple(int(d) for d in x.shape) != tgt_shape:
        raise RuntimeError(
            f"sum_to_shape_cpu internal error: got shape {x.shape}, expected {tgt_shape}"
        )

    req = bool(getattr(self, "requires_grad", False))
    out = Tensor(shape=tgt_shape, device=self.device, requires_grad=req, ctx=None)  # type: ignore[call-arg]
    out.copy_from_numpy(np.ascontiguousarray(x, dtype=np.dtype(self.dtype)))

    if req:
        in_shape = src_shape

        def backward_fn(grad_out: "ITensor"):
            # Backward of sum-to-shape is broadcast back to input shape
            g = grad_out.broadcast_to(in_shape)  # type: ignore[attr-defined]
            return (g,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["sum_to_shape_from"] = src_shape
        ctx.saved_meta["sum_to_shape_to"] = tgt_shape
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMM, TMM.sum_to_shape, Device("cuda:0"))
def tensor_sum_to_shape_gpu(self: ITensor, target_shape: tuple[int, ...]) -> "ITensor":
    """
    Sum-reduce a CUDA tensor to `target_shape` using a CUDA-native implementation.

    This GPU implementation performs the same conceptual operation as
    `tensor_sum_to_shape_cpu` (inverse of broadcasting), but keeps the entire
    forward pass on device by delegating to `reduce_cuda_ext.sum_to_shape`.

    Parameters
    ----------
    self:
        Source tensor to be reduced (CUDA device).
    target_shape:
        Desired output shape after reduction. Must be broadcast-compatible with
        `self.shape` (i.e., could have been broadcast to `self.shape`).

    Returns
    -------
    ITensor
        A new CUDA tensor with shape `target_shape` on the same device as `self`.

    Raises
    ------
    ValueError
        If `target_shape` is not compatible with `self.shape` under broadcasting.

    Autograd
    --------
    - The CUDA extension may return tensors with `requires_grad=False` regardless
      of the input flag. This wrapper re-applies `requires_grad` when needed.
    - Backward rule is identical to CPU: broadcast `grad_out` back to the input
      shape:
          dL/dx = broadcast_to(dL/dy, self.shape)

    Notes
    -----
    Shape compatibility is validated using `_sum_to_shape_reduce_axes` to ensure
    consistent behavior with the CPU implementation.
    """
    # Shape validation (same rule as CPU)
    src_shape = tuple(int(d) for d in self.shape)
    tgt_shape = tuple(int(d) for d in target_shape)
    _ = _sum_to_shape_reduce_axes(src_shape, tgt_shape)  # validates

    from .....infrastructure.ops.reduce_cuda_ext import (
        sum_to_shape as _sum_to_shape_cuda,
    )

    # Forward (GPU)
    out = _sum_to_shape_cuda(
        self,  # type: ignore[arg-type]
        out_shape=tgt_shape,
        device=int(getattr(self.device, "index", 0) or 0),  # best-effort
        sync=True,
    )

    req = bool(getattr(self, "requires_grad", False))
    if not req:
        return out

    # Ensure autograd flag propagates (ext wrappers return requires_grad=False)
    if req:
        if hasattr(out, "_requires_grad"):
            setattr(out, "_requires_grad", True)
        elif hasattr(out, "requires_grad"):
            setattr(out, "requires_grad", True)

    in_shape = src_shape

    def backward_fn(grad_out: "ITensor"):
        # Backward of sum-to-shape is broadcast back to input shape
        g = grad_out.broadcast_to(in_shape)  # type: ignore[attr-defined]
        return (g,)

    ctx = Context(parents=(self,), backward_fn=backward_fn)
    ctx.saved_meta["sum_to_shape_from"] = src_shape
    ctx.saved_meta["sum_to_shape_to"] = tgt_shape
    out._set_ctx(ctx)
    return out
