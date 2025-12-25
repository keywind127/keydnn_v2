"""
Autograd `Function` adapters for 2D pooling (CPU/CUDA backends).

This module connects the pooling primitives in `ops.pool2d_cpu_ext` and
`ops.pool2d_cuda_ext` to the KeyDNN autograd runtime (`Tensor`, `Context`,
`Function`). Each pooling operator is implemented as a `Function` subclass that:

- implements `forward(ctx, ...)` to compute the output tensor, and
- implements `backward(ctx, grad_out)` to compute gradients w.r.t. inputs.

The forward path normalizes pooling hyperparameters (`kernel_size`, `stride`,
`padding`) using `_pair`, and saves the minimal information needed for backward
into `ctx.saved_tensors` and `ctx.saved_meta`.

Implemented operators
---------------------
- `MaxPool2dFn`
    Uses argmax indices from the forward pass to scatter gradients back to the
    maximal elements in each pooling window.
- `AvgPool2dFn`
    Distributes output gradients uniformly over each pooling window.
- `GlobalAvgPool2dFn`
    Averages over spatial dimensions (H, W) and distributes gradients evenly
    across all H*W positions per channel.

Design notes
------------
- Assumes **NCHW** layout: (N, C, H, W).
- Device-dispatch happens here:
  - CPU tensors call `ops.pool2d_cpu_ext`
  - CUDA tensors call `ops.pool2d_cuda_ext`
- The actual numeric work is performed by backend wrappers, which hide backend
  boundaries behind the `Tensor` API (CPU: `to_numpy`/`_from_numpy`,
  CUDA: device pointers via `Tensor.data`/`_from_devptr`).
- For backward compatibility, outputs explicitly mirror the input's
  `requires_grad` flag (`y.requires_grad = x.requires_grad`) instead of relying
  on kernel wrappers to set it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..tensor._tensor_context import Context

from .._function import Function
from .._tensor import Tensor
from ..ops.pool2d_cpu import _pair
from ..ops import pool2d_cpu_ext as cpu_ext
from ..ops import pool2d_cuda_ext as cuda_ext


def _dispatch_backend(x: Tensor) -> str:
    """Return backend tag for x ("cpu" | "cuda") or raise for unsupported device."""
    if x.device.is_cpu():
        return "cpu"
    if x.device.is_cuda():
        return "cuda"
    raise TypeError(f"Unsupported device for pooling: device={x.device}")


class MaxPool2dFn(Function):
    """
    Autograd-enabled 2D max pooling operation (CPU backend).

    The forward pass computes max pooling over the spatial dimensions (H, W)
    and stores argmax metadata required to route gradients during backward.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": input shape (N, C, H, W)
        - "kernel_size": (k_h, k_w)
        - "stride": (s_h, s_w)
        - "padding": (p_h, p_w)
        - "argmax_idx": backend-defined indices from forward
    """

    @staticmethod
    def forward(
        ctx: Context, x: Tensor, *, kernel_size, stride=None, padding=0
    ) -> Tensor:
        """
        Compute 2D max pooling and save metadata for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for storing tensors/metadata required by backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).
        kernel_size : int | tuple[int, int]
            Pooling window size.
        stride : int | tuple[int, int] | None, optional
            Stride between pooling windows. If None, defaults to `kernel_size`.
        padding : int | tuple[int, int], optional
            Zero-padding applied to spatial dimensions before pooling.

        Returns
        -------
        Tensor
            Output tensor of pooled values.

        Notes
        -----
        The returned tensor's `requires_grad` mirrors `x.requires_grad` to preserve
        legacy behavior in the current codebase.
        """
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)

        backend = _dispatch_backend(x)
        if backend == "cpu":
            y, argmax_idx = cpu_ext.maxpool2d_forward(
                x, kernel_size=k, stride=s, padding=p
            )
        else:
            # CUDA: argmax_idx is a DevPtr (int) for int64 indices on device
            y, argmax_idx = cuda_ext.maxpool2d_forward(
                x, kernel_size=k, stride=s, padding=p
            )

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape
        ctx.saved_meta["kernel_size"] = k
        ctx.saved_meta["stride"] = s
        ctx.saved_meta["padding"] = p
        ctx.saved_meta["argmax_idx"] = argmax_idx

        # keep legacy requires_grad behavior
        y.requires_grad = x.requires_grad
        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through 2D max pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the pooled output.

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` matches the input
            shape, or `(None,)` if the input does not require gradients.
        """
        (x,) = ctx.saved_tensors
        if not x.requires_grad:
            return (None,)

        backend = _dispatch_backend(x)
        if backend == "cpu":
            gx = cpu_ext.maxpool2d_backward(
                grad_out,
                argmax_idx=ctx.saved_meta["argmax_idx"],
                x_shape=ctx.saved_meta["x_shape"],
                kernel_size=ctx.saved_meta["kernel_size"],
                stride=ctx.saved_meta["stride"],
                padding=ctx.saved_meta["padding"],
            )
        else:
            gx = cuda_ext.maxpool2d_backward(
                grad_out,
                argmax_idx=ctx.saved_meta["argmax_idx"],
                x_shape=ctx.saved_meta["x_shape"],
                kernel_size=ctx.saved_meta["kernel_size"],
                stride=ctx.saved_meta["stride"],
                padding=ctx.saved_meta["padding"],
            )
        return (gx,)


class AvgPool2dFn(Function):
    """
    Autograd-enabled 2D average pooling operation (CPU backend).

    The forward pass performs windowed average pooling over the spatial
    dimensions. The backward pass distributes gradients uniformly over each
    pooling window according to the forward configuration.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": input shape (N, C, H, W)
        - "kernel_size": (k_h, k_w)
        - "stride": (s_h, s_w)
        - "padding": (p_h, p_w)

    Notes
    -----
    - Assumes NCHW layout.
    - CPU-only: relies on NumPy kernels under `ops.pool2d_cpu_ext`.
    - Preserves legacy `requires_grad` behavior by mirroring from `x`.
    """

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        *,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
    ) -> Tensor:
        """
        Compute 2D average pooling and save metadata for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for storing tensors/metadata required by backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).
        kernel_size : int | tuple[int, int]
            Pooling window size.
        stride : int | tuple[int, int] | None, optional
            Stride between pooling windows. If None, defaults to `kernel_size`.
        padding : int | tuple[int, int], optional
            Zero-padding applied to spatial dimensions before pooling.

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out).
        """
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)

        backend = _dispatch_backend(x)
        if backend == "cpu":
            y = cpu_ext.avgpool2d_forward(x, kernel_size=k, stride=s, padding=p)
        else:
            y = cuda_ext.avgpool2d_forward(x, kernel_size=k, stride=s, padding=p)

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape
        ctx.saved_meta["kernel_size"] = k
        ctx.saved_meta["stride"] = s
        ctx.saved_meta["padding"] = p

        # Preserve legacy behavior: output requires_grad mirrors input grad participation
        y.requires_grad = x.requires_grad
        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through 2D average pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the output, shape (N, C, H_out, W_out).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has shape
            (N, C, H, W), or `(None,)` if the input does not require gradients.
        """
        (x,) = ctx.saved_tensors
        if not x.requires_grad:
            return (None,)

        backend = _dispatch_backend(x)
        if backend == "cpu":
            gx = cpu_ext.avgpool2d_backward(
                grad_out,
                x_shape=ctx.saved_meta["x_shape"],
                kernel_size=ctx.saved_meta["kernel_size"],
                stride=ctx.saved_meta["stride"],
                padding=ctx.saved_meta["padding"],
            )
        else:
            gx = cuda_ext.avgpool2d_backward(
                grad_out,
                x_shape=ctx.saved_meta["x_shape"],
                kernel_size=ctx.saved_meta["kernel_size"],
                stride=ctx.saved_meta["stride"],
                padding=ctx.saved_meta["padding"],
            )
        return (gx,)


class GlobalAvgPool2dFn(Function):
    """
    Autograd-enabled global average pooling (GAP) over 2D spatial dims (CPU backend).

    Global average pooling reduces the spatial dimensions by averaging:

        (N, C, H, W) -> (N, C, 1, 1)

    The backward pass distributes gradients uniformly across all H*W positions
    per channel.

    Saved context
    -------------
    - `saved_tensors`: [x]
    - `saved_meta`:
        - "x_shape": input shape (N, C, H, W)

    Notes
    -----
    - Assumes NCHW layout.
    - CPU-only: relies on NumPy kernels under `ops.pool2d_cpu_ext`.
    - Preserves legacy `requires_grad` behavior by mirroring from `x`.
    """

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> Tensor:
        """
        Compute global average pooling output and save input shape.

        Parameters
        ----------
        ctx : Context
            Autograd context for storing tensors/metadata required by backward.
        x : Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, 1, 1).
        """
        backend = _dispatch_backend(x)
        if backend == "cpu":
            y = cpu_ext.global_avgpool2d_forward(x)
        else:
            y = cuda_ext.global_avgpool2d_forward(x)

        ctx.save_for_backward(x)
        ctx.saved_meta["x_shape"] = x.shape

        # Preserve legacy behavior: output requires_grad mirrors input grad participation
        y.requires_grad = x.requires_grad
        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Backpropagate gradients through global average pooling.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient with respect to the output, shape (N, C, 1, 1).

        Returns
        -------
        Sequence[Optional[Tensor]]
            A single-element tuple `(grad_x,)` where `grad_x` has shape
            (N, C, H, W), or `(None,)` if the input does not require gradients.
        """
        (x,) = ctx.saved_tensors
        if not x.requires_grad:
            return (None,)

        backend = _dispatch_backend(x)
        if backend == "cpu":
            gx = cpu_ext.global_avgpool2d_backward(
                grad_out,
                x_shape=ctx.saved_meta["x_shape"],
            )
        else:
            gx = cuda_ext.global_avgpool2d_backward(
                grad_out,
                x_shape=ctx.saved_meta["x_shape"],
            )
        return (gx,)
