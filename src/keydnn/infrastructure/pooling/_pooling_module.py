"""
Pooling modules for KeyDNN (2D, NCHW).

This module defines high-level `Module` wrappers around pooling `Function`s:

- `MaxPool2d`         : windowed max pooling with argmax-based backward routing
- `AvgPool2d`         : windowed average pooling with uniform gradient distribution
- `GlobalAvgPool2d`   : global average pooling reducing (H, W) -> (1, 1)

Each module is:
- **stateless** (no trainable parameters),
- **shape-preserving** for (N, C) while reducing spatial dimensions,
- **autograd-compatible** via `Context` and corresponding `*Fn` implementations.

Design notes
------------
- All pooling modules assume **NCHW layout**: (N, C, H, W).
- Hyperparameters are normalized to 2-tuples using `_pair`.
- Hyperparameters are stored in `Pool2dMeta` to keep modules lightweight and
  to provide a single source of truth for configuration.
- The `forward()` methods attach an autograd `Context` only when the input
  requires gradients (consistent with the rest of the codebase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.infrastructure.ops.pool2d_cpu import _pair
from src.keydnn.infrastructure.pooling._pooling_function import (
    MaxPool2dFn,
    AvgPool2dFn,
    GlobalAvgPool2dFn,
)


@dataclass(frozen=True)
class Pool2dMeta:
    """
    Immutable configuration container for 2D pooling hyperparameters.

    Attributes
    ----------
    kernel_size : tuple[int, int]
        Pooling window size (k_h, k_w).
    stride : tuple[int, int]
        Stride (s_h, s_w). If the stride is not explicitly provided by the user,
        implementations typically default it to `kernel_size`.
    padding : tuple[int, int]
        Spatial padding (p_h, p_w) applied to height and width dimensions.

    Notes
    -----
    This structure exists to:
    - keep pooling modules stateless and easy to inspect,
    - avoid accidental mutation of hyperparameters during training/inference,
    - centralize hyperparameter storage so properties can simply reference it.
    """

    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]


class MaxPool2d(Module):
    """
    2D max pooling module (NCHW).

    This module applies max pooling over the spatial dimensions (H, W)
    independently per channel. The output retains the batch and channel
    dimensions while reducing spatial resolution according to pooling
    hyperparameters.

    Shape semantics
    ---------------
    Input:
        x.shape == (N, C, H, W)

    Output:
        y.shape == (N, C, H_out, W_out)

    Notes
    -----
    - Backpropagation routes gradients to the input positions that produced the
      maxima during the forward pass (argmax-based routing).
    - The underlying CPU reference implementation pads with `-inf` so padded
      values never become maxima (important for correctness at borders).
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        *,
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
    ) -> None:
        """
        Construct a MaxPool2d module.

        Parameters
        ----------
        kernel_size : int or tuple[int, int]
            Pooling window size. If an int, it is expanded to (k, k).
        stride : int or tuple[int, int] or None, optional
            Pooling stride. If None, defaults to `kernel_size`.
        padding : int or tuple[int, int], optional
            Spatial padding applied before pooling.

        Notes
        -----
        Hyperparameters are normalized to 2-tuples and stored in `Pool2dMeta`.
        """
        super().__init__()
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)
        self._meta = Pool2dMeta(kernel_size=k, stride=s, padding=p)

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
        Return the pooling window size.

        Returns
        -------
        tuple[int, int]
            Kernel size as (k_h, k_w).
        """
        return self._meta.kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Return the pooling stride.

        Returns
        -------
        tuple[int, int]
            Stride as (s_h, s_w).
        """
        return self._meta.stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Return the pooling padding.

        Returns
        -------
        tuple[int, int]
            Padding as (p_h, p_w).
        """
        return self._meta.padding

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply max pooling to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out).

        Notes
        -----
        - A `Context` is attached only if `x.requires_grad` is True.
        - The module delegates computation to `MaxPool2dFn`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: MaxPool2dFn.backward(ctx, grad_out),
        )
        out = MaxPool2dFn.forward(
            ctx,
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)
        return out


class AvgPool2d(Module):
    """
    2D average pooling module (NCHW).

    This module applies average pooling over the spatial dimensions (H, W)
    independently per channel.

    Shape semantics
    ---------------
    Input:
        x.shape == (N, C, H, W)

    Output:
        y.shape == (N, C, H_out, W_out)

    Notes
    -----
    - The underlying reference implementation uses **zero-padding**.
    - The average is computed over the full kernel area (k_h * k_w), which means
      padded zeros contribute to the average when padding > 0.
    - The backward pass distributes gradients uniformly over each pooling window.
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        *,
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
    ) -> None:
        """
        Construct an AvgPool2d module.

        Parameters
        ----------
        kernel_size : int or tuple[int, int]
            Pooling window size. If an int, it is expanded to (k, k).
        stride : int or tuple[int, int] or None, optional
            Pooling stride. If None, defaults to `kernel_size`.
        padding : int or tuple[int, int], optional
            Spatial padding applied before pooling.

        Notes
        -----
        Hyperparameters are normalized to 2-tuples and stored in `Pool2dMeta`.
        """
        super().__init__()
        k = _pair(kernel_size)
        s = _pair(kernel_size if stride is None else stride)
        p = _pair(padding)
        self._meta = Pool2dMeta(kernel_size=k, stride=s, padding=p)

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
        Return the pooling window size.

        Returns
        -------
        tuple[int, int]
            Kernel size as (k_h, k_w).
        """
        return self._meta.kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Return the pooling stride.

        Returns
        -------
        tuple[int, int]
            Stride as (s_h, s_w).
        """
        return self._meta.stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Return the pooling padding.

        Returns
        -------
        tuple[int, int]
            Padding as (p_h, p_w).
        """
        return self._meta.padding

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply average pooling to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out).

        Notes
        -----
        - A `Context` is attached only if `x.requires_grad` is True.
        - The module delegates computation to `AvgPool2dFn`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: AvgPool2dFn.backward(ctx, grad_out),
        )
        out = AvgPool2dFn.forward(
            ctx,
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)
        return out


class GlobalAvgPool2d(Module):
    """
    Global average pooling module (NCHW).

    Global average pooling reduces each channel to a single value by averaging
    over the spatial dimensions:

        (N, C, H, W) -> (N, C, 1, 1)

    This is commonly used near the end of CNN architectures to eliminate
    fully-connected layers and support variable spatial input sizes.

    Notes
    -----
    - This module has no kernel/stride/padding hyperparameters.
    - The backward pass distributes gradients uniformly across all H*W input
      positions per channel.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global average pooling to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, 1, 1).

        Notes
        -----
        - A `Context` is attached only if `x.requires_grad` is True.
        - The module delegates computation to `GlobalAvgPool2dFn`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: GlobalAvgPool2dFn.backward(ctx, grad_out),
        )
        out = GlobalAvgPool2dFn.forward(ctx, x)
        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)
        return out
