"""
Pooling layer interfaces for KeyDNN.

This module defines **domain-level Protocols** for 2D pooling operations,
covering both local (window-based) pooling and global pooling over spatial
dimensions.

These interfaces serve as *contracts* that concrete pooling implementations
in the infrastructure layer (e.g., MaxPool2d, AvgPool2d, GlobalAvgPool2d)
must satisfy.

Design principles
-----------------
- Pooling modules operate on **NCHW tensors**.
- Pooling modules are **stateless** (no trainable parameters).
- Pooling modules must integrate cleanly with the autograd system when used
  in differentiable models.
- Interfaces are expressed using `Protocol` to enable structural subtyping
  (duck typing) rather than inheritance coupling.

Notes
-----
This module contains **no NumPy or backend-specific logic** and is safe to
depend on from any layer of the architecture.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from ._tensor import ITensor
from ._module import IModule


@runtime_checkable
class IPooling2D(IModule, Protocol):
    """
    Protocol for 2D pooling modules operating on NCHW tensors.

    A 2D pooling module reduces spatial resolution by applying a fixed-size
    window over the input tensor's height and width dimensions.

    Shape semantics
    ---------------
    Input:
        x.shape == (N, C, H, W)

    Output:
        y.shape == (N, C, H_out, W_out)

    where:
        H_out = floor((H + 2*p_h - k_h) / s_h) + 1
        W_out = floor((W + 2*p_w - k_w) / s_w) + 1

    Design constraints
    ------------------
    - Pooling modules MUST NOT own trainable parameters.
    - Pooling modules SHOULD be pure functions of their inputs.
    - Pooling modules MUST preserve the batch and channel dimensions.
    - Pooling modules SHOULD be compatible with autograd for backpropagation.

    Examples
    --------
    Typical implementations include:
    - Max pooling
    - Average pooling
    """

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
        Return the pooling window size.

        Returns
        -------
        Tuple[int, int]
            Pooling kernel size as (k_h, k_w).
        """

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Return the pooling stride.

        Returns
        -------
        Tuple[int, int]
            Stride along height and width as (s_h, s_w).
        """

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Return the spatial padding applied before pooling.

        Returns
        -------
        Tuple[int, int]
            Zero-padding along height and width as (p_h, p_w).
        """

    def forward(self, x: ITensor) -> ITensor:
        """
        Apply pooling over the spatial dimensions of the input tensor.

        Parameters
        ----------
        x : ITensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        ITensor
            Output tensor of shape (N, C, H_out, W_out).

        Notes
        -----
        - The returned tensor may carry autograd context if `x.requires_grad`
          is True.
        - Pooling is applied independently per channel.
        """


@runtime_checkable
class IGlobalPooling2D(IModule, Protocol):
    """
    Protocol for global 2D pooling modules.

    Global pooling collapses the spatial dimensions (H, W) into a single value
    per channel, producing a fixed spatial output regardless of input size.

    Shape semantics
    ---------------
    Input:
        x.shape == (N, C, H, W)

    Output:
        y.shape == (N, C, 1, 1)

    Common use cases
    ----------------
    - Global Average Pooling (GAP) in CNN classifiers
    - Removing the need for fully connected layers
    - Enabling variable-sized spatial inputs
    """

    def forward(self, x: ITensor) -> ITensor:
        """
        Apply global pooling over spatial dimensions.

        Parameters
        ----------
        x : ITensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        ITensor
            Output tensor of shape (N, C, 1, 1).

        Notes
        -----
        - Global pooling operates independently per channel.
        - No kernel size, stride, or padding parameters are required.
        - The operation is typically differentiable and compatible with
          backpropagation.
        """
