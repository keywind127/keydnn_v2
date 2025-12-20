"""
Flatten layer interface definitions.

This module defines the domain-level interface (Protocol) for flattening
layers in KeyDNN.

A flattening layer reshapes an input tensor by collapsing all non-batch
dimensions into a single feature dimension. It is commonly used to bridge
spatial layers (e.g., convolutional or pooling layers) and fully connected
layers in neural networks.

Design notes
------------
- Flatten layers are **stateless** and have no trainable parameters.
- They participate in automatic differentiation by wiring the appropriate
  backward function during the forward pass.
- Backward propagation is handled implicitly through the autograd engine
  via `Context`, not through an explicit `backward()` method on the module.
- The interface is backend-agnostic and does not depend on NumPy or CUDA.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from ._tensor import ITensor
from ._module import IModule


@runtime_checkable
class IFlatten(IModule, Protocol):
    """
    Protocol for flattening layers.

    A flattening layer reshapes an input tensor by collapsing all dimensions
    except the batch dimension into a single feature dimension.

    This layer is typically used after convolutional or pooling layers to
    prepare data for fully connected (dense) layers.

    Shape semantics
    ---------------
    Input:
        (N, d1, d2, ..., dk)

    Output:
        (N, d1 * d2 * ... * dk)

    Notes
    -----
    - The batch dimension (`N`) is preserved.
    - The flattening operation is purely a reshape; no data is copied
      conceptually, and no numerical transformation is applied.
    - Implementations should integrate with the autograd system so that
      gradients are reshaped back to the original input shape during
      backpropagation.
    """

    def forward(self, x: ITensor) -> ITensor:
        """
        Flatten the input tensor.

        This method collapses all dimensions of the input tensor except
        the first (batch) dimension into a single feature dimension.

        Parameters
        ----------
        x : ITensor
            Input tensor with batch dimension first, e.g. (N, C, H, W)
            or (N, D).

        Returns
        -------
        ITensor
            Flattened tensor of shape (N, -1), where `-1` is the product
            of all non-batch dimensions of the input tensor.
        """
