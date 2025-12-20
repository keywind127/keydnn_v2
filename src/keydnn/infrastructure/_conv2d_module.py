"""
Conv2D module implementation for KeyDNN.

This module defines a trainable 2D convolution layer (`Conv2d`) built on top of
the KeyDNN autograd system. It serves as the high-level, user-facing abstraction
that owns convolution parameters (weights and optional bias) and delegates
numerical computation to the lower-level `Conv2dFn` and CPU convolution kernels.

Design overview
---------------
- `Conv2d` is a subclass of `Module` and represents a stateful neural network
  layer with learnable parameters.
- Numerical computation (forward and backward) is handled by `Conv2dFn`,
  keeping this module focused on parameter management and graph construction.
- Parameters are stored as `Parameter` objects, which are Tensor subclasses
  capable of accumulating gradients during backpropagation.

Current limitations
-------------------
- CPU-only implementation (NumPy backend).
- Supports only standard dense convolutions (no groups or dilation).
- Assumes NCHW tensor layout.
- Performance is not optimized; this implementation prioritizes correctness
  and architectural clarity.

Intended usage
--------------
This module is intended to be composed with other `Module` instances (e.g.,
activation layers, pooling layers, linear layers) to form convolutional neural
networks.

Example
-------
>>> conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, device=cpu)
>>> y = conv(x)
"""

import numpy as np
from typing import Optional, Tuple

from ._module import Module
from ._tensor import Tensor, Context
from ._parameter import Parameter
from .ops.conv2d_cpu import _pair
from ._function import Conv2dFn


def _param_from_numpy(arr: np.ndarray, *, device) -> Parameter:
    """
    Construct a trainable `Parameter` from a NumPy array.

    This helper initializes a `Parameter` instance using the array's shape
    and copies the provided NumPy data into the parameter's underlying
    tensor storage.

    Parameters
    ----------
    arr : np.ndarray
        NumPy array containing the initial parameter values.
    device : Device
        Target device on which the parameter tensor will be allocated.

    Returns
    -------
    Parameter
        A trainable parameter initialized with the given NumPy data.

    Notes
    -----
    - `Parameter` is a subclass of `Tensor` in this codebase.
    - The created parameter always has `requires_grad=True`.
    - This helper encapsulates the Tensor construction + data copy pattern
      used throughout layer initialization code.
    """
    p = Parameter(shape=arr.shape, device=device, requires_grad=True, ctx=None)
    p.copy_from_numpy(arr)
    return p


class Conv2d(Module):
    """
    Two-dimensional convolution layer (NCHW).

    This module applies a 2D convolution over an input tensor using learnable
    weights and an optional bias term. It supports configurable kernel size,
    stride, and padding, and integrates fully with KeyDNN's autograd system.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel. If an integer is provided, the same
        value is used for both height and width.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Defaults to 1.
    padding : int or tuple[int, int], optional
        Zero-padding applied to the input. Defaults to 0.
    bias : bool, optional
        Whether to include a learnable bias term. Defaults to True.
    device : Device, optional
        Device on which parameters will be allocated.
    dtype : np.dtype, optional
        Data type used to initialize parameters. Defaults to np.float32.

    Attributes
    ----------
    weight : Parameter
        Convolution kernel weights of shape
        (out_channels, in_channels, kernel_height, kernel_width).
    bias : Optional[Parameter]
        Optional bias parameter of shape (out_channels,).
    stride : tuple[int, int]
        Convolution stride as a 2D pair.
    padding : tuple[int, int]
        Convolution padding as a 2D pair.

    Notes
    -----
    - Weight parameters are initialized using He (Kaiming) initialization,
      which is suitable for ReLU-like nonlinearities.
    - This module does not perform any computation directly; it delegates
      forward and backward logic to `Conv2dFn`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        *,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        bias: bool = True,
        device=None,
        dtype=np.float32,
    ):
        """
        Initialize a Conv2d layer and its parameters.

        This constructor allocates and initializes the convolution kernel
        weights and optional bias parameter, and stores convolution
        hyperparameters for use during the forward pass.
        """
        super().__init__()

        k_h, k_w = _pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (k_h, k_w)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # He (Kaiming) initialization
        fan_in = self.in_channels * k_h * k_w
        scale = np.sqrt(2.0 / fan_in)
        w_np = (
            np.random.randn(out_channels, in_channels, k_h, k_w).astype(dtype)
        ) * scale
        self.weight: Parameter = _param_from_numpy(w_np, device=device)

        if bias:
            b_np = np.zeros((out_channels,), dtype=dtype)
            self.bias: Optional[Parameter] = _param_from_numpy(b_np, device=device)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the convolution operation to an input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C_in, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C_out, H_out, W_out).

        Notes
        -----
        - If any of the inputs or parameters require gradients, an autograd
          `Context` is attached to the output tensor.
        - The backward function delegates gradient computation to `Conv2dFn`.
        - No validation of input shape is performed here; mismatches are
          expected to be caught by lower-level kernels.
        """
        if self.bias is None:
            parents = (x, self.weight)
        else:
            parents = (x, self.weight, self.bias)

        ctx = Context(
            parents=parents,
            backward_fn=lambda grad_out: Conv2dFn.backward(ctx, grad_out),
        )

        out = Conv2dFn.forward(
            ctx,
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        if Tensor._result_requires_grad(*parents):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out
