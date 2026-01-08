"""
Conv2D module implementation for KeyDNN.

This module defines a trainable 2D convolution layer (`Conv2d`) built on top of
the KeyDNN autograd system. It serves as the high-level, user-facing abstraction
that owns convolution parameters (weights and optional bias) and delegates
numerical computation to the lower-level `Conv2dFn` and convolution backends.

Design overview
---------------
- `Conv2d` is a subclass of `Module` and represents a stateful neural network
  layer with learnable parameters.
- Numerical computation (forward and backward) is handled by `Conv2dFn`,
  keeping this module focused on parameter management and graph construction.
- Parameters are stored as `Parameter` objects, which are Tensor subclasses
  capable of accumulating gradients during backpropagation.

Backend support
---------------
- Backend availability depends on the installed ops implementations
  (CPU and optional CUDA).
- Supports only standard dense convolutions (no groups or dilation).
- Assumes NCHW tensor layout.

Performance notes
-----------------
This implementation prioritizes correctness and architectural clarity over
raw performance.

Intended usage
--------------
This module is intended to be composed with other `Module` instances (e.g.,
activation layers, pooling layers, linear layers) to form convolutional neural
networks.

Example
-------
>>> conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
>>> y = conv(x)
"""

from __future__ import annotations
from typing import Optional, Tuple, Any, Dict

import numpy as np

from ...domain.device._device import Device

from ..tensor._tensor_context import Context
from ..module._serialization_core import register_module
from ..tensor._tensor import Tensor
from .._parameter import Parameter
from ..ops.conv2d_cpu import _pair
from .._module import Module

from ._conv2d_function import Conv2dFn


@register_module()
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
        Device on which parameters will be allocated. Defaults to CPU.
    dtype : Any, optional
        Data type used to initialize parameters. Defaults to float32 if not
        provided.
    initializer : str, optional
        Name of the weight initializer applied to the convolution kernel.
        Defaults to ``"kaiming"``. The bias parameter, if present, is
        initialized using the ``"zeros"`` initializer.

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
    - Weight initialization is performed via the `Parameter` initializer
      registry, not inside this module.
    - This module does not perform any numerical computation directly; it
      delegates forward and backward logic to `Conv2dFn`.
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
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        initializer: str = "kaiming",
    ):
        """
        Initialize a Conv2d layer and its parameters.

        This constructor allocates and initializes the convolution kernel
        weights and optional bias parameter, and stores convolution
        hyperparameters for use during the forward pass.

        Notes
        -----
        - If ``device`` is not provided, parameters are allocated on CPU.
        - If ``dtype`` is not provided, parameters default to float32.
        - The convolution kernel is initialized using the specified
          ``initializer``.
        - The bias parameter, if enabled, is initialized to zeros.
        """
        super().__init__()

        k_h, k_w = _pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (k_h, k_w)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        if device is None:
            device = Device("cpu")
        if dtype is None:
            dtype = np.float32

        # Weight layout: (C_out, C_in, K_h, K_w)
        self.weight = Parameter(
            shape=(self.out_channels, self.in_channels, int(k_h), int(k_w)),
            device=device,
            dtype=dtype,
            requires_grad=True,
            initializer=initializer,
        )
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias = Parameter(
                shape=(self.out_channels,),
                device=device,
                dtype=dtype,
                requires_grad=True,
                initializer="zeros",
            )
            self.register_parameter("bias", self.bias)
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

    # -------------------------------------------------------------------------
    # JSON serialization hooks
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this layer.

        Notes
        -----
        This configuration captures constructor-level hyperparameters only.
        Trainable parameters (weights and bias) are serialized separately by
        the checkpoint/state_dict mechanism.
        """
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        return {
            "in_channels": int(self.in_channels),
            "out_channels": int(self.out_channels),
            "kernel_size": [int(k_h), int(k_w)],
            "stride": [int(s_h), int(s_w)],
            "padding": [int(p_h), int(p_w)],
            "bias": bool(self.bias is not None),
            # Serialize device via parameter (authoritative source)
            "device": str(self.weight.device),
            # Serialize dtype in a JSON-safe form (currently fixed to float32)
            "dtype": "float32",
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Conv2d":
        """
        Construct a Conv2d layer from a configuration dict.

        Notes
        -----
        This reconstructs the module structure (hyperparameters). Weights are
        expected to be loaded afterward from the checkpoint state.
        """
        device_cfg = cfg.get("device", None)

        if device_cfg is None:
            device = None
        elif isinstance(device_cfg, str):
            device = Device(device_cfg)
        else:
            device = device_cfg

        return cls(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            kernel_size=tuple(cfg["kernel_size"]),
            stride=tuple(cfg["stride"]),
            padding=tuple(cfg["padding"]),
            bias=bool(cfg["bias"]),
            device=device,
            dtype=cfg.get("dtype", "float32"),
        )
