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

from __future__ import annotations

from typing import Optional, Tuple, Any, Dict

from ..tensor._tensor_context import Context

from ..module._serialization_core import register_module
from ...domain.device._device import Device
from ._conv2d_function import Conv2dFn
from ..tensor._tensor import Tensor
from .._parameter import Parameter
from .._module import Module

from ..ops.conv2d_cpu import _pair
from ..ops._initializers_cpu import (
    kaiming_normal_conv2d_weight,
    param_zeros,
)


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
        Device on which parameters will be allocated.
    dtype : Any, optional
        Data type used to initialize parameters. Kept for backward compatibility.
        This module does not interpret dtype directly; the CPU initializer
        boundary normalizes it.

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
        dtype: Any = None,
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

        # He (Kaiming) initialization is performed in the CPU initializer boundary.
        self.weight: Parameter = kaiming_normal_conv2d_weight(
            out_channels=self.out_channels,
            in_channels=self.in_channels,
            k_h=k_h,
            k_w=k_w,
            device=device,
            dtype=dtype,
        )
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias: Optional[Parameter] = param_zeros(
                (self.out_channels,),
                device=device,
                dtype=dtype,
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

        # Preserve autograd attachment semantics
        if Tensor._result_requires_grad(*parents):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out

    # -------------------------------------------------------------------------
    # ADD-ON ONLY: JSON serialization hooks (do not change existing logic above)
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this layer.

        Notes
        -----
        This configuration captures constructor-level hyperparameters only.
        Trainable parameters (weights/bias) are serialized separately by the
        checkpoint/state_dict mechanism.
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
            # Keep dtype JSON-safe without importing NumPy here
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

        # Accept None | "cpu" | "cuda:0" | Device-like
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
