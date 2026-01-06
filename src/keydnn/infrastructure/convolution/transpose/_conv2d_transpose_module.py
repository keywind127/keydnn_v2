"""
Conv2D Transpose (a.k.a. deconvolution) module implementation for KeyDNN.

This module defines a trainable 2D transposed convolution layer (`Conv2dTranspose`)
built on top of KeyDNN's autograd system.

Design overview
---------------
- `Conv2dTranspose` is a `Module` that owns learnable parameters (weight + optional bias).
- Numerical computation (forward/backward) is delegated to `Conv2dTransposeFn`,
  keeping this module focused on parameter management and graph construction.
- Parameters are stored as `Parameter` objects, which are Tensor subclasses capable
  of accumulating gradients.

Layout and semantics
--------------------
- Input layout: NCHW
- Weight layout: **(C_in, C_out, K_h, K_w)** (IOHW), matching KeyDNN's transpose-conv CPU kernels.
- Bias: optional, added per output channel
- Stride, padding, output_padding accept either an int or a 2-tuple and are normalized
  to `(h, w)` pairs.

Current limitations
-------------------
- CPU backend support depends on the underlying ops implementation.
- Groups/dilation are not supported.
- Assumes NCHW.

Example
-------
>>> deconv = Conv2dTranspose(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1)
>>> y = deconv(x)
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, Dict

from ...tensor._tensor_context import Context
from ...module._serialization_core import register_module
from ....domain.device._device import Device

from ._conv2d_transpose_function import Conv2dTransposeFn
from ...tensor._tensor import Tensor
from ..._parameter import Parameter
from ..._module import Module

from ...ops.conv2d_cpu import _pair
from ...ops._initializers_cpu import (
    kaiming_normal_conv2d_weight,
    param_zeros,
)


def _kaiming_normal_conv2d_transpose_weight(
    *,
    in_channels: int,
    out_channels: int,
    k_h: int,
    k_w: int,
    device=None,
    dtype: Any = None,
) -> Parameter:
    """
    Initialize Conv2D-Transpose weight with Kaiming normal, producing IOHW layout.

    Notes
    -----
    KeyDNN's existing conv2d initializer produces OIHW (C_out, C_in, K_h, K_w).
    Transposed-conv CPU kernels in this project use IOHW (C_in, C_out, K_h, K_w),
    so we reuse the conv2d initializer and transpose the first two axes.
    """
    # Create an OIHW weight using the existing initializer, then transpose -> IOHW.
    tmp = kaiming_normal_conv2d_weight(
        out_channels=int(out_channels),
        in_channels=int(in_channels),
        k_h=int(k_h),
        k_w=int(k_w),
        device=device,
        dtype=dtype,
    )

    w_np = tmp.to_numpy().transpose(1, 0, 2, 3)  # (C_in, C_out, K_h, K_w)

    w = param_zeros(w_np.shape, device=device, dtype=dtype)
    w.copy_from_numpy(w_np)
    return w


@register_module()
class Conv2dTranspose(Module):
    """
    Two-dimensional transposed convolution layer (NCHW).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel.
    stride : int or tuple[int, int], optional
        Stride of the transposed convolution. Defaults to 1.
    padding : int or tuple[int, int], optional
        Padding used by the transposed convolution. Defaults to 0.
    output_padding : int or tuple[int, int], optional
        Additional size added to one side of each output dimension. Defaults to 0.
        (Must satisfy output_padding[d] < stride[d] for the corresponding ops.)
    bias : bool, optional
        Whether to include a learnable bias term. Defaults to True.
    device : Device, optional
        Device on which parameters will be allocated.
    dtype : Any, optional
        Data type used to initialize parameters. Kept for backward compatibility.

    Attributes
    ----------
    weight : Parameter
        Kernel weights of shape (in_channels, out_channels, K_h, K_w).
    bias : Optional[Parameter]
        Optional bias parameter of shape (out_channels,).
    stride : tuple[int, int]
        Stride as a 2D pair.
    padding : tuple[int, int]
        Padding as a 2D pair.
    output_padding : tuple[int, int]
        Output padding as a 2D pair.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        *,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        output_padding: int | Tuple[int, int] = 0,
        bias: bool = True,
        device=None,
        dtype: Any = None,
    ):
        super().__init__()

        k_h, k_w = _pair(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(k_h), int(k_w))
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)

        # Weight layout for transpose-conv ops: (C_in, C_out, K_h, K_w)
        self.weight: Parameter = _kaiming_normal_conv2d_transpose_weight(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
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
        Apply the transposed convolution operation to an input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C_in, H_in, W_in).

        Returns
        -------
        Tensor
            Output tensor of shape (N, C_out, H_out, W_out).

        Notes
        -----
        - If any of the inputs or parameters require gradients, an autograd
          `Context` is attached to the output tensor.
        - The backward function delegates gradient computation to `Conv2dTransposeFn`.
        """
        if self.bias is None:
            parents = (x, self.weight)
        else:
            parents = (x, self.weight, self.bias)

        ctx = Context(
            parents=parents,
            backward_fn=lambda grad_out: Conv2dTransposeFn.backward(ctx, grad_out),
        )

        out = Conv2dTransposeFn.forward(
            ctx,
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        if Tensor._result_requires_grad(*parents):
            out.requires_grad = True
            out._set_ctx(ctx)

        return out

    # -------------------------------------------------------------------------
    # JSON serialization hooks
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        op_h, op_w = self.output_padding

        return {
            "in_channels": int(self.in_channels),
            "out_channels": int(self.out_channels),
            "kernel_size": [int(k_h), int(k_w)],
            "stride": [int(s_h), int(s_w)],
            "padding": [int(p_h), int(p_w)],
            "output_padding": [int(op_h), int(op_w)],
            "bias": bool(self.bias is not None),
            "device": str(self.weight.device),
            "dtype": "float32",
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Conv2dTranspose":
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
            stride=tuple(cfg.get("stride", (1, 1))),
            padding=tuple(cfg.get("padding", (0, 0))),
            output_padding=tuple(cfg.get("output_padding", (0, 0))),
            bias=bool(cfg.get("bias", True)),
            device=device,
            dtype=cfg.get("dtype", "float32"),
        )
