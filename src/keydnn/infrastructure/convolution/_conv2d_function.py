"""
Autograd-enabled Conv2D primitive (CPU backend).

This module defines `Conv2dFn`, a differentiable 2D convolution operator that
implements the domain-level `Function` contract for KeyDNN’s autograd engine.

Responsibilities and boundaries
--------------------------------
- `Conv2dFn` contains **no NumPy code** and performs no direct array operations.
- All Tensor ↔ NumPy bridging and CPU kernel execution are delegated to
  boundary helpers in `conv2d_cpu_ext`.
- This keeps the autograd logic (graph wiring, saved tensors, gradient routing)
  cleanly separated from backend-specific implementations.

Layout and semantics
--------------------
- Input layout: **NCHW** (batch, channels, height, width)
- Weight layout: implementation-defined but consistent with the CPU kernels
- Bias: optional, added per-output-channel
- Stride and padding accept either an int or a 2-tuple; both are normalized to
  `(h, w)` pairs internally.

Autograd behavior
-----------------
- During `forward`, required tensors and metadata are saved into `Context`.
- During `backward`, gradients are computed via the CPU boundary helper and
  conditionally returned based on each input’s `requires_grad` flag.
"""

from typing import Sequence, Optional, Tuple

from ..tensor._tensor_context import Context

from ...domain._function import Function
from ..tensor._tensor import Tensor
from ..ops.conv2d_cpu import _pair
from ..ops.conv2d_cpu_ext import (
    conv2d_backward_cpu_tensor,
    conv2d_forward_cpu_tensor,
)
from ..ops.conv2d_cuda_ext import (
    conv2d_backward_cuda_tensor,
    conv2d_forward_cuda_tensor,
)


def _cuda_device_index(dev: object) -> int:
    """
    Best-effort: extract CUDA device index from a Device-like object.

    Prefers `dev.index` when present; otherwise parses string forms like "cuda:0".
    """
    idx = getattr(dev, "index", None)
    if idx is not None:
        try:
            return int(idx)
        except Exception:
            pass

    s = str(dev)
    if "cuda" not in s:
        return 0
    if ":" in s:
        tail = s.split(":", 1)[1].strip()
        try:
            return int(tail)
        except Exception:
            return 0
    return 0


class Conv2dFn(Function):
    """
    Autograd-enabled 2D convolution primitive (NCHW, CPU-only).

    `Conv2dFn` defines the forward and backward passes for a 2D convolution
    operation within KeyDNN’s autograd system. It is a *pure autograd wrapper*:
    all numerical computation is delegated to CPU backend helpers.

    Design notes
    ------------
    - Implements the `Function` interface using static `forward` and `backward`
      methods.
    - Saves only the minimal tensors and metadata required to compute gradients.
    - Does not validate shapes or layouts beyond what the backend kernels expect;
      higher-level modules are responsible for input validation.
    """

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        *,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
    ) -> Tensor:
        """
        Perform the forward pass of a 2D convolution.

        This method normalizes stride/padding arguments, determines whether the
        output requires gradients, and invokes the CPU boundary helper to compute
        the convolution result.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors and metadata for the backward pass.
        x : Tensor
            Input tensor in NCHW layout.
        weight : Tensor
            Convolution kernel tensor.
        bias : Optional[Tensor]
            Optional bias tensor added to each output channel.
        stride : int or Tuple[int, int], optional
            Convolution stride. An integer is expanded to (stride, stride).
            Defaults to 1.
        padding : int or Tuple[int, int], optional
            Zero-padding size. An integer is expanded to (padding, padding).
            Defaults to 0.

        Returns
        -------
        Tensor
            Output tensor resulting from the convolution.

        Notes
        -----
        - This method does not attach a backward context itself; the returned
          tensor carries `requires_grad` and is linked to `ctx` by the autograd
          engine.
        - NumPy arrays are never accessed directly here; all backend work is
          handled by `conv2d_forward_cpu_tensor`.
        """
        stride2 = _pair(stride)
        padding2 = _pair(padding)

        # Require all tensors on the same device (no implicit cross-device copy here)
        if str(x.device) != str(weight.device):
            raise RuntimeError(
                f"conv2d requires x and weight on the same device; got x={x.device} weight={weight.device}"
            )
        if bias is not None and str(bias.device) != str(x.device):
            raise RuntimeError(
                f"conv2d requires bias on the same device as x; got bias={bias.device} x={x.device}"
            )

        out_req = Tensor._result_requires_grad(x, weight) or (
            bias is not None and bias.requires_grad
        )

        # Dispatch by device
        if x.device.is_cuda():
            device_index = _cuda_device_index(x.device)
            y = conv2d_forward_cuda_tensor(
                x,
                weight,
                bias,
                stride=stride2,
                padding=padding2,
                out_requires_grad=out_req,
                device_index=int(device_index),
                sync=True,
            )
        elif x.device.is_cpu():
            y = conv2d_forward_cpu_tensor(
                x,
                weight,
                bias,
                stride=stride2,
                padding=padding2,
                out_requires_grad=out_req,
            )
        else:
            x._raise_device_not_supported("conv2d_forward")
            raise RuntimeError("Unreachable")

        # Save tensors + meta for backward
        ctx.save_for_backward(x, weight)
        if bias is not None:
            ctx.save_for_backward(bias)

        ctx.saved_meta["has_bias"] = bias is not None
        ctx.saved_meta["stride"] = stride2
        ctx.saved_meta["padding"] = padding2
        ctx.saved_meta["is_cuda"] = bool(x.device.is_cuda())

        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Compute gradients for the Conv2D operation.

        This method retrieves saved tensors and metadata from the forward pass,
        invokes the CPU boundary helper to compute gradients, and returns them in
        the order expected by the autograd engine.

        Parameters
        ----------
        ctx : Context
            Autograd context containing saved tensors and metadata from `forward`.
        grad_out : Tensor
            Gradient with respect to the output of the convolution
            (dL / dY), in NCHW layout.

        Returns
        -------
        Sequence[Optional[Tensor]]
            Gradients corresponding to the forward inputs:
            - grad_x : Optional[Tensor]
                Gradient w.r.t. `x`, or None if `x.requires_grad` is False.
            - grad_w : Optional[Tensor]
                Gradient w.r.t. `weight`, or None if `weight.requires_grad` is False.
            - grad_b : Optional[Tensor]
                Gradient w.r.t. `bias`, or None if bias is absent or does not
                require gradients.

            If `bias` was not provided in the forward pass, the returned sequence
            omits `grad_b`.

        Notes
        -----
        - Returned gradient tensors have `requires_grad=False`.
        - Gradient accumulation into leaf tensors is handled by the autograd engine,
          not by this function.
        """
        has_bias: bool = bool(ctx.saved_meta["has_bias"])
        stride: Tuple[int, int] = ctx.saved_meta["stride"]
        padding: Tuple[int, int] = ctx.saved_meta["padding"]
        is_cuda: bool = bool(ctx.saved_meta.get("is_cuda", False))

        x = ctx.saved_tensors[0]
        weight = ctx.saved_tensors[1]
        bias = ctx.saved_tensors[2] if has_bias else None

        # Enforce grad_out device matches forward device
        if str(grad_out.device) != str(x.device):
            raise RuntimeError(
                f"grad_out must be on the same device as forward inputs; got grad_out={grad_out.device} x={x.device}"
            )

        if is_cuda:
            device_index = _cuda_device_index(x.device)
            gx, gw, gb = conv2d_backward_cuda_tensor(
                x,
                weight,
                bias,
                grad_out,
                stride=stride,
                padding=padding,
                device_index=int(device_index),
                sync=True,
            )
        else:
            gx, gw, gb = conv2d_backward_cpu_tensor(
                x,
                weight,
                bias,
                grad_out,
                stride=stride,
                padding=padding,
            )

        grad_x = gx if x.requires_grad else None
        grad_w = gw if weight.requires_grad else None
        grad_b = gb if (bias is not None and bias.requires_grad) else None

        if has_bias:
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)
