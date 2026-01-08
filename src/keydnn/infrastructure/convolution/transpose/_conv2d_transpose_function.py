"""
Autograd-enabled Conv2D-Transpose primitive (CPU backend).

This module defines `Conv2dTransposeFn`, a differentiable transposed 2D
convolution operator that implements the domain-level `Function` contract for
KeyDNN’s autograd engine.

Responsibilities and boundaries
--------------------------------
- `Conv2dTransposeFn` contains **no NumPy code** and performs no direct array ops.
- All Tensor ↔ NumPy bridging and backend kernel execution are delegated to
  boundary helpers in:
  - `conv2d_transpose_cpu_ext` (CPU)
  - `conv2d_transpose_cuda_ext` (CUDA)
- This keeps autograd logic (graph wiring, saved tensors, gradient routing)
  separated from backend-specific implementations.

Layout and semantics
--------------------
- Input layout:  **NCHW** (N, C_in, H_in, W_in)
- Weight layout: **IOHW** (C_in, C_out, K_h, K_w)
- Bias: optional, added per-output-channel
- Stride / padding / output_padding accept int or 2-tuple; normalized internally
  to `(h, w)`.

Autograd behavior
-----------------
- During `forward`, required tensors and metadata are saved into `Context`.
- During `backward`, gradients are computed via the appropriate backend boundary
  helper and conditionally returned based on each input’s `requires_grad` flag.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ...tensor._tensor_context import Context
from ....domain._function import Function
from ...tensor._tensor import Tensor

from ...ops.conv2d_cpu import _pair
from ...ops.conv2d_transpose_cpu_ext import (
    conv2d_transpose_forward_cpu_tensor,
    conv2d_transpose_backward_cpu_tensor,
)
from ...ops.conv2d_transpose_cuda_ext import (
    conv2d_transpose_forward_cuda_tensor,
    conv2d_transpose_backward_cuda_tensor,
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


class Conv2dTransposeFn(Function):
    """
    Autograd-enabled 2D transposed convolution primitive (NCHW, IOHW).

    `Conv2dTransposeFn` is a pure autograd wrapper:
    - it wires forward/backward into KeyDNN's autograd engine,
    - it delegates all numeric compute to backend boundary helpers (CPU/CUDA).
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
        output_padding: int | Tuple[int, int] = 0,
    ) -> Tensor:
        """
        Perform the forward pass of a 2D transposed convolution.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors and metadata for backward.
        x : Tensor
            Input tensor in NCHW layout.
        weight : Tensor
            Transposed-conv kernel tensor in IOHW layout (C_in, C_out, K_h, K_w).
        bias : Optional[Tensor]
            Optional bias tensor of shape (C_out,).
        stride : int or Tuple[int, int], optional
            Stride. Integer expands to (stride, stride). Defaults to 1.
        padding : int or Tuple[int, int], optional
            Padding. Integer expands to (padding, padding). Defaults to 0.
        output_padding : int or Tuple[int, int], optional
            Output padding (must be < stride per dim). Integer expands to
            (output_padding, output_padding). Defaults to 0.

        Returns
        -------
        Tensor
            Output tensor (N, C_out, H_out, W_out).

        Notes
        -----
        - No implicit cross-device copy is performed.
        - Backend dispatch is selected by `x.device` (CPU vs CUDA).
        """
        stride2 = _pair(stride)
        padding2 = _pair(padding)
        outpad2 = _pair(output_padding)

        # Require all tensors on the same device (no implicit cross-device copy here)
        if str(x.device) != str(weight.device):
            raise RuntimeError(
                "conv2d_transpose requires x and weight on the same device; "
                f"got x={x.device} weight={weight.device}"
            )
        if bias is not None and str(bias.device) != str(x.device):
            raise RuntimeError(
                "conv2d_transpose requires bias on the same device as x; "
                f"got bias={bias.device} x={x.device}"
            )

        out_req = Tensor._result_requires_grad(x, weight) or (
            bias is not None and bias.requires_grad
        )

        # Dispatch by device
        if x.device.is_cuda():
            device_index = _cuda_device_index(x.device)
            y = conv2d_transpose_forward_cuda_tensor(
                x,
                weight,
                bias,
                stride=stride2,
                padding=padding2,
                output_padding=outpad2,
                out_requires_grad=out_req,
                device_index=int(device_index),
                sync=True,
            )
        elif x.device.is_cpu():
            y = conv2d_transpose_forward_cpu_tensor(
                x,
                weight,
                bias,
                stride=stride2,
                padding=padding2,
                output_padding=outpad2,
                out_requires_grad=out_req,
            )
        else:
            x._raise_device_not_supported("conv2d_transpose_forward")
            raise RuntimeError("Unreachable")

        # Save tensors + meta for backward
        ctx.save_for_backward(x, weight)
        if bias is not None:
            ctx.save_for_backward(bias)

        ctx.saved_meta["has_bias"] = bias is not None
        ctx.saved_meta["stride"] = stride2
        ctx.saved_meta["padding"] = padding2
        ctx.saved_meta["output_padding"] = outpad2
        ctx.saved_meta["is_cuda"] = bool(x.device.is_cuda())

        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Sequence[Optional[Tensor]]:
        """
        Compute gradients for the Conv2D-Transpose operation.

        Parameters
        ----------
        ctx : Context
            Autograd context containing saved tensors and metadata from `forward`.
        grad_out : Tensor
            Gradient with respect to the output (dL/dY), in NCHW layout.

        Returns
        -------
        Sequence[Optional[Tensor]]
            Gradients corresponding to forward inputs:
            - grad_x : Optional[Tensor]
            - grad_w : Optional[Tensor]
            - grad_b : Optional[Tensor] (only if bias was provided)

        Notes
        -----
        - Returned gradient tensors have `requires_grad=False`.
        - Bias grad is returned only if bias existed in forward; it is None if
          bias did not require grad.
        """
        has_bias: bool = bool(ctx.saved_meta["has_bias"])
        stride: Tuple[int, int] = ctx.saved_meta["stride"]
        padding: Tuple[int, int] = ctx.saved_meta["padding"]
        output_padding: Tuple[int, int] = ctx.saved_meta["output_padding"]
        is_cuda: bool = bool(ctx.saved_meta.get("is_cuda", False))

        x = ctx.saved_tensors[0]
        weight = ctx.saved_tensors[1]
        bias = ctx.saved_tensors[2] if has_bias else None

        # Enforce grad_out device matches forward device
        if str(grad_out.device) != str(x.device):
            raise RuntimeError(
                "grad_out must be on the same device as forward inputs; "
                f"got grad_out={grad_out.device} x={x.device}"
            )

        if is_cuda:
            device_index = _cuda_device_index(x.device)
            gx, gw, gb = conv2d_transpose_backward_cuda_tensor(
                x,
                weight,
                bias,
                grad_out,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                device_index=int(device_index),
                sync=True,
            )
        else:
            gx, gw, gb = conv2d_transpose_backward_cpu_tensor(
                x,
                weight,
                bias,
                grad_out,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )

        grad_x = gx if x.requires_grad else None
        grad_w = gw if weight.requires_grad else None
        grad_b = gb if (bias is not None and bias.requires_grad) else None

        if has_bias:
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)
