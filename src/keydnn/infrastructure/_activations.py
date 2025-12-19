from ._function import Function, exp
from ._tensor import Tensor


import numpy as np

from ..domain._function import Function
from ._tensor import Tensor


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_np = x.to_numpy()
        out_np = 1.0 / (1.0 + np.exp(-x_np))

        out = Tensor(shape=x.shape, device=x.device, requires_grad=x.requires_grad)
        out.copy_from_numpy(out_np)

        # Save output for backward: dσ/dx = σ(x)(1-σ(x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors

        out_np = out.to_numpy()
        grad_np = grad_out.to_numpy() * out_np * (1.0 - out_np)

        grad_x = Tensor(shape=out.shape, device=out.device, requires_grad=False)
        grad_x.copy_from_numpy(grad_np)
        return grad_x


class ReLU(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_np = x.to_numpy()
        mask_np = (x_np > 0).astype(np.float32)
        out_np = x_np * mask_np

        out = Tensor(shape=x.shape, device=x.device, requires_grad=x.requires_grad)
        out.copy_from_numpy(out_np)

        # Save mask tensor for backward
        mask = Tensor(shape=x.shape, device=x.device, requires_grad=False)
        mask.copy_from_numpy(mask_np)
        ctx.save_for_backward(mask)

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (mask,) = ctx.saved_tensors
        grad_np = grad_out.to_numpy() * mask.to_numpy()

        grad_x = Tensor(shape=mask.shape, device=mask.device, requires_grad=False)
        grad_x.copy_from_numpy(grad_np)
        return grad_x
