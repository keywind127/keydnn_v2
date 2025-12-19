from ..domain._function import Function
from ._tensor import Tensor
from ._function import exp


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        out = 1 / (1 + exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        return grad_out * out * (1 - out)


class ReLU(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        mask = x > 0
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (mask,) = ctx.saved_tensors
        return grad_out * mask
