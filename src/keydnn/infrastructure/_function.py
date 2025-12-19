import numpy as np


from ..domain._function import Function
from ._tensor import Tensor, Context


class ExpFn(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # Create output tensor
        out = Tensor(
            shape=x.shape,
            device=x.device,
            requires_grad=x.requires_grad,
        )

        # Forward computation (CPU only for now)
        out.copy_from_numpy(np.exp(x.to_numpy()))

        # Save for backward
        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        return grad_out * out


def exp(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError("exp expects a Tensor")

    ctx = Context(
        parents=(x,),
        backward_fn=lambda grad_out: (ExpFn.backward(ctx, grad_out),),
    )

    out = ExpFn.forward(ctx, x)

    if x.requires_grad:
        out._set_ctx(ctx)

    return out


class SigmoidFn(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        out = 1 / (1 + exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        return grad_out * out * (1 - out)


class ReLUFn(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        mask = x > 0
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (mask,) = ctx.saved_tensors
        return grad_out * mask


class LeakyReLUFn(Function):
    """
    Leaky ReLU activation function.

    f(x) = x               if x > 0
         = alpha * x       otherwise
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float = 0.01) -> Tensor:
        pos_mask = x > 0
        neg_mask = 1 - pos_mask

        out = x * pos_mask + x * neg_mask * alpha

        # Save masks and alpha for backward
        ctx.save_for_backward(pos_mask, neg_mask)
        ctx.saved_meta["alpha"] = alpha

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        pos_mask, neg_mask = ctx.saved_tensors
        alpha = ctx.saved_meta["alpha"]

        grad_x = grad_out * (pos_mask + neg_mask * alpha)
        return grad_x


class TanhFn(Function):
    """
    Hyperbolic tangent activation function.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # Using exp-based definition (since exp is already implemented)
        e_pos = exp(x)
        e_neg = exp(-x)

        out = (e_pos - e_neg) / (e_pos + e_neg)

        # Save output for backward: d/dx = 1 - tanh(x)^2
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        return grad_out * (1 - out * out)
