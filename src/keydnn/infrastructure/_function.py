import numpy as np


from ..domain._function import Function
from ._tensor import Tensor, Context


class Exp(Function):

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
        backward_fn=lambda grad_out: (Exp.backward(ctx, grad_out),),
    )

    out = Exp.forward(ctx, x)

    if x.requires_grad:
        out._set_ctx(ctx)

    return out
