from ._tensor import Tensor, Context
from ._module import Module

from ._function import SigmoidFn, ReLUFn, LeakyReLUFn, TanhFn


class Sigmoid(Module):
    """
    Sigmoid activation module.
    """

    def forward(self, x: Tensor) -> Tensor:
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (SigmoidFn.backward(ctx, grad_out),),
        )
        out = SigmoidFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


class ReLU(Module):
    """
    ReLU activation module.
    """

    def forward(self, x: Tensor) -> Tensor:
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (ReLUFn.backward(ctx, grad_out),),
        )
        out = ReLUFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


class LeakyReLU(Module):
    """
    Leaky ReLU activation module.

    Parameters
    ----------
    alpha : float, default=0.01
        Slope for negative inputs.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (LeakyReLUFn.backward(ctx, grad_out),),
        )
        out = LeakyReLUFn.forward(ctx, x, alpha=self.alpha)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


class Tanh(Module):
    """
    Hyperbolic tangent activation module.
    """

    def forward(self, x: Tensor) -> Tensor:
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (TanhFn.backward(ctx, grad_out),),
        )
        out = TanhFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out
