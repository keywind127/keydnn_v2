"""
Core activation and elementwise function implementations.

This module contains infrastructure-level implementations of common
differentiable functions used in neural networks, expressed in a
function-style autograd API:

- Each differentiable operation is implemented as a `Function` subclass
  with `forward(ctx, ...)` and `backward(ctx, grad_out)` static methods.
- A `Context` instance is used to store tensors and metadata required for
  the backward computation (`save_for_backward`, `saved_meta`).
- Public functional wrappers (e.g., `exp`) are responsible for:
  - validating inputs,
  - constructing the `Context` and wiring `backward_fn`,
  - invoking `forward`,
  - attaching the context to outputs when gradients are required.

Notes
-----
- All computations are CPU-only for now and rely on NumPy via `Tensor.to_numpy()`
  and `Tensor.copy_from_numpy()`.
- These functions assume elementwise semantics and do not implement broadcasting
  beyond what the underlying tensor ops support.
- Some functions (e.g., `SigmoidFn.forward`) are composed from other primitive
  ops (e.g., `exp`) to reuse autograd wiring.
"""

import numpy as np

from ..domain._function import Function
from ._tensor import Tensor, Context


class ExpFn(Function):
    """
    Elementwise exponential function.

    Implements:

        out = exp(x)

    Backward:

        d(exp(x))/dx = exp(x) = out

    Notes
    -----
    This implementation saves the output tensor in the context to reuse it in
    the backward pass.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute the elementwise exponential of `x`.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors for backward.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing `exp(x)` elementwise.
        """
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
        """
        Compute the gradient of the loss with respect to `x`.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors from the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the output of exp(x).

        Returns
        -------
        Tensor
            Gradient of the loss with respect to the input `x`.
        """
        (out,) = ctx.saved_tensors
        return grad_out * out


def exp(x: Tensor) -> Tensor:
    """
    Compute the elementwise exponential of a tensor with autograd support.

    This is the public functional wrapper around `ExpFn`. It constructs the
    `Context`, wires the backward function, invokes `ExpFn.forward`, and
    attaches the context to the output when gradients are required.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor `exp(x)`.

    Raises
    ------
    TypeError
        If `x` is not an infrastructure `Tensor`.
    """
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
    """
    Sigmoid activation function.

    Implements:

        sigmoid(x) = 1 / (1 + exp(-x))

    Backward:

        d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))

    Notes
    -----
    The forward pass is expressed using existing primitive ops (`exp`, negation,
    addition, division) so the computation graph is built compositionally.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute the sigmoid activation.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors for backward.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing sigmoid(x) elementwise.
        """
        out = 1 / (1 + exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient of the loss with respect to the input `x`.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors from the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the sigmoid output.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to the input `x`.
        """
        (out,) = ctx.saved_tensors
        return grad_out * out * (1 - out)


class ReLUFn(Function):
    """
    ReLU activation function.

    Implements:

        relu(x) = max(0, x)

    Backward:

        d(relu)/dx = 1 if x > 0 else 0

    Notes
    -----
    The forward pass uses a mask computed via `x > 0`, saved for backward.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute the ReLU activation.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors for backward.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing relu(x) elementwise.
        """
        mask = x > 0
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient of the loss with respect to the input `x`.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors from the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the ReLU output.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to the input `x`.
        """
        (mask,) = ctx.saved_tensors
        return grad_out * mask


class LeakyReLUFn(Function):
    """
    Leaky ReLU activation function.

    Implements:

        f(x) = x               if x > 0
             = alpha * x       otherwise

    Parameters
    ----------
    alpha : float, optional
        Slope for negative inputs. Defaults to 0.01.

    Notes
    -----
    The forward pass saves both positive and negative masks, and stores `alpha`
    in `ctx.saved_meta` for use during backward.
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float = 0.01) -> Tensor:
        """
        Compute the Leaky ReLU activation.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors for backward.
        x : Tensor
            Input tensor.
        alpha : float, optional
            Negative slope coefficient.

        Returns
        -------
        Tensor
            Output tensor containing leaky_relu(x) elementwise.
        """
        pos_mask = x > 0
        neg_mask = 1 - pos_mask

        out = x * pos_mask + x * neg_mask * alpha

        # Save masks and alpha for backward
        ctx.save_for_backward(pos_mask, neg_mask)
        ctx.saved_meta["alpha"] = alpha

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient of the loss with respect to the input `x`.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors and metadata from the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the LeakyReLU output.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to the input `x`.
        """
        pos_mask, neg_mask = ctx.saved_tensors
        alpha = ctx.saved_meta["alpha"]

        grad_x = grad_out * (pos_mask + neg_mask * alpha)
        return grad_x


class TanhFn(Function):
    """
    Hyperbolic tangent activation function.

    Implements tanh using an exp-based identity:

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Backward:

        d(tanh)/dx = 1 - tanh(x)^2

    Notes
    -----
    The forward pass saves the output `tanh(x)` for an efficient backward
    computation.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute the hyperbolic tangent activation.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors for backward.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing tanh(x) elementwise.
        """
        # Using exp-based definition (since exp is already implemented)
        e_pos = exp(x)
        e_neg = exp(-x)

        out = (e_pos - e_neg) / (e_pos + e_neg)

        # Save output for backward: d/dx = 1 - tanh(x)^2
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient of the loss with respect to the input `x`.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors from the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the tanh output.

        Returns
        -------
        Tensor
            Gradient of the loss with respect to the input `x`.
        """
        (out,) = ctx.saved_tensors
        return grad_out * (1 - out * out)
