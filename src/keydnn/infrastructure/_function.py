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

from typing import Tuple

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


class SoftmaxFn(Function):
    """
    Softmax operation over a specified tensor dimension.

    This function computes the softmax of the input tensor along the given
    axis, producing a probability distribution that sums to 1 along that
    dimension.

    Notes
    -----
    - The implementation uses a numerically stable formulation by subtracting
      the maximum value along the softmax axis prior to exponentiation.
    - This operation is currently implemented using NumPy to avoid requiring
      axis-aware reduction operations in the `Tensor` API.
    - The backward pass is implemented as a Jacobian–vector product (JVP),
      avoiding explicit construction of the full softmax Jacobian.
    """

    @staticmethod
    def forward(ctx: Context, x: Tensor, *, axis: int = -1) -> Tensor:
        """
        Compute the softmax of the input tensor along a specified axis.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store intermediate values for backward.
        x : Tensor
            Input tensor whose values will be normalized using softmax.
        axis : int, optional
            Dimension along which softmax is computed. Defaults to the last
            dimension (`-1`).

        Returns
        -------
        Tensor
            A tensor of the same shape as `x`, where values along the specified
            axis form a probability distribution (sum to 1).

        Raises
        ------
        ValueError
            If the specified axis is invalid for the input tensor shape.
        RuntimeError
            If called on a non-CPU tensor.

        Notes
        -----
        - The returned tensor does not require gradients by default; gradient
          tracking is handled by the enclosing `Module` wrapper when needed.
        - The output tensor is saved in the context for use during the backward
          pass.
        """
        if not x.device.is_cpu():
            x._raise_device_not_supported("softmax")

        x_np = x.to_numpy()

        # Normalize axis
        axis_ = axis if axis >= 0 else x_np.ndim + axis
        if axis_ < 0 or axis_ >= x_np.ndim:
            raise ValueError(f"Invalid softmax axis {axis} for ndim={x_np.ndim}")

        # Stable softmax
        x_shift = x_np - np.max(x_np, axis=axis_, keepdims=True)
        exp_x = np.exp(x_shift)
        y_np = exp_x / np.sum(exp_x, axis=axis_, keepdims=True)

        y = Tensor(shape=y_np.shape, device=x.device, requires_grad=False)
        y.copy_from_numpy(y_np.astype(np.float32))

        # Save output for backward (most convenient)
        ctx.save_for_backward(y)
        ctx.saved_meta["axis"] = axis_

        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor]:
        """
        Compute gradients of the softmax operation with respect to the input.

        Parameters
        ----------
        ctx : Context
            Autograd context populated during the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the softmax output. Must have
            the same shape as the forward output.

        Returns
        -------
        tuple[Tensor]
            A single-element tuple containing the gradient with respect to the
            input tensor `x`.

        Notes
        -----
        The gradient is computed using the softmax Jacobian–vector product:

            dx = y * (g - sum(g * y, axis))

        where:
        - `y` is the softmax output,
        - `g` is the upstream gradient (`grad_out`).

        This formulation avoids explicit construction of the full Jacobian and
        ensures that gradients along the softmax axis sum to zero.
        """
        (y,) = ctx.saved_tensors
        axis = int(ctx.saved_meta["axis"])

        if not grad_out.device.is_cpu():
            grad_out._raise_device_not_supported("softmax_backward")

        y_np = y.to_numpy()
        g_np = grad_out.to_numpy()

        # Jacobian-vector product for softmax:
        # dx = y * (g - sum(g*y, axis))
        dot = np.sum(g_np * y_np, axis=axis, keepdims=True)
        dx_np = y_np * (g_np - dot)

        dx = Tensor(shape=dx_np.shape, device=y.device, requires_grad=False)
        dx.copy_from_numpy(dx_np.astype(np.float32))
        return (dx,)
