"""
Core activation and elementwise function implementations.

This module provides infrastructure-level implementations of common
differentiable elementwise functions used in neural networks, exposed through a
function-style autograd API.

Architecture
------------
Each differentiable operation is implemented as a `Function` subclass with:

- `forward(ctx, ...)`:
    Performs the forward computation and saves any tensors/metadata needed for
    backpropagation via `ctx.save_for_backward(...)` and `ctx.saved_meta`.

- `backward(ctx, grad_out)`:
    Computes gradients w.r.t. the forward inputs using the saved tensors and
    metadata, returning gradients in the same input order.

Public functional wrappers
--------------------------
Where present (e.g., `exp`), a public wrapper is responsible for:

- validating inputs and normalizing arguments,
- constructing a `Context` with `(parents=..., backward_fn=...)`,
- invoking the corresponding `Function.forward`,
- attaching the context to the output tensor when gradients are required.

Backend and semantic constraints
--------------------------------
- Current implementations target the CPU backend.
- Most operations are expressed via existing `Tensor` primitives (e.g., `+`, `-`,
  `*`, `/`, comparisons), allowing autograd graphs to be composed from smaller
  building blocks.
- Some ops (e.g., `SoftmaxFn`) rely on axis-aware tensor utilities and may
  require CPU-only pathways in the current backend.

Notes
-----
- Unless otherwise stated, these functions are elementwise and rely on the
  underlying `Tensor` operation semantics for shape compatibility.
- Several operations cache forward outputs (e.g., `exp(x)`, `sigmoid(x)`) to
  compute gradients efficiently in the backward pass.
"""

from typing import Tuple

from .tensor._tensor_context import Context

from .tensor._tensor import Tensor
from ..domain._function import Function


class ExpFn(Function):
    """
    Elementwise exponential primitive.

    Computes the elementwise exponential:

        out = exp(x)

    and defines the derivative:

        d(out)/dx = exp(x) = out

    Implementation details
    ----------------------
    The forward pass saves the output tensor `out` in the context so the backward
    pass can reuse it directly without recomputing `exp(x)`.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute `exp(x)` elementwise and save the output for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors required by the backward pass.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor containing `exp(x)` elementwise.
        """
        out = x.exp()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient with respect to the input of `exp`.

        Given out = exp(x), the gradient is:

            dL/dx = dL/dout * out

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing the saved output tensor.
        grad_out : Tensor
            Upstream gradient dL/dout.

        Returns
        -------
        Tensor
            Gradient dL/dx.
        """
        (out,) = ctx.saved_tensors
        return grad_out * out


def exp(x: Tensor) -> Tensor:
    """
    Functional wrapper for elementwise exponential with autograd wiring.

    This wrapper:
    - validates the input type,
    - constructs a `Context` with the correct parent set,
    - executes `ExpFn.forward`,
    - attaches the context to the output if gradients are required.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing `exp(x)` elementwise.

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
    Sigmoid activation primitive.

    Computes the sigmoid function:

        sigmoid(x) = 1 / (1 + exp(-x))

    with derivative:

        d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))

    Implementation details
    ----------------------
    The forward pass is expressed using existing primitives (negation, addition,
    division) and the `exp` wrapper. The output is saved for an efficient
    backward computation.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute `sigmoid(x)` elementwise and save the output for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors required by the backward pass.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor containing `sigmoid(x)` elementwise.
        """
        # Original: out = 1 / (1 + exp(-x))
        #
        # Optimized (allocation-reduced):
        #   t = exp(-x)          (alloc)
        #   t += 1               (in-place scalar add; avoids scalar-fill tensor)
        #   out = 1 / t          (alloc; relies on existing scalar / tensor path)
        #
        # NOTE: This is safe because `t` is a fresh temporary.
        t = exp(-x)

        # Prefer CUDA scalar in-place if available; otherwise fallback to expression.
        try:
            t += 1.0
        except Exception:
            # fallback keeps semantics
            t = t + 1.0

        out = 1 / t
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient with respect to the input of `sigmoid`.

        Given y = sigmoid(x), the gradient is:

            dL/dx = dL/dy * y * (1 - y)

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing the saved output tensor.
        grad_out : Tensor
            Upstream gradient dL/dy.

        Returns
        -------
        Tensor
            Gradient dL/dx.
        """
        (out,) = ctx.saved_tensors

        # tmp = grad_out * out
        tmp = grad_out * out

        # Build (1 - out) without scalar-lift tensor allocation:
        # one_minus = -out; one_minus += 1
        one_minus = -out
        try:
            one_minus += 1.0
        except Exception:
            one_minus = one_minus + 1.0

        # tmp *= (1 - out)
        tmp *= one_minus
        return tmp


class ReLUFn(Function):
    """
    ReLU (Rectified Linear Unit) activation primitive.

    Computes:

        relu(x) = max(0, x)

    with derivative:

        d(relu)/dx = 1  if x > 0
                   = 0  otherwise

    Implementation details
    ----------------------
    The forward pass constructs a mask via `(x > 0)`, saves it in the context,
    and multiplies the input by the mask.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute `relu(x)` elementwise and save the activation mask for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors required by the backward pass.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor containing `relu(x)` elementwise.
        """
        mask = x > 0
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient with respect to the input of `relu`.

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing the saved mask tensor.
        grad_out : Tensor
            Upstream gradient dL/d(relu(x)).

        Returns
        -------
        Tensor
            Gradient dL/dx.
        """
        (mask,) = ctx.saved_tensors
        return grad_out * mask


class LeakyReLUFn(Function):
    """
    Leaky ReLU activation primitive.

    Computes:

        f(x) = x          if x > 0
             = alpha * x  otherwise

    where `alpha` is a small positive slope for negative inputs.

    Implementation details
    ----------------------
    - The forward pass builds `pos_mask` and `neg_mask` using comparisons and
      saves both masks in the context.
    - The scalar `alpha` is stored in `ctx.saved_meta` for use during backward.
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float = 0.01) -> Tensor:
        """
        Compute `leaky_relu(x)` elementwise and save masks/alpha for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors/metadata required by backward.
        x : Tensor
            Input tensor.
        alpha : float, optional
            Negative slope coefficient. Defaults to 0.01.

        Returns
        -------
        Tensor
            Tensor containing `leaky_relu(x)` elementwise.
        """
        # Original:
        #   pos_mask = x > 0
        #   neg_mask = 1 - pos_mask
        #   out = x * pos_mask + x * neg_mask * alpha
        #
        # Optimized (allocation-reduced):
        #   pos_mask = x > 0                      (alloc)
        #   neg_mask = -pos_mask; neg_mask += 1   (alloc + in-place scalar add)
        #   out = x * pos_mask                    (alloc)
        #   tmp = x * neg_mask                    (alloc)
        #   tmp *= alpha                          (in-place scalar mul)
        #   out += tmp                            (in-place add)
        pos_mask = x > 0

        neg_mask = -pos_mask
        try:
            neg_mask += 1.0
        except Exception:
            neg_mask = neg_mask + 1.0

        out = x * pos_mask
        tmp = x * neg_mask
        tmp *= float(alpha)
        out += tmp

        ctx.save_for_backward(pos_mask, neg_mask)
        ctx.saved_meta["alpha"] = float(alpha)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient with respect to the input of `leaky_relu`.

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing masks and `alpha`.
        grad_out : Tensor
            Upstream gradient dL/d(leaky_relu(x)).

        Returns
        -------
        Tensor
            Gradient dL/dx.
        """
        pos_mask, neg_mask = ctx.saved_tensors
        alpha = float(ctx.saved_meta["alpha"])

        # Safer than mutating grad_out:
        # dx = grad_out * pos_mask
        dx = grad_out * pos_mask

        # tmp = grad_out * neg_mask; tmp *= alpha; dx += tmp
        tmp = grad_out * neg_mask
        tmp *= alpha
        dx += tmp
        return dx


class TanhFn(Function):
    """
    Hyperbolic tangent activation primitive.

    Computes tanh using an exp-based identity:

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    with derivative:

        d(tanh)/dx = 1 - tanh(x)^2

    Implementation details
    ----------------------
    The forward pass saves `tanh(x)` to compute the backward pass efficiently
    without re-evaluating exponentials.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """
        Compute `tanh(x)` elementwise and save the output for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store tensors required by the backward pass.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor containing `tanh(x)` elementwise.
        """
        # Original:
        #   e_pos = exp(x)
        #   e_neg = exp(-x)
        #   out = (e_pos - e_neg) / (e_pos + e_neg)
        #
        # Optimized:
        #   e_pos = exp(x)           (alloc)
        #   e_neg = exp(-x)          (alloc)
        #   num  = e_pos - e_neg     (alloc)
        #   e_pos += e_neg           (in-place; e_pos becomes denom, saves one alloc)
        #   out  = num / e_pos       (alloc)
        e_pos = exp(x)
        e_neg = exp(-x)

        num = e_pos - e_neg

        try:
            e_pos += e_neg
        except Exception:
            e_pos = e_pos + e_neg

        out = num / e_pos
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Compute the gradient with respect to the input of `tanh`.

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing the saved tanh output.
        grad_out : Tensor
            Upstream gradient dL/d(tanh(x)).

        Returns
        -------
        Tensor
            Gradient dL/dx.
        """
        (out,) = ctx.saved_tensors
        tmp = out * out
        tmp *= grad_out
        return grad_out - tmp


class SoftmaxFn(Function):
    """
    Softmax operation over a specified tensor axis.

    Softmax maps arbitrary real-valued scores to a probability distribution that
    sums to 1 along the selected axis.

    Numerically stable formulation
    ------------------------------
    The implementation subtracts the per-axis maximum before exponentiation:

        y = exp(x - max(x)) / sum(exp(x - max(x)))

    Backward
    --------
    The backward pass is implemented as a Jacobian–vector product (JVP) without
    explicitly materializing the full softmax Jacobian:

        dx = y * (g - sum(g * y, axis))

    Notes
    -----
    - This implementation assumes axis-aware reduction and alignment helpers exist
      on `Tensor` (e.g., `max(..., keepdims=True)`, `sum(..., keepdims=True)`,
      `broadcast_to(...)`).
    - The operation is currently restricted to CPU, consistent with the current
      backend capabilities.
    """

    @staticmethod
    def forward(ctx: Context, x: Tensor, *, axis: int = -1) -> Tensor:
        """
        Compute softmax over the specified axis.

        Parameters
        ----------
        ctx : Context
            Autograd context used to save tensors/metadata required by backward.
        x : Tensor
            Input tensor.
        axis : int, optional
            Axis along which to compute softmax. Negative values are interpreted
            relative to the last dimension. Defaults to -1.

        Returns
        -------
        Tensor
            Tensor of the same shape as `x`, containing softmax probabilities
            along the specified axis.

        Raises
        ------
        ValueError
            If the axis is out of bounds for the input tensor.
        RuntimeError
            If the operation is not supported on the tensor's device.
        """
        if not x.device.is_cpu():
            x._raise_device_not_supported("softmax")

        # Normalize axis
        ndim = len(x.shape)
        axis_ = axis if axis >= 0 else ndim + axis
        if axis_ < 0 or axis_ >= ndim:
            raise ValueError(f"Invalid softmax axis {axis} for ndim={ndim}")

        # Stable softmax via Tensor ops:
        # x_shift = x - max(x, axis, keepdims=True)
        m = x.max(axis=axis_, keepdims=True)  # shape has 1 on axis_
        m_full = m.broadcast_to(x.shape)  # explicit alignment
        x_shift = x - m_full

        exp_x = exp(x_shift)
        denom = exp_x.sum(axis=axis_, keepdims=True)
        denom_full = denom.broadcast_to(x.shape)

        y = exp_x / denom_full

        ctx.save_for_backward(y)
        ctx.saved_meta["axis"] = axis_
        return y

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor]:
        """
        Compute the gradient with respect to the input of softmax.

        Parameters
        ----------
        ctx : Context
            Context populated by `forward`, containing the saved softmax output
            and the normalized axis.
        grad_out : Tensor
            Upstream gradient dL/dy, with the same shape as the softmax output.

        Returns
        -------
        Tuple[Tensor]
            A single-element tuple `(dx,)` where `dx` is the gradient dL/dx.

        Notes
        -----
        This uses the identity:

            dx = y * (g - sum(g*y, axis))

        which is a Jacobian–vector product and avoids explicit Jacobian
        construction.
        """
        (y,) = ctx.saved_tensors
        axis = int(ctx.saved_meta["axis"])

        # dx = y * (g - sum(g*y, axis))
        gy = grad_out * y
        dot = gy.sum(axis=axis, keepdims=True).broadcast_to(y.shape)
        dx = y * (grad_out - dot)
        return (dx,)
