"""
Loss function primitives for KeyDNN.

This module implements common regression and classification loss functions
as subclasses of `Function`, following KeyDNN's explicit forward/backward
autograd design.

Currently implemented losses:
- SSEFn  : Sum of Squared Errors
- MSEFn  : Mean Squared Error
- BinaryCrossEntropyFn : Binary Cross Entropy (probability inputs)
- CategoricalCrossEntropyFn : Categorical Cross Entropy (probabilities with one-hot targets)

Design notes
------------
- Losses are implemented as `Function` subclasses rather than `Module`s to
  emphasize their role as terminal nodes in the computation graph.
- All reductions (`sum`, `mean`) are delegated to `Tensor` methods to keep
  NumPy usage out of the loss logic and preserve graph consistency.
- Backward implementations avoid tensor broadcasting and instead explicitly
  scale gradients using Python scalars, in accordance with the framework's
  strict shape-matching rules.
- Classification losses currently operate on probability inputs; logits-based
  variants and numerically stabilized formulations may be added in the future.

These losses return scalar tensors and are intended to be used as the final
operation before invoking backpropagation.
"""

import numpy as np
from typing import Tuple
from ._function import Function
from ._tensor import Tensor, Context


def _scalar_to_float(t: Tensor) -> float:
    """
    Extract a Python scalar from a scalar Tensor.

    Parameters
    ----------
    t : Tensor
        A scalar tensor (shape `()` or equivalent) residing on the CPU.

    Returns
    -------
    float
        The scalar value stored in the tensor.

    Notes
    -----
    This helper exists to bridge scalar gradient values (`grad_out`) with
    KeyDNN's current non-broadcasting Tensor semantics. It is intentionally
    used inside backward passes where scalar multiplication of higher-rank
    tensors is required.
    """
    return float(np.asarray(t.to_numpy()).reshape(-1)[0])


class SSEFn(Function):
    """
    Sum of Squared Errors (SSE) loss function.

    Computes the scalar loss:

        SSE(pred, target) = sum((pred - target)^2)

    This loss is commonly used in regression tasks and serves as a simple
    baseline loss for validating autograd correctness.

    Notes
    -----
    - The forward pass returns a scalar tensor.
    - The backward pass computes gradients with respect to both `pred`
      and `target`.
    - No broadcasting is assumed; all tensor shapes must match exactly.
    """

    @staticmethod
    def forward(ctx: Context, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Sum of Squared Errors.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store intermediate values for backward.
        pred : Tensor
            Predicted values.
        target : Tensor
            Ground-truth target values.

        Returns
        -------
        Tensor
            A scalar tensor containing the SSE loss.

        Notes
        -----
        The difference tensor `(pred - target)` is saved for use during the
        backward pass.
        """
        diff = pred - target
        ctx.save_for_backward(diff)
        return (diff * diff).sum()

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute gradients of SSE with respect to inputs.

        Parameters
        ----------
        ctx : Context
            Autograd context populated during the forward pass.
        grad_out : Tensor
            Gradient of the total loss with respect to the SSE output.
            This is expected to be a scalar tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Gradients with respect to `pred` and `target`, respectively.

        Notes
        -----
        Gradient formulas:
            dSSE/dpred   =  2 * (pred - target)
            dSSE/dtarget = -2 * (pred - target)

        The upstream gradient `grad_out` is treated as a scalar and applied
        via explicit scaling to avoid tensor broadcasting.
        """
        (diff,) = ctx.saved_tensors

        g = _scalar_to_float(grad_out)

        grad_pred = diff * (2.0 * g)
        grad_target = diff * (-2.0 * g)
        return grad_pred, grad_target


class MSEFn(Function):
    """
    Mean Squared Error (MSE) loss function.

    Computes the scalar loss:

        MSE(pred, target) = mean((pred - target)^2)

    This loss normalizes the Sum of Squared Errors by the total number of
    elements and is one of the most commonly used losses for regression.

    Notes
    -----
    - The forward pass returns a scalar tensor.
    - The backward pass uses the total number of elements (`numel`) to scale
      gradients appropriately.
    - Shape broadcasting is intentionally not supported.
    """

    @staticmethod
    def forward(ctx: Context, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Mean Squared Error.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store intermediate values for backward.
        pred : Tensor
            Predicted values.
        target : Tensor
            Ground-truth target values.

        Returns
        -------
        Tensor
            A scalar tensor containing the MSE loss.

        Notes
        -----
        The difference tensor `(pred - target)` and the total number of elements
        are saved for use during the backward pass.
        """
        diff = pred - target
        ctx.save_for_backward(diff)
        ctx.saved_meta["n"] = diff.numel()
        return (diff * diff).mean()

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute gradients of MSE with respect to inputs.

        Parameters
        ----------
        ctx : Context
            Autograd context populated during the forward pass.
        grad_out : Tensor
            Gradient of the total loss with respect to the MSE output.
            This is expected to be a scalar tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Gradients with respect to `pred` and `target`, respectively.

        Notes
        -----
        Gradient formulas:
            dMSE/dpred   =  2 * (pred - target) / N
            dMSE/dtarget = -2 * (pred - target) / N

        where N is the total number of elements in the input tensor.

        The upstream gradient `grad_out` is applied as a scalar multiplier to
        preserve strict shape matching.
        """
        (diff,) = ctx.saved_tensors
        n = int(ctx.saved_meta["n"])

        g = _scalar_to_float(grad_out)
        scale = (2.0 / n) * g

        grad_pred = diff * scale
        grad_target = diff * (-scale)
        return grad_pred, grad_target


class BinaryCrossEntropyFn(Function):
    """
    Binary Cross Entropy (BCE) loss function.

    Computes the mean binary cross entropy between predicted probabilities
    and binary targets:

        BCE(pred, target) =
            mean( -[ target * log(pred) + (1 - target) * log(1 - pred) ] )

    This loss is commonly used for binary classification tasks where
    predictions are probabilities in the open interval (0, 1).

    Notes
    -----
    - `pred` is expected to contain probabilities (e.g., output of Sigmoid).
    - `target` is expected to contain binary values (0 or 1) with the same shape.
    - The forward pass returns a scalar tensor (mean reduction).
    - The backward pass computes gradients with respect to `pred` only;
      `target` is treated as a constant.
    - No broadcasting is performed; tensor shapes must match exactly.
    """

    @staticmethod
    def forward(ctx: Context, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the mean Binary Cross Entropy loss.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store intermediate values for backward.
        pred : Tensor
            Predicted probabilities with values in (0, 1).
        target : Tensor
            Binary ground-truth targets (0 or 1), same shape as `pred`.

        Returns
        -------
        Tensor
            A scalar tensor containing the mean BCE loss.

        Notes
        -----
        The total number of elements is stored in the context to support
        correct gradient scaling during the backward pass.
        """
        # Save for backward
        ctx.save_for_backward(pred, target)
        ctx.saved_meta["n"] = pred.numel()

        # BCE elementwise loss then mean reduction
        loss = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
        return loss.mean()

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor):
        """
        Compute gradients of the BCE loss with respect to inputs.

        Parameters
        ----------
        ctx : Context
            Autograd context populated during the forward pass.
        grad_out : Tensor
            Gradient of the total loss with respect to the BCE output.
            This is expected to be a scalar tensor.

        Returns
        -------
        tuple[Tensor, None]
            Gradient with respect to `pred`, and `None` for `target`.

        Notes
        -----
        For mean-reduced BCE, the gradient is:

            dL/dpred = (pred - target) / (pred * (1 - pred)) / N

        where N is the total number of elements. The upstream gradient
        `grad_out` is applied as a scalar multiplier.
        """
        pred, target = ctx.saved_tensors
        n = int(ctx.saved_meta["n"])

        g = _scalar_to_float(grad_out)

        # d/dp for mean BCE
        grad_pred = (pred - target) / (pred * (1.0 - pred))
        grad_pred = grad_pred * (g / n)

        # Typically treat targets as constants
        grad_target = None
        return grad_pred, grad_target


class CategoricalCrossEntropyFn(Function):
    """
    Categorical Cross Entropy (CCE) loss function.

    Computes the categorical cross entropy between predicted class
    probabilities and one-hot encoded targets:

        CCE(pred, target) =
            -sum(target * log(pred)) / N

    where N is the batch size.

    This loss is commonly used for multi-class classification tasks when
    predictions are provided as probabilities (e.g., output of Softmax).

    Notes
    -----
    - `pred` is expected to have shape (N, C) and contain class probabilities.
    - `target` must be one-hot encoded with the same shape as `pred`.
    - The forward pass returns a scalar tensor (mean over batch).
    - The backward pass computes gradients with respect to `pred` only;
      `target` is treated as a constant.
    - No broadcasting or class-index targets are supported in this version.
    """

    @staticmethod
    def forward(ctx: Context, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the categorical cross entropy loss.

        Parameters
        ----------
        ctx : Context
            Autograd context used to store intermediate values for backward.
        pred : Tensor
            Predicted class probabilities of shape (N, C).
        target : Tensor
            One-hot encoded class labels of shape (N, C).

        Returns
        -------
        Tensor
            A scalar tensor containing the categorical cross entropy loss
            averaged over the batch.

        Notes
        -----
        The loss is reduced by averaging over the batch dimension (N).
        """
        ctx.save_for_backward(pred, target)

        loss = -(target * pred.log())
        return loss.sum() / pred.shape[0]

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor):
        """
        Compute gradients of the CCE loss with respect to inputs.

        Parameters
        ----------
        ctx : Context
            Autograd context populated during the forward pass.
        grad_out : Tensor
            Gradient of the total loss with respect to the CCE output.
            This is expected to be a scalar tensor.

        Returns
        -------
        tuple[Tensor, None]
            Gradient with respect to `pred`, and `None` for `target`.

        Notes
        -----
        For batch-mean categorical cross entropy, the gradient is:

            dL/dpred = -(target / pred) / N

        where N is the batch size. The upstream gradient `grad_out` is applied
        as a scalar multiplier.
        """
        (pred, target) = ctx.saved_tensors
        g = _scalar_to_float(grad_out)

        # dL/dpred = -target / pred
        grad_pred = -(target / pred)
        grad_pred = grad_pred * (g / pred.shape[0])

        grad_target = None
        return grad_pred, grad_target
