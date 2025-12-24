"""
Module-based activation layers.

This module provides infrastructure-level `Module` wrappers around the
function-style autograd implementations (e.g., `SigmoidFn`, `ReLUFn`).

Why both Function and Module forms exist
----------------------------------------
- `Function` classes (e.g., `ReLUFn`) implement the mathematical operation and
  its derivatives (forward/backward) in a reusable, low-level form.
- `Module` classes (e.g., `ReLU`) provide a higher-level, layer-like interface
  that integrates naturally with model composition, `__call__` usage, and
  parameter traversal (even if these activations have no parameters).

Autograd integration
--------------------
Each module constructs a `Context` during `forward` and wires a `backward_fn`
that delegates to the corresponding `Function.backward`. If the input requires
gradients, the context is attached to the output tensor so that the autograd
engine can traverse the graph.

Notes
-----
- These activation modules are stateless except for `LeakyReLU`, which stores
  the `alpha` hyperparameter.
- `x` is expected to be an infrastructure `Tensor` so it can carry autograd
  context. (This module currently types it as `Tensor` accordingly.)
"""

from typing import Any, Dict

from .tensor._tensor_context import Context

from ..domain.model._stateless_mixin import StatelessConfigMixin
from .module._serialization_core import register_module
from ._tensor import Tensor
from ._module import Module

from ._function import SigmoidFn, ReLUFn, LeakyReLUFn, TanhFn, SoftmaxFn


@register_module()
class Sigmoid(StatelessConfigMixin, Module):
    """
    Sigmoid activation module.

    This layer applies the sigmoid function elementwise:

        sigmoid(x) = 1 / (1 + exp(-x))

    Notes
    -----
    This module is a thin wrapper around `SigmoidFn` that provides a `Module`
    interface and attaches an autograd `Context` to the output when gradients
    are required.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the sigmoid activation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing sigmoid(x) elementwise.

        Notes
        -----
        If `x.requires_grad` is True, the returned tensor will have an attached
        `Context` whose `backward_fn` delegates to `SigmoidFn.backward`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (SigmoidFn.backward(ctx, grad_out),),
        )
        out = SigmoidFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class ReLU(StatelessConfigMixin, Module):
    """
    ReLU activation module.

    This layer applies the rectified linear unit elementwise:

        relu(x) = max(0, x)

    Notes
    -----
    This module is a thin wrapper around `ReLUFn` that provides a `Module`
    interface and attaches an autograd `Context` to the output when gradients
    are required.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the ReLU activation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing relu(x) elementwise.

        Notes
        -----
        If `x.requires_grad` is True, the returned tensor will have an attached
        `Context` whose `backward_fn` delegates to `ReLUFn.backward`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (ReLUFn.backward(ctx, grad_out),),
        )
        out = ReLUFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class LeakyReLU(Module):
    """
    Leaky ReLU activation module.

    This layer applies the leaky ReLU function elementwise:

        f(x) = x               if x > 0
             = alpha * x       otherwise

    Parameters
    ----------
    alpha : float, default=0.01
        Slope for negative inputs.

    Attributes
    ----------
    alpha : float
        Negative slope coefficient.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initialize the LeakyReLU module.

        Parameters
        ----------
        alpha : float, optional
            Negative slope coefficient applied when x <= 0.
        """
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Leaky ReLU activation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing leaky_relu(x) elementwise.

        Notes
        -----
        If `x.requires_grad` is True, the returned tensor will have an attached
        `Context` whose `backward_fn` delegates to `LeakyReLUFn.backward`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (LeakyReLUFn.backward(ctx, grad_out),),
        )
        out = LeakyReLUFn.forward(ctx, x, alpha=self.alpha)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out

    def get_config(self) -> Dict[str, Any]:
        return {"alpha": float(self.alpha)}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LeakyReLU":
        return cls(alpha=float(cfg.get("alpha", 0.01)))


@register_module()
class Tanh(StatelessConfigMixin, Module):
    """
    Hyperbolic tangent activation module.

    This layer applies the tanh function elementwise:

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Notes
    -----
    This module is a thin wrapper around `TanhFn` that provides a `Module`
    interface and attaches an autograd `Context` to the output when gradients
    are required.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the tanh activation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor containing tanh(x) elementwise.

        Notes
        -----
        If `x.requires_grad` is True, the returned tensor will have an attached
        `Context` whose `backward_fn` delegates to `TanhFn.backward`.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (TanhFn.backward(ctx, grad_out),),
        )
        out = TanhFn.forward(ctx, x)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out


@register_module()
class Softmax(Module):
    """
    Softmax activation module.

    This module applies the softmax function to its input tensor along a
    specified axis, producing a normalized probability distribution.

    By default, softmax is applied over the last dimension, which is the
    standard convention for classification outputs.
    """

    def __init__(self, *, axis: int = -1) -> None:
        """
        Construct a Softmax activation module.

        Parameters
        ----------
        axis : int, optional
            Dimension along which the softmax operation is applied.
            Defaults to the last dimension (`-1`).
        """
        super().__init__()
        self._axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the softmax activation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor to which softmax will be applied.

        Returns
        -------
        Tensor
            A tensor of the same shape as `x`, where values along the specified
            axis form a probability distribution (sum to 1).

        Notes
        -----
        - This method delegates the numerical computation to `SoftmaxFn`,
          attaching a backward `Context` when gradient tracking is enabled.
        - The returned tensor will participate in autograd only if
          `x.requires_grad` is True.
        - Gradient propagation is implemented via a Jacobian–vector product
          in `SoftmaxFn.backward`, avoiding explicit Jacobian construction.
        """
        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: (SoftmaxFn.backward(ctx, grad_out)[0],),
        )

        out = SoftmaxFn.forward(ctx, x, axis=self._axis)

        if x.requires_grad:
            out.requires_grad = True
            out._set_ctx(ctx)

        return out

    def get_config(self) -> Dict[str, Any]:
        return {"axis": int(self._axis)}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Softmax":
        return cls(axis=int(cfg.get("axis", -1)))
