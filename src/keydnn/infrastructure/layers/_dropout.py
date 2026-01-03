"""
Dropout regularization module for KeyDNN.

This module implements **inverted dropout** (a.k.a. "scaled dropout").
During training, activations are randomly masked with probability `p`
and scaled by `1 / (1 - p)` to preserve the expected value of the
activations. During evaluation, the layer behaves as an identity
function.

Device support
--------------
- **CPU and CUDA** tensors are supported.
- The dropout mask is materialized on the **same device** as the input.
- Backward uses the same mask (and scaling) as forward. On CUDA this is
  fully device-resident (no NumPy / host round-trips), relying on the
  framework's broadcast + sum-to-shape reduction support.

Design notes
------------
- Uses *inverted dropout*, so no scaling is required at inference time.
- The mask is constructed as:
    mask = (rand(shape) < keep_prob) / keep_prob
  so the output is:
    y = x * mask
- Integrates with KeyDNN's serialization system via `register_module`.
"""

from __future__ import annotations

from typing import Any, Dict

from ..tensor._tensor_context import Context

from .._module import Module
from ..tensor._tensor import Tensor
from ..module._serialization_core import register_module


@register_module()
class Dropout(Module):
    """
    Dropout regularization layer (inverted dropout).

    This layer randomly zeroes elements of the input tensor with
    probability `p` during training and rescales the remaining elements
    by `1 / (1 - p)` so that the expected activation magnitude remains
    unchanged.

    Behavior
    --------
    - Training mode:
        y = x * mask / (1 - p), where mask ~ Bernoulli(1 - p)
      (equivalently y = x * ((rand < keep_prob) / keep_prob))
    - Evaluation mode:
        y = x (identity)

    Parameters
    ----------
    p : float, optional
        Probability of dropping (zeroing) an element. Must satisfy
        0.0 <= p < 1.0. Default is 0.5.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the Dropout module.

        Parameters
        ----------
        p : float, optional
            Dropout probability. Must be in the range [0, 1).
        """
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability p must be in [0, 1).")
        self.p = float(p)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout to the input tensor.

        During training, elements of the input tensor are randomly masked
        according to the dropout probability and scaled using inverted
        dropout. During evaluation, the input tensor is returned
        unchanged.

        Parameters
        ----------
        x : Tensor
            Input tensor (CPU or CUDA).

        Returns
        -------
        Tensor
            Output tensor after applying dropout (or identity if not in
            training mode).

        Raises
        ------
        ValueError
            If `p` implies a non-positive keep probability (numerical guard).
        """
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            raise ValueError("Dropout keep_prob must be > 0.")

        # Device-resident random mask
        r = Tensor.rand(x.shape, device=x.device)  # [0, 1)

        # scalar -> Tensor, then broadcast to match r
        kp = Tensor.full((), keep_prob, device=x.device, requires_grad=False)
        kp = kp.broadcast_to(x.shape)

        mask = r < kp
        mask /= kp  # inverted dropout scaling

        req = bool(x.requires_grad)

        y = x * mask

        out = Tensor(shape=y.shape, device=y.device, requires_grad=req, ctx=None)
        out.copy_from(y)

        if req:
            ctx = Context(
                parents=(x,),
                backward_fn=lambda grad_out: (grad_out * mask,),
            )
            ctx.saved_tensors.append(mask)
            out._set_ctx(ctx)

        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this module.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing the dropout probability.
        """
        return {"p": self.p}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dropout":
        """
        Construct a Dropout module from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary produced by `get_config`.

        Returns
        -------
        Dropout
            A new Dropout instance initialized from the configuration.
        """
        return cls(p=float(config["p"]))
