"""
Linear (fully-connected) layer implementation.

This module defines an infrastructure-level `Linear` layer that implements the
domain-level `IModule` contract via the `Module` base class.

The layer computes an affine transformation:

    y = x @ W^T + b

where:
- x has shape (batch, in_features)
- W has shape (out_features, in_features)
- b has shape (out_features,) or is omitted if bias=False

Autograd integration
--------------------
If any of the participating tensors require gradients, `Linear.forward` attaches
a `Context` to the output tensor. The context stores parents and a `backward_fn`
that computes gradients for:
- input x
- weight W
- bias b (if present)

Important limitations
---------------------
- This implementation currently supports CPU execution only (NumPy backend).
- Broadcasting is not used; shapes must match the expected 2D input format.
- For autograd wiring, the input `x` must be an infrastructure `Tensor`
  (not just an `ITensor`) so that it can carry `ctx` and `grad` state.
"""

from __future__ import annotations

from typing import Optional, Sequence, Any, Dict

import numpy as np


from .module._serialization_core import register_module
from ._module import Module
from ._parameter import Parameter
from ._tensor import Tensor, Context
from ..domain.device._device import Device
from ..domain._tensor import ITensor


@register_module()
class Linear(Module):
    """
    Fully connected (dense) layer: y = x @ W^T + b.

    Parameters are stored as trainable `Parameter` instances:
    - weight: shape (out_features, in_features)
    - bias:   shape (out_features,) or None

    Parameters
    ----------
    in_features : int
        Number of input features per example.
    out_features : int
        Number of output features per example.
    bias : bool, optional
        If True, include a learnable bias term. Defaults to True.
    device : Optional[Device], optional
        Device placement. Defaults to CPU if not provided.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    device : Device
        Device placement for parameters and outputs.
    weight : Parameter
        Trainable weight matrix of shape (out_features, in_features).
    bias : Optional[Parameter]
        Trainable bias vector of shape (out_features,), or None if disabled.

    Raises
    ------
    ValueError
        If `in_features` or `out_features` is not a positive integer.

    Notes
    -----
    Weight initialization uses Xavier/Glorot uniform to break symmetry
    (critical for learning problems like XOR). Bias is initialized to zeros.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize a Linear layer and register its parameters.

        Parameters
        ----------
        in_features : int
            Number of input features per example.
        out_features : int
            Number of output features per example.
        bias : bool, optional
            If True, include a learnable bias term.
        device : Optional[Device], optional
            Device placement. Defaults to CPU if not provided.

        Raises
        ------
        ValueError
            If `in_features` or `out_features` is not positive.
        """
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = device if device is not None else Device("cpu")

        # --- initialize weights (Xavier/Glorot uniform) ---
        limit = float(np.sqrt(6.0 / (self.in_features + self.out_features)))
        w_np = np.random.uniform(
            low=-limit,
            high=limit,
            size=(self.out_features, self.in_features),
        ).astype(np.float32)

        self.weight = Parameter(
            shape=(self.out_features, self.in_features),
            device=self.device,
            requires_grad=True,
        )
        self.weight.copy_from_numpy(w_np)
        self.register_parameter("weight", self.weight)

        # --- initialize bias ---
        if bias:
            b_np = np.zeros((self.out_features,), dtype=np.float32)
            self.bias = Parameter(
                shape=(self.out_features,),
                device=self.device,
                requires_grad=True,
            )
            self.bias.copy_from_numpy(b_np)
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x: ITensor) -> Tensor:
        """
        Compute the forward pass of the Linear layer.

        Given input `x` of shape (batch, in_features), compute:

            y = x @ W^T + b

        Parameters
        ----------
        x : ITensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, out_features).

        Raises
        ------
        ValueError
            If the input is not 2D, or if its feature dimension does not match
            `self.in_features`.
        TypeError
            If gradient wiring is needed and `x` is not an infrastructure `Tensor`
            (because it cannot carry autograd context).
        """
        # --- validate input shape ---
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Linear expects 2D input (batch, in_features), got {x_shape}"
            )
        if x_shape[1] != self.in_features:
            raise ValueError(
                f"Linear expects in_features={self.in_features}, got {x_shape[1]}"
            )

        # --- compute (CPU) ---
        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()
        y_np = x_np @ w_np.T
        if self.bias is not None:
            y_np = y_np + self.bias.to_numpy()

        # --- decide whether output should track gradients ---
        x_req = bool(getattr(x, "requires_grad", False))
        w_req = bool(self.weight.requires_grad)
        b_req = bool(self.bias is not None and self.bias.requires_grad)
        req = x_req or w_req or b_req

        # --- build output tensor (always via shape+copy, which your Tensor supports) ---
        out = Tensor(shape=y_np.shape, device=self.device, requires_grad=req)
        out.copy_from_numpy(y_np.astype(np.float32, copy=False))

        # --- autograd wiring (Context) ---
        if req:
            if not isinstance(x, Tensor):
                raise TypeError(
                    "Linear.forward(): to attach autograd Context, x must be an infrastructure Tensor "
                    "(so it can hold grad/ctx). Got ITensor without ctx."
                )

            def _make_grad(device: Device, arr: np.ndarray) -> Tensor:
                """
                Create a gradient Tensor from a NumPy array using public APIs.

                Parameters
                ----------
                device : Device
                    Target device for the gradient tensor.
                arr : np.ndarray
                    Gradient values to load.

                Returns
                -------
                Tensor
                    A newly created tensor holding the given gradient values.
                """
                t = Tensor(shape=arr.shape, device=device, requires_grad=False)
                t.copy_from_numpy(arr.astype(np.float32, copy=False))
                return t

            def backward_fn(grad_out: Tensor) -> Sequence[Optional[Tensor]]:
                """
                Compute gradients for the Linear operation.

                Parameters
                ----------
                grad_out : Tensor
                    Gradient of the loss with respect to the layer output `y`,
                    with shape (batch, out_features).

                Returns
                -------
                Sequence[Optional[Tensor]]
                    Gradients with respect to each parent listed in `ctx.parents`,
                    in the same order. Entries may be None for parents that do not
                    require gradients.

                Notes
                -----
                The computed gradients (without broadcasting) are:
                - dL/dx = dL/dy @ W
                - dL/dW = (dL/dy)^T @ x
                - dL/db = sum(dL/dy over batch)
                """
                go = grad_out.to_numpy()

                # Saved tensors
                x_saved, w_saved = ctx.saved_tensors
                x_arr = x_saved.to_numpy()  # (batch, in_features)
                w_arr = w_saved.to_numpy()  # (out_features, in_features)

                grad_x = None
                grad_w = None
                grad_b = None

                # dL/dx = dL/dy @ W
                if x_saved.requires_grad:
                    gx = go @ w_arr  # (batch, in_features)
                    grad_x = _make_grad(x_saved.device, gx)

                # dL/dW = (dL/dy)^T @ x
                if w_saved.requires_grad:
                    gw = go.T @ x_arr  # (out_features, in_features)
                    grad_w = _make_grad(w_saved.device, gw)

                # dL/db = sum over batch
                if self.bias is not None and self.bias.requires_grad:
                    gb = go.sum(axis=0)  # (out_features,)
                    grad_b = _make_grad(self.bias.device, gb)

                if self.bias is None:
                    return (grad_x, grad_w)
                return (grad_x, grad_w, grad_b)

            parents = (
                [x, self.weight] if self.bias is None else [x, self.weight, self.bias]
            )
            ctx = Context(parents=parents, backward_fn=backward_fn)

            # Save for backward: x and weight are sufficient
            ctx.save_for_backward(x, self.weight)

            out._set_ctx(ctx)

        return out

    # -------------------------------------------------------------------------
    # ADD-ON ONLY: JSON serialization hooks (no change to existing logic above)
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this layer.

        Notes
        -----
        This configuration captures constructor-level hyperparameters only.
        Trainable parameters (weights/bias) are serialized separately by the
        checkpoint/state_dict mechanism.
        """
        return {
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "bias": bool(self.bias is not None),
            # Store device as a string to keep JSON stable.
            # Assumes Device can be reconstructed from its string form (e.g., "cpu").
            "device": str(self.device),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Linear":
        """
        Construct a Linear layer from a configuration dict.

        Notes
        -----
        This reconstructs the module structure (hyperparameters). Weights are
        expected to be loaded afterward from the checkpoint state.
        """
        dev = cfg.get("device", "cpu")
        return cls(
            in_features=int(cfg["in_features"]),
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=Device(str(dev)),
        )
