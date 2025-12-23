"""
Linear (fully-connected) layer implementation.

This module provides an infrastructure-level `Linear` layer for KeyDNN. It is a
trainable `Module` (and is registered for serialization via `register_module`)
that performs an affine projection of 2D, batch-major inputs:

    y = x @ W^T + b

Shape conventions
-----------------
- x : (batch, in_features)
- W : (out_features, in_features)
- b : (out_features,)  (optional; omitted if bias=False)
- y : (batch, out_features)

Computation and backend constraints
-----------------------------------
- CPU-only (current NumPy-backed Tensor storage).
- Input must be 2D. Higher-rank inputs are not implicitly flattened.
- Bias addition avoids implicit broadcasting by explicitly expanding `b` to
  (batch, out_features) using `Tensor.stack`.

Autograd integration
--------------------
`forward()` computes results using Tensor operations (e.g., `@`, `.T`, `+`) and,
when gradients are required, attaches a *legacy* `Context` to a fresh output
tensor. The context uses parents ordered as:
- (x, weight) if bias is disabled
- (x, weight, bias) if bias is enabled

The backward rule for out = x @ W^T (+ b) is:
- dL/dx = dL/dout @ W
- dL/dW = (dL/dout)^T @ x
- dL/db = sum(dL/dout, axis=0)

Design note
-----------
Parameter initialization is isolated in `_reset_parameters()` and currently uses
NumPy (Xavier/Glorot uniform for weights, zeros for bias). This allows future
replacement with a device-aware RNG/initializer without changing constructor
logic.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from .module._serialization_core import register_module
from ._module import Module
from ._parameter import Parameter
from ._tensor import Tensor, Context
from ..domain.device._device import Device


@register_module()
class Linear(Module):
    """
    Fully-connected (dense) layer performing an affine transform: y = x @ W^T + b.

    This layer projects 2D batch-major inputs from `in_features` to `out_features`
    using a learnable weight matrix and an optional bias vector.

    Parameters
    ----------
    in_features : int
        Number of input features per example.
    out_features : int
        Number of output features per example.
    bias : bool, optional
        If True, include a learnable bias term. Defaults to True.
    device : Optional[Device], optional
        Device placement for parameters and outputs. Defaults to CPU if not provided.

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
    - Weight initialization uses Xavier/Glorot uniform.
    - Bias (when enabled) is initialized to zeros.
    - Current implementation is CPU-only and expects 2D inputs.
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

        This constructor validates sizes, allocates `Parameter` storage for weights
        (and optional bias), registers parameters for state management, and then
        initializes parameter values via `_reset_parameters()`.

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

        # Allocate parameters (storage + metadata)
        self.weight = Parameter(
            shape=(self.out_features, self.in_features),
            device=self.device,
            requires_grad=True,
        )
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias = Parameter(
                shape=(self.out_features,),
                device=self.device,
                requires_grad=True,
            )
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

        # Initialize values
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize learnable parameters.

        Weights are initialized with Xavier/Glorot uniform and bias (if present)
        is initialized to zeros. The initializer currently uses NumPy to generate
        CPU arrays, then copies them into parameter storage.

        Notes
        -----
        This method is intentionally isolated so it can be replaced by a device-aware
        RNG initializer in the future (e.g., CUDA kernels, seeded generators).
        """
        import numpy as np  # keep numpy dependency localized to initialization

        limit = float(np.sqrt(6.0 / (self.in_features + self.out_features)))
        w_np = np.random.uniform(
            low=-limit,
            high=limit,
            size=(self.out_features, self.in_features),
        ).astype(np.float32)

        self.weight.copy_from_numpy(w_np)

        if self.bias is not None:
            b_np = np.zeros((self.out_features,), dtype=np.float32)
            self.bias.copy_from_numpy(b_np)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the affine transform to a 2D input tensor.

        The forward pass computes:
            y = x @ W^T (+ b)

        where `x` must have shape (batch, in_features). Bias addition is performed
        without implicit broadcasting by explicitly expanding `b` across the batch
        dimension using `Tensor.stack`.

        Autograd behavior
        -----------------
        If gradients are required for any of (x, weight, bias), this method attaches
        a legacy `Context` to a fresh output tensor whose parents are exactly:
        - (x, weight) if bias is disabled
        - (x, weight, bias) if bias is enabled

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, out_features).

        Raises
        ------
        ValueError
            If `x` is not 2D or if its second dimension does not match `in_features`.
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

        # --- decide whether output should track gradients ---
        x_req = bool(x.requires_grad)
        w_req = bool(self.weight.requires_grad)
        b_req = bool(self.bias is not None and self.bias.requires_grad)
        req = x_req or w_req or b_req

        # --- compute forward using Tensor ops only ---
        y = x @ self.weight.T  # (batch, out_features)

        if self.bias is not None:
            # No broadcasting: expand bias to 2D by stacking along batch axis.
            batch = x_shape[0]
            b2d = Tensor.stack([self.bias] * batch, axis=0)  # (batch, out_features)
            y = y + b2d

        # --- return early if no autograd needed ---
        if not req:
            return y

        # --- attach legacy Context with parents (x, weight, bias?) ---
        # Create a fresh output tensor so its ctx is exactly what Linear defines.
        out = Tensor(shape=y.shape, device=self.device, requires_grad=True)
        out.copy_from(y)

        def backward_fn(grad_out: Tensor):
            """
            Compute gradients for Linear's parents given gradient at the output.

            Parameters
            ----------
            grad_out : Tensor
                Gradient w.r.t. the output of this layer, shape (batch, out_features).

            Returns
            -------
            tuple[Optional[Tensor], ...]
                Gradients for (x, weight) or (x, weight, bias) depending on whether
                bias is enabled. Non-required gradients may be returned as None.
            """
            x_saved, w_saved = ctx.saved_tensors

            grad_x = None
            grad_w = None
            grad_b = None

            if x_saved.requires_grad:
                grad_x = grad_out @ w_saved  # (batch, in_features)

            if w_saved.requires_grad:
                grad_w = grad_out.T @ x_saved  # (out_features, in_features)

            if self.bias is not None and self.bias.requires_grad:
                grad_b = grad_out.sum(axis=0)  # (out_features,)

            if self.bias is None:
                return (grad_x, grad_w)
            return (grad_x, grad_w, grad_b)

        parents = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        ctx = Context(parents=parents, backward_fn=backward_fn)
        ctx.save_for_backward(x, self.weight)
        out._set_ctx(ctx)

        return out

    # -------------------------------------------------------------------------
    # ADD-ON ONLY: JSON serialization hooks (no change to existing logic above)
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this layer.

        The returned configuration contains constructor-level hyperparameters only.
        Trainable parameter values (weights/bias) are expected to be handled by the
        checkpoint/state mechanism.

        Returns
        -------
        Dict[str, Any]
            A JSON-serializable dict containing `in_features`, `out_features`, `bias`,
            and `device` (as a string).
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

        This reconstructs the module structure (hyperparameters). Weights are
        expected to be loaded afterward from the checkpoint/state.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        Linear
            A newly constructed `Linear` instance with matching hyperparameters.
        """
        dev = cfg.get("device", "cpu")
        return cls(
            in_features=int(cfg["in_features"]),
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=Device(str(dev)),
        )
