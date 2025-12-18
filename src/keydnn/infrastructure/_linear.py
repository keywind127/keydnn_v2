from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..domain._device import Device
from ..domain._tensor import ITensor
from ._module import Module
from ._parameter import Parameter
from ._tensor import Tensor, Context


class Linear(Module):
    """
    Fully connected (dense) layer: y = x @ W^T + b

    Parameters:
    - weight: (out_features, in_features)
    - bias:   (out_features,) or None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = device if device is not None else Device("cpu")

        # --- initialize weights (simple small random normal) ---
        w_np = (np.random.randn(out_features, in_features) * 0.01).astype(np.float32)

        # Prefer data-based construction if supported; otherwise fall back to shape+fill(0)
        try:
            self.weight = Parameter(data=w_np, device=self.device, requires_grad=True)
        except TypeError:
            self.weight = Parameter(
                (out_features, in_features), self.device, requires_grad=True
            )
            # If Tensor supports fill, at least initialize deterministically
            try:
                self.weight.fill(0.0)
            except Exception:
                pass

        self.register_parameter("weight", self.weight)

        # --- initialize bias ---
        if bias:
            b_np = np.zeros((out_features,), dtype=np.float32)
            try:
                self.bias = Parameter(data=b_np, device=self.device, requires_grad=True)
            except TypeError:
                self.bias = Parameter((out_features,), self.device, requires_grad=True)
                try:
                    self.bias.fill(0.0)
                except Exception:
                    pass
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x: ITensor) -> Tensor:
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

        # --- CPU compute ---
        x_np = x.to_numpy()
        w_np = self.weight.to_numpy()
        y_np = x_np @ w_np.T
        if self.bias is not None:
            y_np = y_np + self.bias.to_numpy()

        # --- build output tensor ---
        try:
            out = Tensor(data=y_np.astype(np.float32), device=self.device)
        except TypeError:
            out = Tensor(y_np.shape, self.device)
            if hasattr(out, "from_numpy") and callable(getattr(out, "from_numpy")):
                out.from_numpy(y_np.astype(np.float32))
            elif hasattr(out, "copy_from_numpy") and callable(
                getattr(out, "copy_from_numpy")
            ):
                out.copy_from_numpy(y_np.astype(np.float32))
            else:
                raise RuntimeError(
                    "No public way to load NumPy data into Tensor. "
                    "Add Tensor(data=...) or from_numpy()/copy_from_numpy()."
                )

        # --- autograd wiring (Context) ---
        # We only attach ctx if something requires grad.
        x_req = getattr(x, "requires_grad", False)
        w_req = self.weight.requires_grad
        b_req = self.bias is not None and self.bias.requires_grad

        if x_req or w_req or b_req:
            # IMPORTANT: parents should be actual Tensors so the engine can walk the graph.
            if not isinstance(x, Tensor):
                raise TypeError(
                    "Linear.forward(): to attach autograd Context, x must be an infrastructure Tensor "
                    "(so it can hold grad/ctx). Got ITensor without ctx."
                )

            # Save what we need for backward
            # (Saving x and weight is sufficient for Linear backward.)
            def backward_fn(grad_out: Tensor) -> Sequence[Optional[Tensor]]:
                # grad_out: (batch, out_features)
                go = grad_out.to_numpy()
                x_saved, w_saved = ctx.saved_tensors
                x_arr = x_saved.to_numpy()  # (batch, in_features)
                w_arr = w_saved.to_numpy()  # (out_features, in_features)

                grad_x = None
                grad_w = None
                grad_b = None

                def _make_grad(device: Device, arr: np.ndarray) -> Tensor:
                    """Create a Tensor by shape, then load NumPy into it via public API."""
                    t = Tensor(arr.shape, device)
                    if hasattr(t, "copy_from_numpy") and callable(
                        getattr(t, "copy_from_numpy")
                    ):
                        t.copy_from_numpy(arr.astype(np.float32, copy=False))
                        return t
                    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
                        t.from_numpy(arr.astype(np.float32, copy=False))
                        return t
                    raise RuntimeError(
                        "No public way to load NumPy data into Tensor for gradients. "
                        "Need copy_from_numpy() or from_numpy()."
                    )

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

                # parents order must match ctx.parents order below
                if self.bias is None:
                    return (grad_x, grad_w)
                return (grad_x, grad_w, grad_b)

            parents = (
                [x, self.weight] if self.bias is None else [x, self.weight, self.bias]
            )
            ctx = Context(parents=parents, backward_fn=backward_fn)
            ctx.save_for_backward(x, self.weight)

            out.requires_grad = True
            out._set_ctx(ctx)  # uses your internal hook

        return out
