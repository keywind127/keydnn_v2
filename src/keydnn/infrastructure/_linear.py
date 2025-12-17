from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain._device import Device
from ..domain._tensor import ITensor
from ._module import Module
from ._parameter import Parameter
from ._tensor import Tensor


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
        """
        CPU implementation via NumPy (using to_numpy()).

        Notes:
        - Requires x to be CPU if your Tensor.to_numpy() is CPU-only.
        - Produces an infrastructure Tensor output.
        """
        # Basic shape validation (expects x shape: (batch, in_features))
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Linear expects a 2D input (batch, in_features), got {x_shape}"
            )
        if x_shape[1] != self.in_features:
            raise ValueError(
                f"Linear expects input feature dim {self.in_features}, got {x_shape[1]}"
            )

        # NumPy compute path (CPU-only for now)
        x_np = x.to_numpy()  # should raise on CUDA per your current Tensor contract
        w_np = self.weight.to_numpy()

        y_np = x_np @ w_np.T  # (batch, out_features)

        if self.bias is not None:
            b_np = self.bias.to_numpy()
            y_np = y_np + b_np  # broadcast over batch

        # Create output Tensor (prefer data-based construction)
        try:
            return Tensor(data=y_np.astype(np.float32), device=self.device)
        except TypeError:
            # Fallback: shape-only construction, then attempt to populate via a public method if available
            out = Tensor(y_np.shape, self.device)

            if hasattr(out, "from_numpy") and callable(getattr(out, "from_numpy")):
                out.from_numpy(y_np.astype(np.float32))
                return out

            if hasattr(out, "copy_from_numpy") and callable(
                getattr(out, "copy_from_numpy")
            ):
                out.copy_from_numpy(y_np.astype(np.float32))
                return out

            # No public way to set array yet -> fail loudly & clearly
            raise RuntimeError(
                "Tensor constructor does not accept data=..., and no public method "
                "exists to load NumPy data into a Tensor. "
                "Add Tensor(data=..., device=...) support or a public from_numpy()/copy_from_numpy()."
            )
