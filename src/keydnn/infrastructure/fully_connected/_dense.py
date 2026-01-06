"""
Dense (Keras-style) layer with lazy in_features inference.

`Dense` behaves like a Keras Dense layer and like PyTorch's LazyLinear:
users specify only `out_features`, and `in_features` is inferred from the
first input `x` at runtime (x.shape[1]).

Once built, this module delegates computation to the existing `Linear`
implementation (CPU/CUDA supported there).

Notes
-----
- Input must be 2D: (batch, in_features), consistent with `Linear`.
- If `device` is not provided, `Dense` adopts the first input's device.
- If `device` is provided, forward enforces x.device matches it (no implicit moves).
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from .._module import Module
from .._linear import Linear
from ..module._serialization_core import register_module
from ...domain.device._device import Device

from ..tensor._tensor import Tensor


@register_module()
class Dense(Module):
    """
    Keras-style Dense layer with lazy input-dimension inference.

    Parameters
    ----------
    out_features : int
        Number of output features per example.
    bias : bool, optional
        If True, include a learnable bias term. Defaults to True.
    device : Optional[Device], optional
        Desired device placement. If None, adopts the first input's device.
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer")

        self.out_features = int(out_features)
        self._use_bias = bool(bias)
        self.device = device  # may be None until first forward

        # Lazy state
        self.in_features: Optional[int] = None
        self._linear: Optional[Linear] = None  # created on first forward

    @property
    def is_built(self) -> bool:
        return self._linear is not None

    def _build(self, in_features: int, *, device: Device) -> None:
        """
        Materialize parameters by constructing an internal `Linear`.
        This is called exactly once (first forward), or during from_config
        when in_features is known.
        """
        if in_features <= 0:
            raise ValueError("in_features must be a positive integer")

        if self._linear is not None:
            # already built: enforce consistency
            if self.in_features != int(in_features):
                raise RuntimeError(
                    f"Dense already built with in_features={self.in_features}, "
                    f"but got in_features={int(in_features)}"
                )
            return

        self.in_features = int(in_features)
        self.device = device

        # Delegate all math/autograd/device paths to your proven Linear
        self._linear = Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self._use_bias,
            device=self.device,
        )
        # Register as submodule so state/serialization can see it
        self.register_module("linear", self._linear)

    def forward(self, x: Tensor) -> Tensor:
        # Validate 2D input (same contract as Linear)
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Dense expects 2D input (batch, in_features), got {x_shape}"
            )

        inferred_in = int(x_shape[1])

        # Decide device: if self.device is None, adopt x.device on first call
        if self.device is None:
            self._build(inferred_in, device=x.device)
        else:
            # enforce device match (no implicit moves)
            if str(x.device) != str(self.device):
                raise RuntimeError(
                    f"Dense.forward device mismatch: x.device={x.device} vs layer.device={self.device}"
                )
            # build if needed
            self._build(inferred_in, device=self.device)

        assert self._linear is not None
        return self._linear.forward(x)

    # -------------------------------------------------------------------------
    # Serialization (config-only) hooks
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-serializable constructor configuration.

        For lazy modules, we store `in_features` if the module has been built.
        This allows reconstructing the module structure deterministically.
        """
        return {
            "out_features": int(self.out_features),
            "bias": bool(self._use_bias),
            "device": str(self.device) if self.device is not None else None,
            "in_features": (
                int(self.in_features) if self.in_features is not None else None
            ),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Dense":
        dev = cfg.get("device", None)
        device = Device(str(dev)) if dev is not None else None

        m = cls(
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=device,
        )

        # If in_features is known (built module was saved), eagerly build
        in_features = cfg.get("in_features", None)
        if in_features is not None:
            # If device is still None here, default to CPU for deterministic rebuild
            # (weights will typically be loaded from checkpoint afterwards anyway).
            build_dev = device if device is not None else Device("cpu")
            m._build(int(in_features), device=build_dev)

        return m
