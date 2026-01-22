# =========================
# file: _dense.py
# =========================
"""
Lazy `Dense` (Keras-style) layer implementation.

This module defines `Dense`, a registered, lazy fully-connected layer that
infers `in_features` from the first input tensor seen during `forward`.

Unlike wrapper-based designs, `Dense` inherits from `_BaseLinear` and directly
owns its parameters (`weight`, `bias`). This ensures stable parameter naming
and improves checkpoint and optimizer-state continuity across save/load cycles.

Serialization
-------------
`Dense` is decorated with `@register_module()`. Its config includes `in_features`
if the layer has already been built, enabling deterministic reconstruction so
weights can be loaded afterward.
"""

from __future__ import annotations

from typing import Optional, Any, Dict
from dataclasses import dataclass

import numpy as np

from ..module._serialization_core import register_module
from ...domain.device._device import Device
from ._base_linear import _BaseLinear
from ..tensor._tensor import Tensor
from .._parameter import Parameter


@dataclass
class _LinearCompatView:
    """
    Backward-compatibility view that mimics the old Dense._linear contract.

    This is NOT a Module and does NOT own parameters. It simply exposes
    `.weight` and `.bias` attributes so legacy tests and utility code that
    expects `d._linear.weight` continue to work.

    Notes
    -----
    - The exposed objects are the actual `Parameter` instances owned by Dense.
    - This view should not be registered as a submodule.
    """

    weight: Parameter
    bias: Optional[Parameter]


@register_module()
class Dense(_BaseLinear):
    """
    Keras-style Dense layer with lazy input-dimension inference.

    Users specify only `out_features` at construction time. The corresponding
    `in_features` dimension is inferred from the first input tensor passed to
    `forward` (`x.shape[1]`).

    Device behavior
    ---------------
    - If `device` is None at construction time, the layer adopts `x.device` on
      the first forward pass.
    - If `device` is provided, `forward` enforces that inputs already reside on
      that device (no implicit transfers).
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        initializer: str = "xavier_uniform",
    ) -> None:
        """
        Initialize a lazy Dense layer.

        Parameters
        ----------
        out_features : int
            Number of output features per input example.
        bias : bool, optional
            Whether to include a learnable bias term. Defaults to True.
        device : Optional[Device], optional
            Desired device placement. If None, adopts the device of the first
            input tensor.
        dtype : Any, optional
            Parameter dtype. Defaults to float32 if not provided.
        initializer : str, optional
            Weight initializer name. Bias (if enabled) uses zeros initializer.

        Raises
        ------
        ValueError
            If `out_features` is not positive.
        """
        super().__init__(
            out_features=int(out_features),
            bias=bool(bias),
            device=device,
            dtype=dtype,
            initializer=str(initializer),
            in_features=None,
        )

    @property
    def _linear(self) -> Optional[_LinearCompatView]:
        """
        Backward-compatibility view for legacy code/tests.

        Returns a lightweight object exposing `.weight` and `.bias` that always
        reflects the *current* Parameter objects owned by this Dense layer.

        Notes
        -----
        - This is not a Module and is not registered as a submodule.
        - Returns None until parameters are materialized.
        """
        if getattr(self, "weight", None) is None:
            return None
        return _LinearCompatView(weight=self.weight, bias=self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Dense transform to a 2D input tensor.

        On first call, infers `in_features` from `x.shape[1]` and materializes
        parameters.

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
            If input is not 2D.
        RuntimeError
            If `device` was specified and does not match `x.device`.
        """
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Dense expects 2D input (batch, in_features), got {x_shape}"
            )

        inferred_in = int(x_shape[1])

        if self.device is None:
            self._materialize(inferred_in, device=x.device)
        else:
            if str(x.device) != str(self.device):
                raise RuntimeError(
                    f"Dense.forward device mismatch: x.device={x.device} vs layer.device={self.device}"
                )
            self._materialize(inferred_in, device=self.device)

        return super().forward(x)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Dense":
        """
        Reconstruct a Dense layer from configuration.

        If `in_features` is present, eagerly materializes parameters so a
        subsequent weight-load can attach values deterministically.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        Dense
            Reconstructed Dense module.
        """
        dev = cfg.get("device", None)
        device = Device(str(dev)) if dev is not None else None

        dtype_name = cfg.get("dtype", "float32")
        initializer = cfg.get("initializer", "xavier_uniform")
        dt = np.dtype(str(dtype_name))

        m = cls(
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=device,
            dtype=dt,
            initializer=str(initializer),
        )

        in_features = cfg.get("in_features", None)
        if in_features is not None:
            build_dev = device if device is not None else Device("cpu")
            m._materialize(int(in_features), device=build_dev)

        return m


LazyLinear = Dense

__all__ = [
    "Dense",
    "LazyLinear",
]
