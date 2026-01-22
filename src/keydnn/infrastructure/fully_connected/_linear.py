# =========================
# file: _linear.py
# =========================
"""
Eager `Linear` layer implementation.

This module defines `Linear`, a registered, eager fully-connected layer that
materializes its parameters at construction time. All numerical behavior is
implemented in `_BaseLinear`, ensuring parity with `Dense`.

Serialization
-------------
`Linear` is decorated with `@register_module()` so it can be reconstructed from
a config node. Parameter values must be loaded separately via the state payload
mechanism.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np

from ..module._serialization_core import register_module
from ...domain.device._device import Device

from ._base_linear import _BaseLinear


@register_module()
class Linear(_BaseLinear):
    """
    Fully-connected (affine) layer with eager parameter allocation.

    `Linear` allocates `weight` and (optionally) `bias` during initialization,
    making it immediately usable for both training and inference.

    Notes
    -----
    This class preserves the historical `Linear(in_features, out_features, ...)`
    API while delegating all core functionality to `_BaseLinear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        initializer: str = "xavier_uniform",
    ) -> None:
        """
        Initialize an eager Linear layer and materialize parameters.

        Parameters
        ----------
        in_features : int
            Number of input features per example.
        out_features : int
            Number of output features per example.
        bias : bool, optional
            Whether to include a learnable bias term. Defaults to True.
        device : Optional[Device], optional
            Device placement. Defaults to CPU if not provided.
        dtype : Any, optional
            Parameter dtype. Defaults to float32 if not provided.
        initializer : str, optional
            Weight initializer name. Bias (if enabled) uses zeros initializer.

        Raises
        ------
        ValueError
            If `in_features` or `out_features` is not positive.
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        dev = device if device is not None else Device("cpu")
        super().__init__(
            out_features=int(out_features),
            bias=bool(bias),
            device=dev,
            dtype=dtype,
            initializer=str(initializer),
            in_features=int(in_features),
        )
        self._materialize(int(in_features), device=dev)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Linear":
        """
        Construct a Linear layer from a configuration dict.

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
        dtype_name = cfg.get("dtype", "float32")
        initializer = cfg.get("initializer", "xavier_uniform")
        dt = np.dtype(str(dtype_name))

        return cls(
            in_features=int(cfg["in_features"]),
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=Device(str(dev)),
            dtype=dt,
            initializer=str(initializer),
        )


__all__ = [
    "Linear",
]
