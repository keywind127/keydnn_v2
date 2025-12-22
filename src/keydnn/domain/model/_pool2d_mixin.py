"""
Configuration mixins for 2D pooling layers.

This module defines `Pool2dConfigMixin`, a lightweight mixin that provides
JSON-serializable configuration hooks for stateless 2D pooling modules
(e.g., MaxPool2d, AvgPool2d).

The mixin standardizes how pooling layers expose and reconstruct their
hyperparameters (`kernel_size`, `stride`, `padding`), enabling consistent
model serialization, deserialization, and reproducibility across the
framework.

Design notes
------------
- Intended for stateless pooling layers whose behavior is fully determined
  by structural hyperparameters.
- Assumes the host class defines `kernel_size`, `stride`, and `padding`
  attributes or properties.
- Uses plain Python types (lists, ints) to ensure JSON compatibility.
- Provides a `from_config` constructor to support model loading, cloning,
  and checkpoint restoration.
- Implemented as a mixin to avoid inheritance constraints and to keep
  pooling modules focused on computation logic.
"""

from typing import Dict, Any, TypeVar, Type


T = TypeVar("T", bound="Pool2dConfigMixin")


class Pool2dConfigMixin:
    """
    Mixin providing JSON serialization hooks for 2D pooling modules.

    This mixin assumes the host class exposes the following attributes
    or properties:
    - kernel_size : tuple[int, int]
    - stride      : tuple[int, int]
    - padding     : tuple[int, int]

    It is intended for stateless pooling layers such as MaxPool2d
    and AvgPool2d.
    """

    # ---------------------------------------------------------------------
    # JSON serialization
    # ---------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-serializable configuration for this pooling layer.
        """
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        return {
            "kernel_size": [int(k_h), int(k_w)],
            "stride": [int(s_h), int(s_w)],
            "padding": [int(p_h), int(p_w)],
        }

    @classmethod
    def from_config(cls: Type[T], cfg: Dict[str, Any]) -> T:
        """
        Reconstruct the pooling layer from a JSON configuration dict.
        """
        return cls(
            kernel_size=tuple(cfg["kernel_size"]),
            stride=tuple(cfg["stride"]),
            padding=tuple(cfg["padding"]),
        )
