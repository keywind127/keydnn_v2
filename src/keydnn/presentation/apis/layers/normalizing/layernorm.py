"""
Presentation-layer Layer Normalization wrapper for KeyDNN.

This module provides a user-facing `LayerNorm` class that subclasses the
infrastructure implementation while applying ergonomic defaults.

Why a subclass (instead of a factory)?
--------------------------------------
A factory function hides the underlying type and prevents users from accessing
classmethod/staticmethod APIs on the symbol they import. Using a subclass:

- preserves `LayerNorm.from_config(...)` and other class-level helpers
- keeps `LayerNorm` as a real type for `isinstance` checks and tooling
- avoids repeating the Tensor wrapper mistake (losing classmethods)

Design intent
-------------
- Default `device` to CPU when not explicitly specified.
- Accept flexible device specifications (`Device`, string, or None).
- Delegate all computation, parameter registration, autograd behavior, and
  serialization logic to the infrastructure layer.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .....domain.device._device import Device
from .....infrastructure.layers._layernorm import LayerNorm as _InfraLayerNorm


def _normalize_device(device: Optional[Any]) -> Device:
    """
    Normalize a user-facing device specification into a `Device`.

    Parameters
    ----------
    device : Device | str | None
        - If None, defaults to `Device("cpu")`.
        - If str, parsed as `Device(device)`.
        - If already a `Device`, returned as-is.

    Returns
    -------
    Device
        Normalized device object.

    Raises
    ------
    TypeError
        If `device` is not None/str/Device.
    """
    if device is None:
        return Device("cpu")
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        return Device(device)
    raise TypeError(f"`device` must be Device | str | None, got {type(device)!r}.")


class LayerNorm(_InfraLayerNorm):
    """
    Presentation-layer LayerNorm with ergonomic defaults.

    This class subclasses the infrastructure `LayerNorm` and only adjusts
    constructor ergonomics:

    - `device` becomes optional and defaults to CPU.
    - `device` may be provided as `Device` or a string like `"cuda:0"`.

    All numerical behavior, parameter management, and autograd logic are
    inherited unchanged from the infrastructure implementation.
    """

    def __init__(
        self,
        normalized_shape: Iterable[int],
        *,
        device: Optional[Any] = None,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        """
        Initialize a LayerNorm layer.

        Parameters
        ----------
        normalized_shape : Iterable[int]
            The shape of the dimensions to be normalized. Must match the trailing
            dimensions of the input tensor.
        device : Device | str | None, optional
            Target device for parameters and outputs. Defaults to CPU if omitted.
            If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
            `Device` instance.
        eps : float, default=1e-5
            Small constant added to variance for numerical stability.
        affine : bool, default=True
            Whether to include learnable affine parameters (gamma and beta).
        """
        dev = _normalize_device(device)
        super().__init__(
            normalized_shape=normalized_shape,
            device=dev,
            eps=float(eps),
            affine=bool(affine),
        )


__all__ = ["LayerNorm"]
