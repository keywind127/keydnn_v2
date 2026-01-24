"""
Presentation-layer Layer Normalization factory for KeyDNN.

This module provides a user-facing constructor for Layer Normalization that
wraps the infrastructure-layer implementation (`infrastructure.layers._layernorm.LayerNorm`)
and applies ergonomic defaults suitable for high-level usage.

Design intent
-------------
- Default `device` to CPU when not explicitly specified.
- Accept flexible device specifications (e.g., Device instance, string, or None).
- Keep infrastructure-layer constructors strict and explicit (especially in beta),
  while providing a friendlier presentation API.

Notes
-----
- This factory returns an infrastructure-layer module instance.
- All numerical behavior, parameter registration, and autograd logic reside in
  the infrastructure implementation.
- Device mismatches between inputs and the module are still treated as errors
  during forward execution.
"""

from __future__ import annotations
from typing import Optional, Any, Iterable

from .....domain.device._device import Device
from .....infrastructure.layers._layernorm import LayerNorm as _LayerNorm


def LayerNorm(
    normalized_shape: Iterable[int],
    *,
    device: Optional[Any] = None,
    eps: float = 1e-5,
    affine: bool = True,
) -> _LayerNorm:
    """
    Create a LayerNorm module with a CPU default device.

    This is a presentation-layer factory that constructs an infrastructure-level
    `LayerNorm` module while providing a user-friendly default device
    (`Device("cpu")`) when none is specified.

    Parameters
    ----------
    normalized_shape : Iterable[int]
        The shape of the dimensions to be normalized. Must match the trailing
        dimensions of the input tensor.
    device : Device | str | None, optional
        Target device for parameters and buffers. If None, defaults to CPU.
        If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
        `Device` instance.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    affine : bool, default=True
        Whether to include learnable affine parameters (gamma and beta).

    Returns
    -------
    LayerNorm
        An infrastructure-layer `LayerNorm` module instance.

    Notes
    -----
    - This function does not alter or wrap the returned module; it only
      normalizes constructor arguments.
    - Device mismatches between inputs and the module are still treated as
      errors during forward execution.
    """
    if device is None:
        device = Device("cpu")
    elif isinstance(device, str):
        device = Device(device)

    return _LayerNorm(
        normalized_shape=normalized_shape,
        device=device,
        eps=float(eps),
        affine=bool(affine),
    )


__all__ = ["LayerNorm"]
