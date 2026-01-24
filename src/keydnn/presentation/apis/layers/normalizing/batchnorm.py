"""
Presentation-layer Batch Normalization factories for KeyDNN.

This module provides user-facing constructors for Batch Normalization layers
that wrap the infrastructure-level implementations (`BatchNorm1d`,
`BatchNorm2d`) and apply sensible defaults for high-level usage.

Design intent
-------------
- Default the device to CPU when not explicitly specified.
- Accept flexible device specifications (e.g., Device, string, or None).
- Preserve strict behavior and implementation details in the infrastructure
  layer while offering a more ergonomic API to end users.

Notes
-----
- These factory functions return infrastructure-layer module instances.
- All numerical behavior, parameter registration, and autograd logic live in
  the infrastructure implementation.
- The presentation layer does not relax device mismatch checks performed during
  forward execution.
"""

from __future__ import annotations
from typing import Optional, Any

from .....domain.device._device import Device
from .....infrastructure.layers._batchnorm import BatchNorm1d as _BatchNorm1d
from .....infrastructure.layers._batchnorm import BatchNorm2d as _BatchNorm2d


def BatchNorm1d(
    num_features: int,
    *,
    device: Optional[Any] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
) -> _BatchNorm1d:
    """
    Create a 1D Batch Normalization layer with a CPU default device.

    This is a presentation-layer factory that constructs an infrastructure-level
    `BatchNorm1d` module while providing a user-friendly default device
    (`Device("cpu")`) when none is specified.

    Parameters
    ----------
    num_features : int
        Number of feature channels C (second dimension of input tensors).
    device : Device | str | None, optional
        Target device for parameters and buffers. If None, defaults to CPU.
        If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
        `Device` instance.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    momentum : float, default=0.1
        Exponential moving average factor for running statistics.
    affine : bool, default=True
        Whether to include learnable affine parameters (gamma and beta).

    Returns
    -------
    BatchNorm1d
        An infrastructure-layer `BatchNorm1d` module instance.

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

    return _BatchNorm1d(
        int(num_features),
        device=device,
        eps=float(eps),
        momentum=float(momentum),
        affine=bool(affine),
    )


def BatchNorm2d(
    num_features: int,
    *,
    device: Optional[Any] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
) -> _BatchNorm2d:
    """
    Create a 2D Batch Normalization layer with a CPU default device.

    This is a presentation-layer factory that constructs an infrastructure-level
    `BatchNorm2d` module while providing a user-friendly default device
    (`Device("cpu")`) when none is specified.

    Parameters
    ----------
    num_features : int
        Number of channels C (second dimension of NCHW input tensors).
    device : Device | str | None, optional
        Target device for parameters and buffers. If None, defaults to CPU.
        If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
        `Device` instance.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    momentum : float, default=0.1
        Exponential moving average factor for running statistics.
    affine : bool, default=True
        Whether to include learnable affine parameters (gamma and beta).

    Returns
    -------
    BatchNorm2d
        An infrastructure-layer `BatchNorm2d` module instance.

    Notes
    -----
    - This function exists purely for API ergonomics and does not modify
      the behavior of the underlying BatchNorm implementation.
    - All computation, parameter management, and autograd logic remain in
      the infrastructure layer.
    """
    if device is None:
        device = Device("cpu")
    elif isinstance(device, str):
        device = Device(device)

    return _BatchNorm2d(
        int(num_features),
        device=device,
        eps=float(eps),
        momentum=float(momentum),
        affine=bool(affine),
    )


# Backward-compatible aliases
BatchNorm1D = BatchNorm1d
BatchNorm2D = BatchNorm2d

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm1D",
    "BatchNorm2D",
]
