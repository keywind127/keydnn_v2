"""
Presentation-layer Batch Normalization wrappers for KeyDNN.

This module defines user-facing `BatchNorm1d` and `BatchNorm2d` classes that
subclass the infrastructure implementations while applying ergonomic defaults.

Why subclasses (instead of factories)?
--------------------------------------
Using subclasses preserves *type-like behavior* and avoids accidentally hiding
classmethod/staticmethod APIs that may exist now or be added later. In practice
this means:

- `BatchNorm1d.from_config(...)` and other class-level helpers continue to work.
- The object remains a real class type for `isinstance` checks and tooling.
- Documentation and introspection remain consistent.

Design intent
-------------
- Default `device` to CPU when not explicitly specified.
- Accept flexible `device` inputs (`Device`, string, or None).
- Delegate all computation, parameter/buffer registration, autograd behavior,
  and serialization logic to the infrastructure layer.

Notes
-----
- These presentation classes do not override numerical behavior.
- Device mismatch checks in the infrastructure `forward()` remain strict.
"""

from __future__ import annotations

from typing import Any, Optional

from .....domain.device._device import Device
from .....infrastructure.layers._batchnorm import BatchNorm1d as _InfraBatchNorm1d
from .....infrastructure.layers._batchnorm import BatchNorm2d as _InfraBatchNorm2d


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


class BatchNorm1d(_InfraBatchNorm1d):
    """
    Presentation-layer BatchNorm1d with ergonomic defaults.

    This class subclasses the infrastructure `BatchNorm1d` implementation and
    only adjusts constructor ergonomics:

    - `device` becomes optional and defaults to CPU.
    - `device` may be provided as `Device` or a string like `"cuda:0"`.

    All numerical behavior, buffer updates, and autograd logic are inherited
    unchanged from the infrastructure implementation.
    """

    def __init__(
        self,
        num_features: int,
        *,
        device: Optional[Any] = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        """
        Initialize a BatchNorm1d layer.

        Parameters
        ----------
        num_features : int
            Number of feature channels C (the second dimension of input tensors).
        device : Device | str | None, optional
            Target device for parameters and buffers. Defaults to CPU if omitted.
            If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
            `Device` instance.
        eps : float, default=1e-5
            Small constant added to variance for numerical stability.
        momentum : float, default=0.1
            Exponential moving average factor for running statistics.
        affine : bool, default=True
            Whether to include learnable affine parameters (gamma and beta).
        """
        dev = _normalize_device(device)
        super().__init__(
            int(num_features),
            device=dev,
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
        )


class BatchNorm2d(_InfraBatchNorm2d):
    """
    Presentation-layer BatchNorm2d with ergonomic defaults.

    This class subclasses the infrastructure `BatchNorm2d` implementation and
    only adjusts constructor ergonomics:

    - `device` becomes optional and defaults to CPU.
    - `device` may be provided as `Device` or a string like `"cuda:0"`.

    All numerical behavior, buffer updates, and autograd logic are inherited
    unchanged from the infrastructure implementation.
    """

    def __init__(
        self,
        num_features: int,
        *,
        device: Optional[Any] = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        """
        Initialize a BatchNorm2d layer.

        Parameters
        ----------
        num_features : int
            Number of channels C (the second dimension of NCHW input tensors).
        device : Device | str | None, optional
            Target device for parameters and buffers. Defaults to CPU if omitted.
            If a string is provided (e.g., "cpu", "cuda:0"), it is parsed into a
            `Device` instance.
        eps : float, default=1e-5
            Small constant added to variance for numerical stability.
        momentum : float, default=0.1
            Exponential moving average factor for running statistics.
        affine : bool, default=True
            Whether to include learnable affine parameters (gamma and beta).
        """
        dev = _normalize_device(device)
        super().__init__(
            int(num_features),
            device=dev,
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
