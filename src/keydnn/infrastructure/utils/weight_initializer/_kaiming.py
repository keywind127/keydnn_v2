"""
Kaiming (He) weight initializers.

This module provides Kaiming (He) initialization strategies and registers them
into the global `WeightInitializer` registry.

Implemented variants
--------------------
- ``kaiming``:
    Alias of standard Kaiming normal initialization using
    ``std = sqrt(2 / fan_in)``.
- ``kaiming_relu``:
    Explicit Kaiming initialization for ReLU activations.
- ``kaiming_leaky_relu_*``:
    Kaiming initialization adjusted for LeakyReLU activations with different
    negative slopes, registered via a helper.

Notes
-----
- Fan-in is computed from the weight tensor shape via ``_calculate_fan_in``.
- All initializers mutate the provided tensor in-place and return it.
- These initializers are intended for weight tensors of Linear / Conv layers
  using ReLU-family activations.
"""

import math

import numpy as np

from ._base import WeightInitializer
from ...tensor._tensor import Tensor
from ....domain.utils._weight_initialization import _calculate_fan_in


@WeightInitializer.register_initializer("kaiming")
def kaiming(tensor: Tensor) -> Tensor:
    """
    Apply standard Kaiming (He) normal initialization.

    This initializer is equivalent to Kaiming initialization for ReLU
    activations and uses:

        std = sqrt(2 / fan_in)

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in = _calculate_fan_in(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))

    scale = math.sqrt(2.0 / float(fan_in))
    dt = np.dtype(tensor.dtype)

    w = np.random.randn(*tensor.shape).astype(dt, copy=False) * scale
    tensor.copy_from_numpy(w)
    return tensor


@WeightInitializer.register_initializer("kaiming_relu")
def kaiming_relu(tensor: Tensor) -> Tensor:
    """
    Apply Kaiming normal initialization for ReLU activations.

    This is the canonical He initialization derived specifically for ReLU:

        std = sqrt(2 / fan_in)

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in = _calculate_fan_in(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))

    std = math.sqrt(2.0 / float(fan_in))

    w = np.random.randn(*tensor.shape).astype(tensor.dtype) * std
    tensor.copy_from_numpy(w)
    return tensor


def register_kaiming_leaky_relu(name: str, *, negative_slope: float) -> None:
    """
    Register a Kaiming initializer configured for LeakyReLU.

    Despite the helper name, this function registers **Kaiming (He)** variants
    adapted for LeakyReLU activations.

    For LeakyReLU with negative slope ``a``, the Kaiming variance becomes:

        std = sqrt(2 / ((1 + a^2) * fan_in))

    Parameters
    ----------
    name:
        Registry key to associate with the initializer.
    negative_slope:
        The LeakyReLU negative slope parameter (``a``).
    """

    @WeightInitializer.register_initializer(name)
    def _init(tensor: Tensor) -> Tensor:
        """
        Apply Kaiming normal initialization with LeakyReLU gain.

        Parameters
        ----------
        tensor:
            The tensor to initialize in-place.

        Returns
        -------
        Tensor
            The initialized tensor (same object).
        """
        fan_in = _calculate_fan_in(tuple(tensor.shape))
        fan_in = max(1, int(fan_in))

        std = math.sqrt(2.0 / ((1.0 + negative_slope * negative_slope) * fan_in))

        w = np.random.randn(*tensor.shape).astype(tensor.dtype) * std
        tensor.copy_from_numpy(w)
        return tensor


register_kaiming_leaky_relu("kaiming_leaky_relu_0.2", negative_slope=0.2)
register_kaiming_leaky_relu("kaiming_leaky_relu_0.01", negative_slope=0.01)
