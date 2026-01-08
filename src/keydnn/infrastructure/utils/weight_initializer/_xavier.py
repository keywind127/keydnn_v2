"""
Xavier/Glorot weight initializers.

This module provides Xavier (Glorot) initialization strategies and registers
them into the global `WeightInitializer` registry.

Implemented variants
--------------------
- ``xavier``:
    Xavier normal initialization using ``std = sqrt(2 / (fan_in + fan_out))``.
- ``xavier_uniform``:
    Xavier uniform initialization using
    ``U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))``.
- ``xavier_relu``:
    Xavier normal with ReLU gain (``gain = sqrt(2)``).
- ``xavier_tanh``:
    Xavier normal with tanh gain (``gain = 5/3``).
- ``xavier_leaky_relu_*``:
    Xavier normal with LeakyReLU gain
    (``gain = sqrt(2 / (1 + negative_slope^2))``), registered via a helper.

Notes
-----
- Fan-in and fan-out are computed from the weight tensor shape via
  ``_calculate_fan_in_and_fan_out``.
- Initializers mutate the provided tensor in-place and return it.
- These initializers are intended for parameter tensors (e.g. Linear/Conv
  weight matrices) rather than activations.
"""

import math

import numpy as np

from ._base import WeightInitializer
from ...tensor._tensor import Tensor
from ....domain.utils._weight_initialization import _calculate_fan_in_and_fan_out


@WeightInitializer.register_initializer("xavier")
def xavier(tensor: Tensor) -> Tensor:
    """
    Apply Xavier (Glorot) normal initialization.

    This initializes weights from a zero-mean normal distribution with
    standard deviation:

        std = sqrt(2 / (fan_in + fan_out))

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))

    std = math.sqrt(2.0 / float(fan_in + fan_out))
    dt = np.dtype(tensor.dtype)

    w = np.random.randn(*tensor.shape).astype(dt, copy=False) * std
    tensor.copy_from_numpy(w)
    return tensor


@WeightInitializer.register_initializer("xavier_uniform")
def xavier_uniform(tensor: Tensor) -> Tensor:
    """
    Apply Xavier (Glorot) uniform initialization.

    This initializes weights from a uniform distribution:

        U(-bound, +bound), where bound = sqrt(6 / (fan_in + fan_out))

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))

    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    dt = np.dtype(tensor.dtype)

    w = np.random.uniform(-bound, bound, size=tensor.shape).astype(dt, copy=False)
    tensor.copy_from_numpy(w)
    return tensor


@WeightInitializer.register_initializer("xavier_relu")
def xavier_relu(tensor: Tensor) -> Tensor:
    """
    Apply Xavier normal initialization with ReLU gain.

    Uses the ReLU gain:

        gain = sqrt(2)

    and standard deviation:

        std = gain * sqrt(2 / (fan_in + fan_out))

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))

    gain = math.sqrt(2.0)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    dt = np.dtype(tensor.dtype)
    w = np.random.randn(*tensor.shape).astype(dt, copy=False) * std
    tensor.copy_from_numpy(w)
    return tensor


@WeightInitializer.register_initializer("xavier_tanh")
def xavier_tanh(tensor: Tensor) -> Tensor:
    """
    Apply Xavier normal initialization with tanh gain.

    Uses the tanh gain commonly adopted in deep learning libraries:

        gain = 5/3

    and standard deviation:

        std = gain * sqrt(2 / (fan_in + fan_out))

    Parameters
    ----------
    tensor:
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tuple(tensor.shape))
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))

    gain = 5.0 / 3.0
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    dt = np.dtype(tensor.dtype)
    w = np.random.randn(*tensor.shape).astype(dt, copy=False) * std
    tensor.copy_from_numpy(w)
    return tensor


def register_xavier_leaky_relu(name: str, *, negative_slope: float) -> None:
    """
    Register a Xavier normal initializer configured for LeakyReLU.

    The gain for LeakyReLU with negative slope ``a`` is:

        gain = sqrt(2 / (1 + a^2))

    This function registers a new initializer under ``name`` that applies:

        std = gain * sqrt(2 / (fan_in + fan_out))

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
        Apply Xavier normal initialization with LeakyReLU gain.

        Parameters
        ----------
        tensor:
            The tensor to initialize in-place.

        Returns
        -------
        Tensor
            The initialized tensor (same object).
        """
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tuple(tensor.shape))
        fan_in = max(1, int(fan_in))
        fan_out = max(1, int(fan_out))

        a = float(negative_slope)
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

        dt = np.dtype(tensor.dtype)
        w = np.random.randn(*tensor.shape).astype(dt, copy=False) * std
        tensor.copy_from_numpy(w)
        return tensor


register_xavier_leaky_relu("xavier_leaky_relu_0.2", negative_slope=0.2)
register_xavier_leaky_relu("xavier_leaky_relu_0.01", negative_slope=0.01)
