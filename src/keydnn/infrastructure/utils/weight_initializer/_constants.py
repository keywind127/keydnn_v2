"""
Constant weight initializers.

This module defines simple constant-valued weight initializers and registers
them with the global `WeightInitializer` registry.

Provided initializers
---------------------
- ``zeros``:
    Initialize a tensor with all elements set to zero.
- ``ones``:
    Initialize a tensor with all elements set to one.

These initializers are typically used for bias parameters, testing, or
deterministic model setups.
"""

from ._base import WeightInitializer
from ...tensor._tensor import Tensor


@WeightInitializer.register_initializer("zeros")
def zeros(tensor: Tensor) -> Tensor:
    """
    Initialize a tensor with all elements set to zero.

    This initializer replaces the tensor's contents with a zero-filled tensor
    of the same shape, device, and data type.

    Parameters
    ----------
    tensor : Tensor
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    shape = tensor.shape
    dtype = tensor.dtype
    device = tensor.device

    tensor.copy_from(
        Tensor.zeros(
            shape=shape,
            device=device,
            dtype=dtype,
        )
    )
    return tensor


@WeightInitializer.register_initializer("ones")
def ones(tensor: Tensor) -> Tensor:
    """
    Initialize a tensor with all elements set to one.

    This initializer replaces the tensor's contents with a one-filled tensor
    of the same shape, device, and data type.

    Parameters
    ----------
    tensor : Tensor
        The tensor to initialize in-place.

    Returns
    -------
    Tensor
        The initialized tensor (same object).
    """
    shape = tensor.shape
    dtype = tensor.dtype
    device = tensor.device

    tensor.copy_from(
        Tensor.ones(
            shape=shape,
            device=device,
            dtype=dtype,
        )
    )
    return tensor
