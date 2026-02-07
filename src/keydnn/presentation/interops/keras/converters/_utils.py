# src/keydnn/presentation/interops/keras/converters/_utils.py
"""
Shared utilities for Keras interoperability converters.

This module provides small, reusable helpers for:
- resolving device and dtype from a conversion context,
- inspecting Keras activation configuration,
- copying NumPy arrays into KeyDNN Parameters/Tensors using public APIs.

The functions here are intentionally framework-light and do not import
TensorFlow/Keras at module import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import KerasInteropError


def resolve_device(ctx: Any):
    """
    Resolve the target device from the conversion context.

    Parameters
    ----------
    ctx : Any
        Conversion context that may define a `device` attribute.

    Returns
    -------
    Device
        The resolved KeyDNN device. Defaults to CPU if not provided in `ctx`.
    """
    dev = getattr(ctx, "device", None)
    if dev is not None:
        return dev
    from .....domain.device._device import Device

    return Device("cpu")


def resolve_dtype(ctx: Any):
    """
    Resolve the target dtype from the conversion context.

    Parameters
    ----------
    ctx : Any
        Conversion context that may define a `dtype` attribute.

    Returns
    -------
    Any
        The resolved dtype. Defaults to `np.float32` if not provided in `ctx`.
    """
    return getattr(ctx, "dtype", np.float32)


def is_linear_activation(act: Any) -> bool:
    """
    Check whether a Keras activation represents a linear (identity) transform.

    Parameters
    ----------
    act : Any
        Keras activation callable or object (commonly `tf.keras.activations.*`).

    Returns
    -------
    bool
        True if the activation is linear or unspecified; False otherwise.
    """
    if act is None:
        return True
    name = getattr(act, "__name__", None) or getattr(act, "name", None)
    return name == "linear"


def copy_param_from_numpy(param: Any, arr: np.ndarray) -> None:
    """
    Copy a NumPy array into a KeyDNN Parameter/Tensor using public APIs.

    Parameters
    ----------
    param : Any
        Target KeyDNN Parameter/Tensor-like object.
    arr : np.ndarray
        Source array to load. Converted to float32 before copy.

    Raises
    ------
    KerasInteropError
        If the target object does not expose a supported public API for loading
        NumPy arrays.
    """
    arr = np.asarray(arr, dtype=np.float32)

    fn = getattr(param, "from_numpy", None)
    if callable(fn):
        fn(arr)
        return

    fn = getattr(param, "copy_from_numpy", None)
    if callable(fn):
        fn(arr)
        return

    fn = getattr(param, "copy_from", None)
    if callable(fn):
        try:
            fn(arr)
            return
        except TypeError:
            pass

    raise KerasInteropError(
        f"Cannot load numpy data into {type(param).__name__}. "
        "Expected Parameter.from_numpy(...) or Parameter.copy_from_numpy(...)."
    )
