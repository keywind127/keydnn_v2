"""
CPU-only parameter initialization helpers.

This module intentionally contains NumPy usage and serves as the boundary
between backend array generation and framework tensors/parameters.

Why this exists
---------------
Higher-level modules (e.g., Conv2d, Linear) should not depend on NumPy for
numerical work or data creation, so we isolate CPU RNG + array math here.
This also prepares the codebase for future device-specific initializers
(CUDA, etc.) without touching module code.
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np

from ...domain.device._device import Device
from .._parameter import Parameter


def _normalize_device(device: Optional[Device]) -> Device:
    return device if device is not None else Device("cpu")


def _normalize_dtype(dtype: Any) -> np.dtype:
    """
    Normalize dtype inputs to a NumPy dtype for CPU array creation.

    Accepts:
    - numpy dtype objects (np.float32, np.dtype("float32"))
    - strings ("float32")
    - None (defaults to float32)
    """
    if dtype is None:
        return np.dtype(np.float32)
    try:
        return np.dtype(dtype)
    except Exception:
        # Last-resort fallback
        return np.dtype(np.float32)


def param_zeros(
    shape: tuple[int, ...], *, device: Optional[Device], dtype: Any
) -> Parameter:
    """
    Create a trainable Parameter initialized to zeros (CPU boundary).
    """
    dev = _normalize_device(device)
    dt = _normalize_dtype(dtype)

    arr = np.zeros(shape, dtype=dt)
    p = Parameter(shape=arr.shape, device=dev, requires_grad=True, ctx=None)
    p.copy_from_numpy(arr.astype(np.float32, copy=False))
    return p


def kaiming_normal_conv2d_weight(
    out_channels: int,
    in_channels: int,
    k_h: int,
    k_w: int,
    *,
    device: Optional[Device],
    dtype: Any,
) -> Parameter:
    """
    He/Kaiming normal initialization for Conv2d weights.

    weight shape: (out_channels, in_channels, k_h, k_w)
    """
    dev = _normalize_device(device)
    dt = _normalize_dtype(dtype)

    fan_in = int(in_channels) * int(k_h) * int(k_w)
    scale = float(np.sqrt(2.0 / float(fan_in)))

    w = (np.random.randn(out_channels, in_channels, k_h, k_w).astype(dt)) * scale
    p = Parameter(shape=w.shape, device=dev, requires_grad=True, ctx=None)
    p.copy_from_numpy(w.astype(np.float32, copy=False))
    return p
