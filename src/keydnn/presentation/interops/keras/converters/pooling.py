# src/keydnn/presentation/interops/keras/converters/pooling.py
"""
Keras pooling layer converters.

This module provides converters for common 2D pooling layers from Keras to
KeyDNN pooling modules.

Supported Keras layers (Phase 1)
--------------------------------
- tf.keras.layers.MaxPooling2D
- tf.keras.layers.AveragePooling2D
- tf.keras.layers.GlobalAveragePooling2D

Mapping summary
---------------
All KeyDNN pooling implementations assume **NCHW** layout.

Keras pooling layers commonly default to `data_format="channels_last"`.
Phase 1 conversion requires `data_format="channels_first"`.

Keras MaxPooling2D / AveragePooling2D
- pool_size    -> kernel_size
- strides      -> stride (if None in Keras, defaults to pool_size)
- padding      -> static padding pair
  - "valid" -> (0, 0)
  - "same"  -> supported only when:
      - stride == (1, 1)
      - pool_size is odd in both dimensions
    maps to (k_h//2, k_w//2)
  - explicit tuple/list -> treated as static padding pair when present

Keras GlobalAveragePooling2D
- No kernel/stride/padding parameters; converts to KeyDNN GlobalAvgPool2d

Notes
-----
- These converters are parameter-free; `load_weights` is a no-op.
- `padding="same"` is constrained for static padding compatibility in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ._base import BaseConverter, KerasInteropError


def _pair_int(v: Any, *, name: str) -> Tuple[int, int]:
    """
    Normalize an integer-or-pair value into a pair of integers.

    Parameters
    ----------
    v : Any
        Value that is either an integer or a length-2 tuple/list of integers.
    name : str
        Attribute name for error reporting.

    Returns
    -------
    Tuple[int, int]
        Normalized (h, w) pair.

    Raises
    ------
    KerasInteropError
        If the value cannot be normalized to a pair of integers.
    """
    if isinstance(v, (int, np.integer)):
        return int(v), int(v)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise KerasInteropError(f"Expected {name} to be int or pair, got {v!r}.")


def _require_channels_first(k_layer: Any) -> None:
    """
    Require NCHW semantics for conversion.

    Parameters
    ----------
    k_layer : Any
        Keras pooling layer.

    Raises
    ------
    KerasInteropError
        If `data_format` is not 'channels_first'.
    """
    data_format = getattr(k_layer, "data_format", None) or "channels_last"
    if str(data_format).lower() != "channels_first":
        raise KerasInteropError(
            "Keras pooling layers with data_format='channels_last' are not supported in Phase 1. "
            "KeyDNN pooling assumes NCHW (channels_first)."
        )


def _resolve_padding(
    k_layer: Any, *, kernel_size: Tuple[int, int], stride: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Resolve Keras pooling padding into a static (p_h, p_w) pair.

    Parameters
    ----------
    k_layer : Any
        Keras pooling layer.
    kernel_size : Tuple[int, int]
        Pool size / kernel size (k_h, k_w).
    stride : Tuple[int, int]
        Stride (s_h, s_w).

    Returns
    -------
    Tuple[int, int]
        Padding pair (p_h, p_w).

    Raises
    ------
    KerasInteropError
        If padding configuration cannot be mapped to KeyDNN's static padding.
    """
    padding = getattr(k_layer, "padding", "valid")

    if isinstance(padding, str):
        p = padding.lower()
        if p == "valid":
            return 0, 0
        if p == "same":
            if stride != (1, 1):
                raise KerasInteropError(
                    "Keras pooling padding='same' with stride != (1, 1) is not supported in Phase 1 "
                    "because KeyDNN pooling uses static padding."
                )
            k_h, k_w = kernel_size
            if (k_h % 2) != 1 or (k_w % 2) != 1:
                raise KerasInteropError(
                    "Keras pooling padding='same' requires odd pool_size in Phase 1."
                )
            return k_h // 2, k_w // 2
        raise KerasInteropError(f"Unsupported Keras pooling padding='{padding}'.")
    return _pair_int(padding, name="padding")


def _get_stride_defaulting_to_kernel(
    k_layer: Any, kernel_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Resolve Keras pooling strides with default-to-pool_size behavior.

    Parameters
    ----------
    k_layer : Any
        Keras pooling layer.
    kernel_size : Tuple[int, int]
        Pool size.

    Returns
    -------
    Tuple[int, int]
        Stride pair.
    """
    strides = getattr(k_layer, "strides", None)
    if strides is None:
        return int(kernel_size[0]), int(kernel_size[1])
    return _pair_int(strides, name="strides")


@dataclass(frozen=True)
class MaxPooling2DConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.MaxPooling2D` -> KeyDNN `MaxPool2d`.

    Phase 1 behavior
    ----------------
    - Requires NCHW (`data_format="channels_first"`).
    - Supports `padding` in {"valid", "same"} under constraints or explicit pairs.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN MaxPool2d module corresponding to a Keras MaxPooling2D layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras MaxPooling2D layer.
        ctx : Any
            Conversion context (unused; reserved for future use).

        Returns
        -------
        Any
            Constructed KeyDNN MaxPool2d module.

        Raises
        ------
        KerasInteropError
            If unsupported attributes are present or configuration is invalid.
        """
        _require_channels_first(k_layer)

        pool_size = _pair_int(getattr(k_layer, "pool_size", 2), name="pool_size")
        stride = _get_stride_defaulting_to_kernel(k_layer, pool_size)
        padding = _resolve_padding(k_layer, kernel_size=pool_size, stride=stride)

        from .....infrastructure.pooling import MaxPool2d

        return MaxPool2d(kernel_size=pool_size, stride=stride, padding=padding)

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN MaxPool2d.

        Notes
        -----
        MaxPool2d is parameter-free; this is a no-op.
        """
        return


@dataclass(frozen=True)
class AveragePooling2DConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.AveragePooling2D` -> KeyDNN `AvgPool2d`.

    Phase 1 behavior
    ----------------
    - Requires NCHW (`data_format="channels_first"`).
    - Supports `padding` in {"valid", "same"} under constraints or explicit pairs.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN AvgPool2d module corresponding to a Keras AveragePooling2D layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras AveragePooling2D layer.
        ctx : Any
            Conversion context (unused; reserved for future use).

        Returns
        -------
        Any
            Constructed KeyDNN AvgPool2d module.

        Raises
        ------
        KerasInteropError
            If unsupported attributes are present or configuration is invalid.
        """
        _require_channels_first(k_layer)

        pool_size = _pair_int(getattr(k_layer, "pool_size", 2), name="pool_size")
        stride = _get_stride_defaulting_to_kernel(k_layer, pool_size)
        padding = _resolve_padding(k_layer, kernel_size=pool_size, stride=stride)

        from .....infrastructure.pooling import AvgPool2d

        return AvgPool2d(kernel_size=pool_size, stride=stride, padding=padding)

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN AvgPool2d.

        Notes
        -----
        AvgPool2d is parameter-free; this is a no-op.
        """
        return


@dataclass(frozen=True)
class GlobalAveragePooling2DConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.GlobalAveragePooling2D` -> KeyDNN `GlobalAvgPool2d`.

    Phase 1 behavior
    ----------------
    - Requires NCHW (`data_format="channels_first"`).
    - Converts to a stateless KeyDNN GlobalAvgPool2d module.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN GlobalAvgPool2d corresponding to a Keras GlobalAveragePooling2D layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras GlobalAveragePooling2D layer.
        ctx : Any
            Conversion context (unused; reserved for future use).

        Returns
        -------
        Any
            Constructed KeyDNN GlobalAvgPool2d module.

        Raises
        ------
        KerasInteropError
            If `data_format` is not 'channels_first'.
        """
        _require_channels_first(k_layer)

        from .....infrastructure.pooling import GlobalAvgPool2d

        return GlobalAvgPool2d()

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN GlobalAvgPool2d.

        Notes
        -----
        GlobalAvgPool2d is parameter-free; this is a no-op.
        """
        return
