# src/keydnn/presentation/interops/keras/converters/activations.py
"""
Keras activation layer converters.

This module provides converters for common stateless activation layers.

Supported Keras layers (Phase 1)
--------------------------------
- tf.keras.layers.ReLU
- tf.keras.layers.Activation with activation in {"relu", "sigmoid", "tanh", "softmax"}
- tf.keras.layers.Softmax (axis supported)

These converters construct the corresponding KeyDNN activation Modules and do
not load any weights (activations are parameter-free in KeyDNN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ._base import BaseConverter, KerasInteropError


def _activation_name(act: Any) -> Optional[str]:
    """
    Normalize a Keras activation specification to a short name when possible.

    Parameters
    ----------
    act : Any
        Keras activation spec. Common forms:
        - string ("relu")
        - callable (tf.keras.activations.relu)
        - object with `.name`

    Returns
    -------
    Optional[str]
        Lowercased activation name if detected, otherwise None.
    """
    if act is None:
        return None
    if isinstance(act, str):
        return act.lower()
    name = getattr(act, "__name__", None) or getattr(act, "name", None)
    if name is None:
        return None
    return str(name).lower()


def _require_activation_name(k_layer: Any) -> str:
    """
    Extract a normalized activation name from a tf.keras.layers.Activation layer.

    Raises
    ------
    KerasInteropError
        If the activation cannot be identified.
    """
    act = getattr(k_layer, "activation", None)
    name = _activation_name(act)
    if not name:
        raise KerasInteropError(
            f"Unsupported Activation spec on layer type {type(k_layer).__name__}."
        )
    return name


@dataclass(frozen=True)
class ReLUConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.ReLU -> KeyDNN ReLU.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Build a KeyDNN ReLU module.

        Returns
        -------
        Any
            KeyDNN ReLU module instance.
        """
        from .....infrastructure.activations import ReLU

        return ReLU()

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN ReLU.

        Notes
        -----
        ReLU is parameter-free; this is a no-op.
        """
        return


@dataclass(frozen=True)
class LeakyReLUConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.LeakyReLU -> KeyDNN LeakyReLU.

    Notes
    -----
    Keras has used both `alpha` and `negative_slope` to represent the negative
    slope parameter across versions. This converter accepts either attribute.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        from .....infrastructure.activations import LeakyReLU

        # Keras may expose `alpha` (common) or `negative_slope` (newer naming).
        alpha = getattr(k_layer, "alpha", None)
        if alpha is None:
            alpha = getattr(k_layer, "negative_slope", 0.3)
        return LeakyReLU(alpha=float(alpha))

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        return


@dataclass(frozen=True)
class SigmoidConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.Activation('sigmoid') -> KeyDNN Sigmoid.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        from .....infrastructure.activations import Sigmoid

        return Sigmoid()

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        return


@dataclass(frozen=True)
class TanhConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.Activation('tanh') -> KeyDNN Tanh.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        from .....infrastructure.activations import Tanh

        return Tanh()

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        return


@dataclass(frozen=True)
class SoftmaxConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.Softmax / Activation('softmax') -> KeyDNN Softmax.

    Notes
    -----
    - Keras Softmax exposes an `axis` attribute; it is preserved.
    - For Activation('softmax'), axis defaults to -1.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        from .....infrastructure.activations import Softmax

        axis = getattr(k_layer, "axis", -1)
        return Softmax(axis=int(axis))

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        return


@dataclass(frozen=True)
class ActivationConverter(BaseConverter[Any]):
    """
    Converter for tf.keras.layers.Activation -> KeyDNN activation Modules.

    Supported activations:
    - "relu"
    - "sigmoid"
    - "tanh"
    - "softmax" (axis defaults to -1)

    This converter is designed specifically for Keras' generic Activation layer.
    For tf.keras.layers.ReLU and tf.keras.layers.Softmax, use the dedicated
    converters for clearer configuration handling.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        name = _require_activation_name(k_layer)

        if name == "relu":
            from .....infrastructure.activations import ReLU

            return ReLU()

        if name == "sigmoid":
            from .....infrastructure.activations import Sigmoid

            return Sigmoid()

        if name == "tanh":
            from .....infrastructure.activations import Tanh

            return Tanh()

        if name == "softmax":
            from .....infrastructure.activations import Softmax

            return Softmax(axis=-1)

        raise KerasInteropError(f"Unsupported Activation('{name}') in Phase 1.")

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        return
