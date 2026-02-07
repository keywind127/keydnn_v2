# src/keydnn/presentation/interops/keras/layer_registry.py
"""
Layer converter registry for Keras interoperability.

This module maps concrete Keras layer classes to converter instances.
The registry is constructed lazily (after TensorFlow import) to avoid
introducing TensorFlow as a hard dependency for KeyDNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from .context import KerasImportContext
from .converters._base import BaseConverter, KerasInteropError
from .converters.dense import DenseConverter
from .converters.activations import (
    ActivationConverter,
    SigmoidConverter,
    ReLUConverter,
    LeakyReLUConverter,
    TanhConverter,
    SoftmaxConverter,
)


class UnsupportedKerasLayerError(KerasInteropError):
    """
    Raised when a Keras layer type is not supported by the interop registry.
    """

    pass


@dataclass(frozen=True)
class LayerRegistry:
    """
    Registry holding Keras layer class -> converter mappings.

    Parameters
    ----------
    mapping : Dict[Type[Any], BaseConverter[Any]]
        A mapping from Keras layer classes to converter instances.
    """

    mapping: Dict[Type[Any], BaseConverter[Any]]

    def get(self, k_layer: Any) -> Optional[BaseConverter[Any]]:
        """
        Return the converter for the given Keras layer instance, or None if not found.
        """
        return self.mapping.get(type(k_layer), None)

    def require(self, k_layer: Any, *, ctx: KerasImportContext) -> BaseConverter[Any]:
        """
        Return the converter for the given Keras layer instance, raising if missing.
        """
        conv = self.get(k_layer)
        if conv is None:
            if ctx.strict:
                raise UnsupportedKerasLayerError(
                    f"Unsupported Keras layer type: {type(k_layer).__name__}"
                )
            return None  # type: ignore[return-value]
        return conv


def build_registry(tf: Any, *, ctx: KerasImportContext) -> LayerRegistry:
    """
    Build and return a registry for supported Keras layers.

    Parameters
    ----------
    tf : Any
        Imported TensorFlow module.
    ctx : KerasImportContext
        Conversion context used to configure converter behavior.

    Returns
    -------
    LayerRegistry
        The constructed registry.
    """
    mapping: Dict[type[Any], BaseConverter[Any]] = {
        # Core layers
        tf.keras.layers.Dense: DenseConverter(
            allow_non_linear_activation=bool(ctx.allow_non_linear_activation)
        ),
        # Generic activation wrapper
        tf.keras.layers.Activation: ActivationConverter(),
        # Explicit activation layers
        tf.keras.layers.ReLU: ReLUConverter(),
        tf.keras.layers.LeakyReLU: LeakyReLUConverter(),
        tf.keras.layers.Sigmoid: SigmoidConverter(),
        tf.keras.layers.Tanh: TanhConverter(),
        tf.keras.layers.Softmax: SoftmaxConverter(),
    }
    return LayerRegistry(mapping=mapping)
