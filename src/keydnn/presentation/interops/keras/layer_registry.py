# src/keydnn/presentation/interops/keras/layer_registry.py
"""
Layer converter registry for Keras interoperability.

This module maps concrete Keras layer classes to converter instances.
The registry is constructed lazily (after TensorFlow import) to avoid
introducing TensorFlow as a hard dependency for KeyDNN.

Supported layers (Phase 1)
--------------------------
Core:
- tf.keras.layers.Dense
- tf.keras.layers.Conv2D
- tf.keras.layers.Flatten
- tf.keras.layers.Dropout

Pooling (NCHW only):
- tf.keras.layers.MaxPooling2D
- tf.keras.layers.AveragePooling2D
- tf.keras.layers.GlobalAveragePooling2D

Activations:
- tf.keras.layers.Activation (relu/sigmoid/tanh/softmax)
- tf.keras.layers.ReLU
- tf.keras.layers.LeakyReLU
- tf.keras.layers.Softmax
- (optional) tf.keras.layers.Sigmoid
- (optional) tf.keras.layers.Tanh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from .context import KerasImportContext
from .converters._base import BaseConverter, KerasInteropError
from .converters.conv2d import Conv2DConverter
from .converters.dense import DenseConverter
from .converters.dropout import DropoutConverter
from .converters.flatten import FlattenConverter
from .converters.activations import (
    ActivationConverter,
    SigmoidConverter,
    ReLUConverter,
    LeakyReLUConverter,
    TanhConverter,
    SoftmaxConverter,
)
from .converters.pooling import (
    AveragePooling2DConverter,
    MaxPooling2DConverter,
    GlobalAveragePooling2DConverter,
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

        Parameters
        ----------
        k_layer : Any
            A Keras layer instance.

        Returns
        -------
        Optional[BaseConverter[Any]]
            The converter instance if registered, otherwise None.
        """
        return self.mapping.get(type(k_layer), None)

    def require(self, k_layer: Any, *, ctx: KerasImportContext) -> BaseConverter[Any]:
        """
        Return the converter for the given Keras layer instance, raising if missing.

        Parameters
        ----------
        k_layer : Any
            A Keras layer instance.
        ctx : KerasImportContext
            Import context controlling strictness and feature flags.

        Returns
        -------
        BaseConverter[Any]
            Converter registered for the layer.

        Raises
        ------
        UnsupportedKerasLayerError
            If the layer type is not registered and strict mode is enabled.
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
        tf.keras.layers.Conv2D: Conv2DConverter(),
        tf.keras.layers.Flatten: FlattenConverter(),
        tf.keras.layers.Dropout: DropoutConverter(),
        # Pooling (NCHW only)
        tf.keras.layers.MaxPooling2D: MaxPooling2DConverter(),
        tf.keras.layers.AveragePooling2D: AveragePooling2DConverter(),
        tf.keras.layers.GlobalAveragePooling2D: GlobalAveragePooling2DConverter(),
        # Generic activation wrapper
        tf.keras.layers.Activation: ActivationConverter(),
        # Explicit activation layers (common across TF/Keras builds)
        tf.keras.layers.ReLU: ReLUConverter(),
        tf.keras.layers.LeakyReLU: LeakyReLUConverter(),
        tf.keras.layers.Softmax: SoftmaxConverter(),
    }

    # Optional explicit activation layers (not present in all TF/Keras builds)
    SigmoidLayer = getattr(tf.keras.layers, "Sigmoid", None)
    if SigmoidLayer is not None:
        mapping[SigmoidLayer] = SigmoidConverter()

    TanhLayer = getattr(tf.keras.layers, "Tanh", None)
    if TanhLayer is not None:
        mapping[TanhLayer] = TanhConverter()

    return LayerRegistry(mapping=mapping)
