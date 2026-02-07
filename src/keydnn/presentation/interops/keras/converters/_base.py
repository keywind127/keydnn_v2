# src/keydnn/presentation/interops/keras/converters/_base.py
"""
Converter base types for Keras interoperability.

This module defines the minimal interface that all Keras-to-KeyDNN converters
must implement. Converters are responsible for:

1) constructing an equivalent KeyDNN module from a Keras layer object, and
2) transferring learned parameters (weights) from the Keras layer into the
   constructed KeyDNN module.

The base types here are intentionally small to keep converter implementations
consistent while avoiding unnecessary inheritance complexity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

KD = TypeVar("KD")  # KeyDNN layer type (optional)


class KerasInteropError(RuntimeError):
    """
    Raised when Keras-to-KeyDNN conversion cannot be performed safely.

    This error is intended for user-facing failures such as unsupported layer
    configurations, missing weights, or incompatible parameter layouts.
    """

    pass


@dataclass(frozen=True)
class BaseConverter(ABC, Generic[KD]):
    """
    Abstract base class for Keras-to-KeyDNN converters.

    Subclasses must implement:
    - `build`: create the corresponding KeyDNN module from a Keras layer.
    - `load_weights`: copy weights from the Keras layer into the KeyDNN module.

    Notes
    -----
    Converter instances are typically registered in a layer registry and used by
    an importer that orchestrates conversion of an entire Keras model.
    """

    @abstractmethod
    def build(self, k_layer: Any, ctx: Any) -> KD:
        """
        Construct an equivalent KeyDNN module for the given Keras layer.

        Parameters
        ----------
        k_layer : Any
            A Keras layer instance.
        ctx : Any
            Conversion context carrying settings such as device and dtype.

        Returns
        -------
        KD
            The constructed KeyDNN module.
        """
        ...

    @abstractmethod
    def load_weights(self, kd_layer: KD, k_layer: Any, ctx: Any) -> None:
        """
        Copy learned parameters from the Keras layer into the KeyDNN module.

        Parameters
        ----------
        kd_layer : KD
            The destination KeyDNN module created by `build`.
        k_layer : Any
            The source Keras layer instance.
        ctx : Any
            Conversion context carrying settings such as device and dtype.
        """
        ...
