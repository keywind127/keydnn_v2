# src/keydnn/presentation/interops/keras/converters/flatten.py
"""
Keras Flatten layer conversion.

This module implements conversion from `tf.keras.layers.Flatten` to KeyDNN's
`Flatten` module.

Mapping summary
---------------
Keras Flatten:
- Reshapes an input tensor by collapsing all non-batch dimensions into one.
- Optionally accepts `data_format` ("channels_last" or "channels_first") in
  some Keras versions. For Flatten, `data_format` does not affect the output
  shape because the operation preserves axis order and only reshapes.

KeyDNN Flatten:
- Reshapes input (N, d1, d2, ..., dk) -> (N, d1*d2*...*dk)
- Stateless and parameter-free.

Notes
-----
- This converter is parameter-free; `load_weights` is a no-op.
- Unsupported or irrelevant Keras attributes are ignored when they do not
  change semantics (e.g., `data_format` on Flatten).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._base import BaseConverter, KerasInteropError


def _validate_flatten_layer(k_layer: Any) -> None:
    """
    Validate a Keras Flatten layer for conversion.

    Parameters
    ----------
    k_layer : Any
        A `tf.keras.layers.Flatten` instance.

    Raises
    ------
    KerasInteropError
        If the object does not appear to be a Keras Flatten layer, or if
        attributes indicate unsupported behavior.
    """
    if k_layer is None:
        raise KerasInteropError("Keras Flatten layer is None.")


@dataclass(frozen=True)
class FlattenConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.Flatten` -> KeyDNN `Flatten`.

    This converter constructs a stateless KeyDNN `Flatten` module.
    No parameters are loaded because Flatten has no weights.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN Flatten module corresponding to a Keras Flatten layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras Flatten layer.
        ctx : Any
            Conversion context (unused; reserved for future use).

        Returns
        -------
        Any
            Constructed KeyDNN Flatten module.

        Raises
        ------
        KerasInteropError
            If validation fails.
        """
        _validate_flatten_layer(k_layer)

        from .....infrastructure.flatten import Flatten

        return Flatten()

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN Flatten.

        Notes
        -----
        Flatten is parameter-free; this method is a no-op.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN Flatten module.
        k_layer : Any
            Source Keras Flatten layer.
        ctx : Any
            Conversion context (unused).
        """
        return
