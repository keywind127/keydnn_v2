# src/keydnn/presentation/interops/keras/converters/dropout.py
"""
Keras Dropout layer conversion.

This module implements conversion from `tf.keras.layers.Dropout` to KeyDNN's
`Dropout` module.

Mapping summary
---------------
Keras Dropout stores:
- rate: float in [0, 1)  (probability of dropping units)

KeyDNN Dropout stores:
- p: float in [0, 1)     (probability of dropping units)

Therefore the mapping is direct:
    keydnn.Dropout(p = keras.rate)

Notes
-----
- Both Keras and KeyDNN use inverted dropout semantics (scaled during training,
  identity during inference). No weight loading is required.
- Keras `noise_shape` is not supported in Phase 1 (KeyDNN Dropout currently
  samples an elementwise mask for `x.shape`).
- Keras `seed` is not propagated in Phase 1. If deterministic dropout is
  required, control randomness via KeyDNN's global seeding utilities (if any).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._base import BaseConverter, KerasInteropError


def _require_dropout_rate(k_layer: Any) -> float:
    """
    Extract and validate the dropout rate from a Keras Dropout layer.

    Parameters
    ----------
    k_layer : Any
        A `tf.keras.layers.Dropout` instance.

    Returns
    -------
    float
        The dropout rate (drop probability), in the range [0, 1).

    Raises
    ------
    KerasInteropError
        If the layer does not expose a valid `rate`, or if the rate is out of
        range.
    """
    rate = getattr(k_layer, "rate", None)
    if rate is None:
        raise KerasInteropError(
            f"Keras Dropout layer missing 'rate' attribute: {type(k_layer).__name__}."
        )

    try:
        p = float(rate)
    except Exception as e:  # pragma: no cover
        raise KerasInteropError(f"Keras Dropout rate is not a float: {rate!r}.") from e

    if not 0.0 <= p < 1.0:
        raise KerasInteropError(f"Keras Dropout rate must be in [0, 1), got rate={p}.")

    return p


def _reject_unsupported_options(k_layer: Any) -> None:
    """
    Reject Keras Dropout options not supported by KeyDNN in Phase 1.

    Parameters
    ----------
    k_layer : Any
        A `tf.keras.layers.Dropout` instance.

    Raises
    ------
    KerasInteropError
        If unsupported options such as `noise_shape` are configured.
    """
    noise_shape = getattr(k_layer, "noise_shape", None)
    if noise_shape is not None:
        raise KerasInteropError(
            "Keras Dropout with noise_shape is not supported in Phase 1. "
            "Please remove noise_shape or implement an explicit KeyDNN dropout "
            "variant that supports broadcasted masks."
        )


@dataclass(frozen=True)
class DropoutConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.Dropout` -> KeyDNN `Dropout`.

    This converter maps Keras' `rate` directly to KeyDNN's drop probability `p`.
    The layer is parameter-free, so `load_weights` is a no-op.

    Phase 1 limitations
    -------------------
    - `noise_shape` is not supported.
    - `seed` is not propagated.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN Dropout module corresponding to a Keras Dropout layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras Dropout layer.
        ctx : Any
            Conversion context (unused; reserved for future use).

        Returns
        -------
        Any
            Constructed KeyDNN Dropout module.

        Raises
        ------
        KerasInteropError
            If the Keras Dropout rate is invalid or unsupported options are set.
        """
        _reject_unsupported_options(k_layer)
        p = _require_dropout_rate(k_layer)

        from .....infrastructure.layers._dropout import Dropout

        return Dropout(p=p)

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Load weights into KeyDNN Dropout.

        Notes
        -----
        Dropout is parameter-free; this is a no-op.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN Dropout module.
        k_layer : Any
            Source Keras Dropout layer.
        ctx : Any
            Conversion context (unused).
        """
        return
