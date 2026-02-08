# tests/presentation/interops/keras/converters/test_dropout_converter.py
"""
Unit tests for the Keras -> KeyDNN Dropout converter.

These tests verify that:
- A built Keras Dropout layer converts to a KeyDNN Dropout module with the
  correct drop probability mapping (rate -> p).
- Unsupported configuration such as `noise_shape` is rejected.
- Invalid rates are rejected.
- `load_weights` is a no-op and does not error.

Notes
-----
- TensorFlow is treated as an optional dependency; tests are skipped when
  TensorFlow is not installed.
- Dropout forward parity is intentionally not tested here because stochastic
  masks make strict numerical parity fragile across frameworks and RNG streams.
  Converter tests focus on configuration correctness and error handling.
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

from src.keydnn.presentation.interops.keras.converters.dropout import (
    DropoutConverter,
    KerasInteropError,
)


def _tf_available() -> bool:
    """
    Return True if TensorFlow is importable.

    Returns
    -------
    bool
        True if `import tensorflow` succeeds, otherwise False.
    """
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def _get_tf():
    """
    Import and return TensorFlow.

    Returns
    -------
    Any
        Imported TensorFlow module.
    """
    import tensorflow as tf

    return tf


class _Ctx:
    """
    Minimal ctx object for converters.

    Attributes
    ----------
    device : Device
        Target KeyDNN device for constructed layers.
    dtype : Any
        Target dtype (reserved for future use by some converters).
    """

    def __init__(self, device: Device, dtype=np.float32):
        self.device = device
        self.dtype = dtype


@unittest.skipUnless(
    _tf_available(), "TensorFlow not installed; skipping Keras interop tests."
)
class TestKerasDropoutConverter(TestCase):
    """
    Test suite for `DropoutConverter`.

    This suite validates correct configuration mapping (rate -> p) and
    rejection of unsupported/invalid Keras Dropout options.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()

    def _build_keras_dropout_layer(
        self,
        *,
        rate: float,
        input_shape=(4,),
        seed: int = 0,
        noise_shape=None,
    ):
        """
        Create and BUILD a Keras Dropout layer by running a forward pass.

        Parameters
        ----------
        rate : float
            Dropout rate for Keras (drop probability).
        input_shape : tuple, optional
            Input shape excluding batch dimension.
        seed : int, optional
            Seed for TensorFlow/NumPy RNG setup (best-effort).
        noise_shape : Any, optional
            Optional Keras noise_shape (unsupported in converter; used to
            test rejection path).

        Returns
        -------
        Any
            Built Keras Dropout layer instance (model.layers[0]).
        """
        tf = self.tf
        tf.random.set_seed(seed)
        np.random.seed(seed)

        kwargs = dict(rate=rate, input_shape=input_shape)
        if noise_shape is not None:
            kwargs["noise_shape"] = noise_shape

        model = tf.keras.Sequential([tf.keras.layers.Dropout(**kwargs)])
        x = np.zeros((2,) + tuple(input_shape), dtype=np.float32)
        _ = model(x, training=True)  # build
        return model.layers[0]

    def test_build_maps_rate_to_keydnn_p(self):
        """
        Converter.build should map Keras Dropout `rate` to KeyDNN Dropout `p`.
        """
        k_layer = self._build_keras_dropout_layer(rate=0.25, input_shape=(3,))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        conv = DropoutConverter()
        kd = conv.build(k_layer, ctx)

        self.assertAlmostEqual(float(getattr(kd, "p")), 0.25, places=7)

        self.assertEqual(type(kd).__name__, "Dropout")

    def test_load_weights_is_noop(self):
        """
        Converter.load_weights should be a no-op for Dropout and not raise.
        """
        k_layer = self._build_keras_dropout_layer(rate=0.5, input_shape=(3,))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        conv = DropoutConverter()
        kd = conv.build(k_layer, ctx)

        conv.load_weights(kd, k_layer, ctx)

        self.assertAlmostEqual(float(getattr(kd, "p")), 0.5, places=7)

    def test_build_rejects_noise_shape(self):
        """
        Converter.build should reject Keras Dropout configured with noise_shape.
        """
        k_layer = self._build_keras_dropout_layer(
            rate=0.1, input_shape=(3,), noise_shape=(None, 1)
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DropoutConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_rate_out_of_range(self):
        """
        Converter.build should reject invalid dropout rates (<0 or >=1).
        """
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DropoutConverter()

        # rate < 0
        k_layer_neg = self._build_keras_dropout_layer(rate=0.0, input_shape=(3,))
        setattr(k_layer_neg, "rate", -0.01)  # force invalid configuration
        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer_neg, ctx)

        # rate >= 1
        k_layer_big = self._build_keras_dropout_layer(rate=0.0, input_shape=(3,))
        setattr(k_layer_big, "rate", 1.0)  # force invalid configuration
        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer_big, ctx)

    def test_build_rejects_missing_rate_attribute(self):
        """
        Converter.build should error if the Keras layer has no `rate` attribute.
        """
        tf = self.tf
        # Build a normal Dropout layer, then delete the attribute to simulate corruption
        k_layer = self._build_keras_dropout_layer(rate=0.2, input_shape=(3,))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DropoutConverter()

        # Some TF objects may block delattr; fall back to shadowing with None.
        try:
            delattr(k_layer, "rate")
        except Exception:
            setattr(k_layer, "rate", None)

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)


if __name__ == "__main__":
    unittest.main()
