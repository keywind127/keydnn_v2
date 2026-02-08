# tests/presentation/interops/keras/converters/test_flatten_converter.py
"""
Unit tests for the Keras -> KeyDNN Flatten converter.

These tests verify that:
- A Keras Flatten layer converts to a KeyDNN Flatten module.
- The KeyDNN Flatten forward produces the expected output shape.
- The converter is parameter-free and `load_weights` is a no-op.

Notes
-----
- TensorFlow is treated as an optional dependency; tests are skipped when
  TensorFlow is not installed.
- Forward parity is validated via shape and value equality against Keras,
  since Flatten is a pure reshape with deterministic behavior.
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

from src.keydnn.presentation.interops.keras.converters.flatten import (
    FlattenConverter,
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
class TestKerasFlattenConverter(TestCase):
    """
    Test suite for `FlattenConverter`.

    This suite validates converter construction and deterministic reshape parity.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()

    def _build_keras_flatten_layer(self, *, input_shape=(2, 3), seed: int = 0):
        """
        Create and BUILD a Keras Flatten layer by running a forward pass.

        Parameters
        ----------
        input_shape : tuple, optional
            Input shape excluding batch dimension.
        seed : int, optional
            Seed for TensorFlow/NumPy RNG setup (best-effort).

        Returns
        -------
        Any
            Built Keras Flatten layer instance (model.layers[0]).
        """
        tf = self.tf
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=tuple(input_shape)),
            ]
        )
        x = np.zeros((2,) + tuple(input_shape), dtype=np.float32)
        _ = model(x)  # build
        return model.layers[0]

    def test_build_creates_keydnn_flatten(self):
        """
        Converter.build should construct a KeyDNN Flatten module.
        """
        k_layer = self._build_keras_flatten_layer(input_shape=(2, 3))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        conv = FlattenConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "Flatten")

    def test_load_weights_is_noop(self):
        """
        Converter.load_weights should be a no-op for Flatten and not raise.
        """
        k_layer = self._build_keras_flatten_layer(input_shape=(2, 3))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        conv = FlattenConverter()
        kd = conv.build(k_layer, ctx)

        # Should not raise
        conv.load_weights(kd, k_layer, ctx)

    def test_forward_shape_and_values_match_keras(self):
        """
        KeyDNN Flatten forward should match Keras Flatten for shape and values.
        """
        tf = self.tf

        k_layer = self._build_keras_flatten_layer(input_shape=(2, 3))
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        conv = FlattenConverter()
        kd = conv.build(k_layer, ctx)

        # Deterministic input
        x_np = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[-1.0, -2.0, -3.0], [7.0, 8.0, 9.0]],
            ],
            dtype=np.float32,
        )  # (2,2,3)

        y_keras = tf.keras.backend.eval(k_layer(x_np))
        self.assertEqual(tuple(y_keras.shape), (2, 6))

        # Build KeyDNN tensor
        from src.keydnn.infrastructure.tensor._tensor import Tensor

        try:
            x_kd = Tensor(data=x_np, device=Device("cpu"))
        except TypeError:
            x_kd = Tensor(x_np.shape, Device("cpu"))
            if hasattr(x_kd, "from_numpy"):
                x_kd.from_numpy(x_np)
            else:
                x_kd.copy_from_numpy(x_np)

        y_kd = kd.forward(x_kd)
        y_kd_np = np.asarray(y_kd.to_numpy())

        self.assertEqual(tuple(y_kd_np.shape), (2, 6))
        np.testing.assert_allclose(y_kd_np, y_keras, rtol=0, atol=0)

    def test_build_rejects_none_layer(self):
        """
        Converter.build should reject a None layer input.
        """
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = FlattenConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(None, ctx)


if __name__ == "__main__":
    unittest.main()
