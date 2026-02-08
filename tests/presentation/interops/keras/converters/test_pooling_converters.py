# tests/presentation/interops/keras/converters/test_pooling_converters.py
"""
Unit tests for Keras -> KeyDNN pooling converters.

Covered converters
------------------
- MaxPooling2DConverter
- AveragePooling2DConverter
- GlobalAveragePooling2DConverter

These tests validate:
- Correct hyperparameter mapping from Keras to KeyDNN pooling modules.
- Default stride behavior (Keras strides=None defaults to pool_size).
- Padding mapping rules:
  - "valid" -> (0, 0)
  - "same"  -> (k//2, k//2) only when stride=(1,1) and odd kernel sizes
- NCHW requirement (data_format="channels_first") is enforced.
- load_weights is a no-op (pooling layers are parameter-free).

Notes
-----
TensorFlow is treated as an optional dependency; tests are skipped when
TensorFlow is not installed.
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

from src.keydnn.presentation.interops.keras.converters._base import KerasInteropError
from src.keydnn.presentation.interops.keras.converters.pooling import (
    MaxPooling2DConverter,
    AveragePooling2DConverter,
    GlobalAveragePooling2DConverter,
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
class TestKerasPoolingConverters(TestCase):
    """
    Test suite for pooling converters.

    This suite focuses on configuration mapping and Phase 1 constraints.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()

    def test_maxpool2d_build_maps_pool_size_stride_padding_valid(self):
        """
        MaxPooling2DConverter should map pool_size/strides/padding('valid') correctly.
        """
        tf = self.tf
        k_layer = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 3),
            strides=(1, 2),
            padding="valid",
            data_format="channels_first",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = MaxPooling2DConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "MaxPool2d")
        self.assertEqual(tuple(getattr(kd, "kernel_size")), (2, 3))
        self.assertEqual(tuple(getattr(kd, "stride")), (1, 2))
        self.assertEqual(tuple(getattr(kd, "padding")), (0, 0))

        # Parameter-free: load_weights no-op
        conv.load_weights(kd, k_layer, ctx)

    def test_avgpool2d_build_maps_pool_size_default_stride(self):
        """
        AveragePooling2DConverter should default stride to pool_size when Keras strides=None.
        """
        tf = self.tf
        k_layer = tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            strides=None,
            padding="valid",
            data_format="channels_first",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = AveragePooling2DConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "AvgPool2d")
        self.assertEqual(tuple(getattr(kd, "kernel_size")), (3, 3))
        self.assertEqual(tuple(getattr(kd, "stride")), (3, 3))
        self.assertEqual(tuple(getattr(kd, "padding")), (0, 0))

        conv.load_weights(kd, k_layer, ctx)

    def test_maxpool2d_same_padding_stride1_odd_kernel_maps_static_padding(self):
        """
        padding='same' should map to (k//2, k//2) when stride=(1,1) and odd pool_size.
        """
        tf = self.tf
        k_layer = tf.keras.layers.MaxPooling2D(
            pool_size=(5, 3),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = MaxPooling2DConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(tuple(getattr(kd, "padding")), (2, 1))

    def test_avgpool2d_same_padding_rejects_stride_not_one(self):
        """
        padding='same' with stride != (1,1) should be rejected (static padding limitation).
        """
        tf = self.tf
        k_layer = tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            strides=(2, 1),
            padding="same",
            data_format="channels_first",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = AveragePooling2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_maxpool2d_same_padding_rejects_even_kernel(self):
        """
        padding='same' with even pool_size should be rejected in Phase 1.
        """
        tf = self.tf
        k_layer = tf.keras.layers.MaxPooling2D(
            pool_size=(4, 3),  # even k_h
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = MaxPooling2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_pooling_rejects_channels_last(self):
        """
        data_format='channels_last' should be rejected for pooling converters.
        """
        tf = self.tf

        k_max = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            data_format="channels_last",
        )
        k_avg = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            data_format="channels_last",
        )
        k_gap = tf.keras.layers.GlobalAveragePooling2D(
            data_format="channels_last",
            keepdims=True,
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)

        with self.assertRaises(KerasInteropError):
            _ = MaxPooling2DConverter().build(k_max, ctx)

        with self.assertRaises(KerasInteropError):
            _ = AveragePooling2DConverter().build(k_avg, ctx)

        with self.assertRaises(KerasInteropError):
            _ = GlobalAveragePooling2DConverter().build(k_gap, ctx)

    def test_global_average_pooling_builds_keydnn_module(self):
        """
        GlobalAveragePooling2DConverter should map to KeyDNN GlobalAvgPool2d.
        """
        tf = self.tf
        k_layer = tf.keras.layers.GlobalAveragePooling2D(
            data_format="channels_first",
            keepdims=True,
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = GlobalAveragePooling2DConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "GlobalAvgPool2d")

        conv.load_weights(kd, k_layer, ctx)


if __name__ == "__main__":
    unittest.main()
