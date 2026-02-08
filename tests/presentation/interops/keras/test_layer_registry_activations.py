import unittest
from unittest import TestCase

from src.keydnn.presentation.interops.keras.context import KerasImportContext
from src.keydnn.presentation.interops.keras.layer_registry import build_registry


class _Dense:
    """Fake tf.keras.layers.Dense class."""

    ...


class _Conv2D:
    """Fake tf.keras.layers.Conv2D class."""

    ...


class _Flatten:
    """Fake tf.keras.layers.Flatten class."""

    ...


class _Dropout:
    """Fake tf.keras.layers.Dropout class."""

    ...


class _MaxPooling2D:
    """Fake tf.keras.layers.MaxPooling2D class."""

    ...


class _AveragePooling2D:
    """Fake tf.keras.layers.AveragePooling2D class."""

    ...


class _GlobalAveragePooling2D:
    """Fake tf.keras.layers.GlobalAveragePooling2D class."""

    ...


class _Activation:
    """Fake tf.keras.layers.Activation class."""

    ...


class _ReLU:
    """Fake tf.keras.layers.ReLU class."""

    ...


class _LeakyReLU:
    """Fake tf.keras.layers.LeakyReLU class."""

    ...


class _Sigmoid:
    """Fake tf.keras.layers.Sigmoid class."""

    ...


class _Tanh:
    """Fake tf.keras.layers.Tanh class."""

    ...


class _Softmax:
    """Fake tf.keras.layers.Softmax class."""

    ...


class _BatchNormalization:
    """Fake tf.keras.layers.BatchNormalization class."""

    ...


class _Conv2DTranspose:
    """Fake tf.keras.layers.Conv2DTranspose class."""

    ...


class _LayerNormalization:
    """Fake tf.keras.layers.LayerNormalization class."""

    ...


class _FakeLayers:
    """
    Fake `tf.keras.layers` namespace.

    This namespace includes all layer types referenced by `build_registry`
    so registry construction can be tested without importing TensorFlow.
    """

    # Core layers
    Dense = _Dense
    Conv2D = _Conv2D
    Flatten = _Flatten
    Dropout = _Dropout

    # Pooling layers
    MaxPooling2D = _MaxPooling2D
    AveragePooling2D = _AveragePooling2D
    GlobalAveragePooling2D = _GlobalAveragePooling2D

    # Activation layers
    Activation = _Activation
    ReLU = _ReLU
    LeakyReLU = _LeakyReLU
    Sigmoid = _Sigmoid
    Tanh = _Tanh
    Softmax = _Softmax

    # Batchnorm layer
    BatchNormalization = _BatchNormalization

    # Conv2D transpose layer
    Conv2DTranspose = _Conv2DTranspose

    # Layer normalization layer
    LayerNormalization = _LayerNormalization


class _FakeKeras:
    """Fake `tf.keras` namespace."""

    layers = _FakeLayers()


class _FakeTF:
    """Fake `tensorflow` module."""

    keras = _FakeKeras()


class TestLayerRegistryActivations(TestCase):
    """
    Tests for activation registrations in the Keras layer registry.

    This suite verifies that activation-related Keras layer classes are present
    in the registry mapping produced by `build_registry`.
    """

    def test_registry_includes_activation_layers(self):
        """
        `build_registry` should include activation layer entries when present on tf.keras.layers.
        """
        ctx = KerasImportContext(device="cpu")
        reg = build_registry(_FakeTF(), ctx=ctx)

        for cls in (_Activation, _ReLU, _LeakyReLU, _Sigmoid, _Tanh, _Softmax):
            self.assertIn(
                cls, reg.mapping, f"Missing registry entry for {cls.__name__}"
            )


if __name__ == "__main__":
    unittest.main()
