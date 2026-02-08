import unittest
from unittest import TestCase

from src.keydnn.presentation.interops.keras.context import KerasImportContext
from src.keydnn.presentation.interops.keras.layer_registry import (
    LayerRegistry,
    UnsupportedKerasLayerError,
    build_registry,
)


class _FakeDense:  # acts as tf.keras.layers.Dense class
    pass


class _FakeActivation:  # acts as tf.keras.layers.Activation class
    pass


class _FakeReLU:  # acts as tf.keras.layers.ReLU class
    pass


class _FakeLeakyReLU:  # acts as tf.keras.layers.LeakyReLU class
    pass


class _FakeSoftmax:  # acts as tf.keras.layers.Softmax class
    pass


class _FakeConv2D:
    pass


class _FakeFlatten:
    pass


class _FakeDropout:
    pass


class _FakeMaxPooling2D:
    pass


class _FakeAveragePooling2D:
    pass


class _FakeGlobalAveragePooling2D:
    pass


class _FakeBatchNormalization:
    pass


class _FakeKerasLayers:
    Dense = _FakeDense
    Conv2D = _FakeConv2D
    Flatten = _FakeFlatten
    Dropout = _FakeDropout
    MaxPooling2D = _FakeMaxPooling2D
    AveragePooling2D = _FakeAveragePooling2D
    GlobalAveragePooling2D = _FakeGlobalAveragePooling2D
    Activation = _FakeActivation
    ReLU = _FakeReLU
    LeakyReLU = _FakeLeakyReLU
    Softmax = _FakeSoftmax
    BatchNormalization = _FakeBatchNormalization


class _FakeKeras:
    layers = _FakeKerasLayers()


class _FakeTF:
    keras = _FakeKeras()


class _Ctx:
    def __init__(self, strict=True, allow_non_linear_activation=False):
        self.strict = strict
        self.allow_non_linear_activation = allow_non_linear_activation
        self.device = "cpu"


class TestLayerRegistry(TestCase):
    def test_build_registry_contains_dense(self):
        ctx = KerasImportContext(
            device="cpu", strict=True, allow_non_linear_activation=False
        )
        reg = build_registry(_FakeTF(), ctx=ctx)
        self.assertIsInstance(reg, LayerRegistry)
        self.assertIn(_FakeDense, reg.mapping)

    def test_get_returns_converter_for_known_layer(self):
        ctx = KerasImportContext(device="cpu")
        reg = build_registry(_FakeTF(), ctx=ctx)

        conv = reg.get(_FakeDense())
        self.assertIsNotNone(conv)

    def test_require_raises_when_strict_and_missing(self):
        ctx = KerasImportContext(device="cpu", strict=True)
        reg = LayerRegistry(mapping={})

        class UnknownLayer:
            pass

        with self.assertRaises(UnsupportedKerasLayerError):
            _ = reg.require(UnknownLayer(), ctx=ctx)

    def test_require_returns_none_when_not_strict_and_missing(self):
        ctx = KerasImportContext(device="cpu", strict=False)
        reg = LayerRegistry(mapping={})

        class UnknownLayer:
            pass

        conv = reg.require(UnknownLayer(), ctx=ctx)
        self.assertIsNone(conv)


if __name__ == "__main__":
    unittest.main()
