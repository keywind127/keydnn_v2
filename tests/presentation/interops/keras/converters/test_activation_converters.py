import unittest
from unittest import TestCase

from src.keydnn.presentation.interops.keras.converters._base import KerasInteropError
from src.keydnn.presentation.interops.keras.converters.activations import (
    ActivationConverter,
    ReLUConverter,
    LeakyReLUConverter,
    SigmoidConverter,
    TanhConverter,
    SoftmaxConverter,
)


class _Ctx:
    device = "cpu"
    dtype = None
    strict = True
    allow_non_linear_activation = False


class _FakeActCallable:
    def __init__(self, name: str):
        self.__name__ = name


class _FakeActivationLayer:
    def __init__(self, activation):
        self.activation = activation


class _FakeSoftmaxLayer:
    def __init__(self, axis=-1):
        self.axis = axis


class _FakeLeakyReLULayerAlpha:
    def __init__(self, alpha=0.3):
        self.alpha = alpha


class _FakeLeakyReLULayerNegativeSlope:
    def __init__(self, negative_slope=0.3):
        self.negative_slope = negative_slope


class TestActivationConverters(TestCase):
    def test_relu_converter_builds_relu(self):
        conv = ReLUConverter()
        kd = conv.build(k_layer=object(), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "ReLU")
        conv.load_weights(kd, object(), _Ctx())  # no-op

    def test_sigmoid_converter_builds_sigmoid(self):
        conv = SigmoidConverter()
        kd = conv.build(k_layer=object(), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Sigmoid")
        conv.load_weights(kd, object(), _Ctx())

    def test_tanh_converter_builds_tanh(self):
        conv = TanhConverter()
        kd = conv.build(k_layer=object(), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Tanh")
        conv.load_weights(kd, object(), _Ctx())

    def test_softmax_converter_preserves_axis(self):
        conv = SoftmaxConverter()
        kd = conv.build(k_layer=_FakeSoftmaxLayer(axis=1), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Softmax")
        self.assertEqual(kd.get_config()["axis"], 1)
        conv.load_weights(kd, object(), _Ctx())

    def test_activation_converter_relu(self):
        conv = ActivationConverter()
        kd = conv.build(k_layer=_FakeActivationLayer("relu"), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "ReLU")

    def test_activation_converter_sigmoid(self):
        conv = ActivationConverter()
        kd = conv.build(k_layer=_FakeActivationLayer("sigmoid"), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Sigmoid")

    def test_activation_converter_tanh(self):
        conv = ActivationConverter()
        kd = conv.build(k_layer=_FakeActivationLayer("tanh"), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Tanh")

    def test_activation_converter_softmax_defaults_axis_last(self):
        conv = ActivationConverter()
        kd = conv.build(k_layer=_FakeActivationLayer("softmax"), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "Softmax")
        self.assertEqual(kd.get_config()["axis"], -1)

    def test_activation_converter_accepts_callable_activation(self):
        conv = ActivationConverter()
        kd = conv.build(
            k_layer=_FakeActivationLayer(_FakeActCallable("relu")), ctx=_Ctx()
        )
        self.assertEqual(type(kd).__name__, "ReLU")

    def test_activation_converter_rejects_unknown_activation(self):
        conv = ActivationConverter()
        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer=_FakeActivationLayer("gelu"), ctx=_Ctx())

    def test_activation_converter_rejects_unidentifiable_activation(self):
        conv = ActivationConverter()

        class Weird:
            pass

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer=_FakeActivationLayer(Weird()), ctx=_Ctx())

    def test_leaky_relu_converter_reads_alpha(self):
        conv = LeakyReLUConverter()
        kd = conv.build(k_layer=_FakeLeakyReLULayerAlpha(alpha=0.123), ctx=_Ctx())
        self.assertEqual(type(kd).__name__, "LeakyReLU")
        self.assertAlmostEqual(float(kd.get_config()["alpha"]), 0.123, places=7)

    def test_leaky_relu_converter_reads_negative_slope(self):
        conv = LeakyReLUConverter()
        kd = conv.build(
            k_layer=_FakeLeakyReLULayerNegativeSlope(negative_slope=0.456), ctx=_Ctx()
        )
        self.assertEqual(type(kd).__name__, "LeakyReLU")
        self.assertAlmostEqual(float(kd.get_config()["alpha"]), 0.456, places=7)


if __name__ == "__main__":
    unittest.main()
