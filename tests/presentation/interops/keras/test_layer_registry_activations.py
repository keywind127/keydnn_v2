import unittest
from unittest import TestCase

from src.keydnn.presentation.interops.keras.context import KerasImportContext
from src.keydnn.presentation.interops.keras.layer_registry import build_registry


class _Dense: ...


class _Activation: ...


class _ReLU: ...


class _LeakyReLU: ...


class _Sigmoid: ...


class _Tanh: ...


class _Softmax: ...


class _FakeLayers:
    Dense = _Dense
    Activation = _Activation
    ReLU = _ReLU
    LeakyReLU = _LeakyReLU
    Sigmoid = _Sigmoid
    Tanh = _Tanh
    Softmax = _Softmax


class _FakeKeras:
    layers = _FakeLayers()


class _FakeTF:
    keras = _FakeKeras()


class TestLayerRegistryActivations(TestCase):
    def test_registry_includes_activation_layers(self):
        ctx = KerasImportContext(device="cpu")
        reg = build_registry(_FakeTF(), ctx=ctx)

        for cls in (_Activation, _ReLU, _LeakyReLU, _Sigmoid, _Tanh, _Softmax):
            self.assertIn(
                cls, reg.mapping, f"Missing registry entry for {cls.__name__}"
            )


if __name__ == "__main__":
    unittest.main()
