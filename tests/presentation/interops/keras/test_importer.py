import unittest
from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np

from src.keydnn.presentation.interops.keras.importer import from_keras
from src.keydnn.presentation.interops.keras.converters._base import KerasInteropError


class _FakeKerasSequential:
    def __init__(self, layers):
        self.layers = layers


class _FakeKerasModels:
    def __init__(self, model_to_return):
        self._model_to_return = model_to_return

    def load_model(self, path, compile=False):
        self.last_args = (path, compile)
        return self._model_to_return


class _FakeKeras:
    def __init__(self, sequential_cls, models_obj):
        self.Sequential = sequential_cls
        self.models = models_obj


class _FakeTF:
    def __init__(self, sequential_cls, models_obj):
        self.keras = _FakeKeras(sequential_cls, models_obj)


class TestKerasImporter(TestCase):
    def test_from_keras_rejects_non_sequential(self):
        fake_model = object()
        fake_tf = _FakeTF(_FakeKerasSequential, _FakeKerasModels(model_to_return=None))

        with patch(
            "src.keydnn.presentation.interops.keras.importer._require_tensorflow",
            return_value=fake_tf,
        ):
            with self.assertRaises(KerasInteropError):
                _ = from_keras(fake_model)

    def test_from_keras_loads_model_from_path_with_compile_false(self):
        fake_layer = object()
        fake_seq = _FakeKerasSequential([fake_layer])
        fake_models = _FakeKerasModels(model_to_return=fake_seq)
        fake_tf = _FakeTF(_FakeKerasSequential, fake_models)

        # stub converter that does nothing
        conv = Mock()
        conv.build.return_value = "kd_layer"

        fake_registry = Mock()
        fake_registry.require.return_value = conv

        with patch(
            "src.keydnn.presentation.interops.keras.importer._require_tensorflow",
            return_value=fake_tf,
        ), patch(
            "src.keydnn.presentation.interops.keras.importer.build_registry",
            return_value=fake_registry,
        ), patch(
            "src.keydnn.presentation.interops.keras.importer._resolve_device",
            return_value="cpu_dev",
        ), patch(
            "src.keydnn.presentation.interops.keras.importer._try_make_sequential",
            side_effect=lambda xs: xs,
        ):

            out = from_keras("model.keras", device="cpu", dtype=np.float32)

        # load_model called with compile=False
        self.assertEqual(fake_models.last_args, ("model.keras", False))

        # converter methods called
        conv.build.assert_called_once()
        conv.load_weights.assert_called_once()

        # output is list of built layers (due to patched _try_make_sequential)
        self.assertEqual(out, ["kd_layer"])

    def test_from_keras_passes_context_to_registry(self):
        fake_layer = object()
        fake_seq = _FakeKerasSequential([fake_layer])
        fake_tf = _FakeTF(
            _FakeKerasSequential, _FakeKerasModels(model_to_return=fake_seq)
        )

        conv = Mock()
        conv.build.return_value = "kd_layer"
        fake_registry = Mock()
        fake_registry.require.return_value = conv

        captured = {}

        def _build_registry_capture(tf, *, ctx):
            captured["ctx"] = ctx
            return fake_registry

        with patch(
            "src.keydnn.presentation.interops.keras.importer._require_tensorflow",
            return_value=fake_tf,
        ), patch(
            "src.keydnn.presentation.interops.keras.importer.build_registry",
            side_effect=_build_registry_capture,
        ), patch(
            "src.keydnn.presentation.interops.keras.importer._resolve_device",
            return_value="cpu_dev",
        ), patch(
            "src.keydnn.presentation.interops.keras.importer._try_make_sequential",
            side_effect=lambda xs: xs,
        ):

            _ = from_keras(
                fake_seq,
                device="cpu",
                dtype=np.float16,
                strict=False,
                allow_non_linear_activation=True,
            )

        ctx = captured["ctx"]
        self.assertEqual(ctx.device, "cpu_dev")
        self.assertEqual(ctx.dtype, np.float16)
        self.assertFalse(ctx.strict)
        self.assertTrue(ctx.allow_non_linear_activation)


if __name__ == "__main__":
    unittest.main()
