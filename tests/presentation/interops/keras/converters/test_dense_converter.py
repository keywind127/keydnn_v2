# tests/presentation/interops/keras/converters/test_dense_converter.py

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

# Converter under test
from src.keydnn.presentation.interops.keras.converters.dense import (
    DenseConverter,
    KerasInteropError,
)


def _tf_available() -> bool:
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def _get_tf():
    import tensorflow as tf

    return tf


class _Ctx:
    """Minimal ctx object for converters."""

    def __init__(self, device: Device, dtype=np.float32):
        self.device = device
        self.dtype = dtype


def _param_to_numpy(param):
    """
    Best-effort conversion of a KeyDNN Parameter/Tensor-like object to numpy.

    Returns:
        np.ndarray if possible, otherwise None
    """
    if hasattr(param, "to_numpy") and callable(getattr(param, "to_numpy")):
        return np.asarray(param.to_numpy())

    # Sometimes Parameter stores a CPU ndarray in `.data`
    if hasattr(param, "data") and isinstance(getattr(param, "data"), np.ndarray):
        return np.asarray(param.data)

    return None


@unittest.skipUnless(
    _tf_available(), "TensorFlow not installed; skipping Keras interop tests."
)
class TestKerasDenseConverter(TestCase):
    def setUp(self):
        self.tf = _get_tf()

    def _build_keras_dense_layer(
        self,
        *,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        activation: str = "linear",
        seed: int = 0,
    ):
        """
        Create and BUILD a Keras Dense layer by running a forward pass.
        """
        tf = self.tf
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=out_features,
                    use_bias=use_bias,
                    activation=activation,
                    input_shape=(in_features,),
                )
            ]
        )
        x = np.zeros((2, in_features), dtype=np.float32)
        _ = model(x)  # build
        return model.layers[0]

    def test_build_materializes_keydnn_dense_with_correct_shapes(self):
        k_layer = self._build_keras_dense_layer(
            in_features=3, out_features=4, use_bias=True
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter(allow_non_linear_activation=False)

        kd = conv.build(k_layer, ctx)

        # KeyDNN Dense should be built/materialized
        self.assertTrue(
            getattr(kd, "is_built"),
            "Expected KeyDNN Dense to be materialized by converter.build().",
        )
        self.assertEqual(int(getattr(kd, "in_features")), 3)
        self.assertEqual(int(getattr(kd, "out_features")), 4)

        self.assertIsNotNone(getattr(kd, "weight", None))
        self.assertEqual(tuple(kd.weight.shape), (4, 3))

        self.assertIsNotNone(getattr(kd, "bias", None))
        self.assertEqual(tuple(kd.bias.shape), (4,))

        # Device should be set to ctx.device
        self.assertEqual(str(getattr(kd, "device")), "cpu")

    def test_load_weights_copies_kernel_transposed_and_bias(self):
        in_f, out_f = 3, 2
        k_layer = self._build_keras_dense_layer(
            in_features=in_f, out_features=out_f, use_bias=True
        )

        # Set deterministic Keras weights
        kernel = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )  # (in, out) = (3,2)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        k_layer.set_weights([kernel, bias])

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter(allow_non_linear_activation=False)

        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        # Expect KeyDNN weight (out, in) = kernel.T
        w_np = _param_to_numpy(kd.weight)
        b_np = _param_to_numpy(kd.bias)

        expected_w = kernel.T
        expected_b = bias

        if w_np is not None:
            np.testing.assert_allclose(w_np, expected_w, rtol=0, atol=0)
        else:
            self.assertEqual(tuple(kd.weight.shape), expected_w.shape)

        if b_np is not None:
            np.testing.assert_allclose(b_np, expected_b, rtol=0, atol=0)
        else:
            self.assertEqual(tuple(kd.bias.shape), expected_b.shape)

    def test_build_rejects_non_linear_activation_by_default(self):
        k_layer = self._build_keras_dense_layer(
            in_features=3, out_features=4, use_bias=True, activation="relu"
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter(allow_non_linear_activation=False)

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_allows_non_linear_activation_when_enabled(self):
        k_layer = self._build_keras_dense_layer(
            in_features=3, out_features=4, use_bias=True, activation="relu"
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter(allow_non_linear_activation=True)

        kd = conv.build(k_layer, ctx)
        self.assertTrue(getattr(kd, "is_built"))

    def test_unbuilt_keras_dense_raises(self):
        tf = self.tf
        # Create layer but DO NOT build it (no forward pass)
        k_layer = tf.keras.layers.Dense(units=4, use_bias=True, activation="linear")

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_forward_parity_matches_keras(self):
        tf = self.tf

        in_f, out_f = 3, 2
        k_layer = self._build_keras_dense_layer(
            in_features=in_f, out_features=out_f, use_bias=True
        )

        kernel = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        k_layer.set_weights([kernel, bias])

        # Convert
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = DenseConverter()
        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        # Same input to both
        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)

        y_keras = tf.keras.backend.eval(k_layer(x_np))

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

        np.testing.assert_allclose(y_kd_np, y_keras, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
