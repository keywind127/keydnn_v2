import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.presentation.interops.keras import from_keras


def _tf_available() -> bool:
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(
    _tf_available(),
    "TensorFlow not installed; skipping Keras activation integration tests.",
)
class TestKerasDenseActivationIntegration(TestCase):
    def setUp(self):
        import tensorflow as tf

        self.tf = tf
        tf.random.set_seed(0)
        np.random.seed(0)

    def _make_tensor(self, arr: np.ndarray):
        """
        Construct a KeyDNN Tensor from numpy using public APIs only.
        """
        from src.keydnn.infrastructure.tensor._tensor import Tensor

        arr = np.asarray(arr, dtype=np.float32)

        try:
            return Tensor(data=arr, device=Device("cpu"))
        except TypeError:
            t = Tensor(arr.shape, Device("cpu"))
            if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
                t.from_numpy(arr)
            else:
                t.copy_from_numpy(arr)
            return t

    def _unwrap_layers(self, kd_model):
        """
        Normalize the importer output into a list of KeyDNN layers in order.
        """
        if isinstance(kd_model, (list, tuple)):
            return list(kd_model)
        if hasattr(kd_model, "modules"):
            return list(kd_model.modules)
        if hasattr(kd_model, "__iter__"):
            return list(kd_model)
        return [kd_model]

    def _run_keydnn_sequential(self, layers, x_np: np.ndarray) -> np.ndarray:
        x = self._make_tensor(x_np)
        y = x
        for layer in layers:
            y = layer.forward(y)
        return np.asarray(y.to_numpy())

    def _build_and_set_dense_weights(self, model, kernel, bias=None):
        """
        Ensure model is built, then set Dense weights deterministically.
        """
        _ = model(np.zeros((1, kernel.shape[0]), dtype=np.float32))
        if bias is None:
            model.layers[0].set_weights([kernel.astype(np.float32)])
        else:
            model.layers[0].set_weights(
                [kernel.astype(np.float32), bias.astype(np.float32)]
            )

    def test_dense_relu_parity(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    2, use_bias=True, activation="linear", input_shape=(3,)
                ),
                tf.keras.layers.ReLU(),
            ]
        )

        kernel = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # (in=3, out=2)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        self._build_and_set_dense_weights(model, kernel, bias)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        kd = from_keras(model, device="cpu")
        layers = self._unwrap_layers(kd)
        self.assertEqual(len(layers), 2)

        y_kd = self._run_keydnn_sequential(layers, x_np)
        np.testing.assert_allclose(y_kd, y_keras, rtol=1e-6, atol=1e-6)

    def test_dense_sigmoid_parity(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    2, use_bias=True, activation="linear", input_shape=(3,)
                ),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )

        kernel = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        self._build_and_set_dense_weights(model, kernel, bias)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        kd = from_keras(model, device="cpu")
        layers = self._unwrap_layers(kd)
        self.assertEqual(len(layers), 2)

        y_kd = self._run_keydnn_sequential(layers, x_np)
        np.testing.assert_allclose(y_kd, y_keras, rtol=1e-6, atol=1e-6)

    def test_dense_tanh_parity(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    2, use_bias=True, activation="linear", input_shape=(3,)
                ),
                tf.keras.layers.Activation("tanh"),
            ]
        )

        kernel = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        self._build_and_set_dense_weights(model, kernel, bias)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        kd = from_keras(model, device="cpu")
        layers = self._unwrap_layers(kd)
        self.assertEqual(len(layers), 2)

        y_kd = self._run_keydnn_sequential(layers, x_np)
        np.testing.assert_allclose(y_kd, y_keras, rtol=1e-6, atol=1e-6)

    def test_dense_softmax_default_axis_parity(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    3, use_bias=True, activation="linear", input_shape=(3,)
                ),
                tf.keras.layers.Activation("softmax"),
            ]
        )

        kernel = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.float32,
        )
        bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        self._build_and_set_dense_weights(model, kernel, bias)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        kd = from_keras(model, device="cpu")
        layers = self._unwrap_layers(kd)
        self.assertEqual(len(layers), 2)

        y_kd = self._run_keydnn_sequential(layers, x_np)
        np.testing.assert_allclose(y_kd, y_keras, rtol=1e-6, atol=1e-6)

    def test_dense_softmax_axis_1_parity(self):
        """
        Use a 3D input to validate Softmax(axis=1) is preserved and matches Keras.

        Notes
        -----
        Phase 1 importer is Sequential-only and KeyDNN Dense is 2D-only, so this
        case requires additional converters (e.g., Reshape/Flatten) or a graph
        importer. It is intentionally skipped in Phase 1.
        """
        self.skipTest(
            "Phase 1 importer is Sequential-only and Dense is 2D-only; "
            "axis!= -1 Softmax parity is better covered in Phase 2 graph importer."
        )

    def test_dense_leaky_relu_parity(self):
        tf = self.tf

        alpha = 0.2
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    2, use_bias=True, activation="linear", input_shape=(3,)
                ),
                tf.keras.layers.LeakyReLU(alpha=alpha),
            ]
        )

        kernel = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        bias = np.array([0.25, -2.0], dtype=np.float32)
        self._build_and_set_dense_weights(model, kernel, bias)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        kd = from_keras(model, device="cpu")
        layers = self._unwrap_layers(kd)
        self.assertEqual(len(layers), 2)

        y_kd = self._run_keydnn_sequential(layers, x_np)
        np.testing.assert_allclose(y_kd, y_keras, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
