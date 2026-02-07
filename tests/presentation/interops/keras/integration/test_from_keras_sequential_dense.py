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
    _tf_available(), "TensorFlow not installed; skipping Keras integration tests."
)
class TestKerasImporterIntegration(TestCase):
    def setUp(self):
        import tensorflow as tf

        self.tf = tf
        tf.random.set_seed(0)
        np.random.seed(0)

    def _make_tensor(self, arr: np.ndarray):
        """
        Create a KeyDNN Tensor from numpy using public APIs only.
        """
        from src.keydnn.infrastructure.tensor._tensor import Tensor

        try:
            return Tensor(data=arr, device=Device("cpu"))
        except TypeError:
            t = Tensor(arr.shape, Device("cpu"))
            if hasattr(t, "from_numpy"):
                t.from_numpy(arr)
            else:
                t.copy_from_numpy(arr)
            return t

    def test_from_keras_single_dense_forward_parity(self):
        tf = self.tf

        # -------------------------
        # Build Keras model
        # -------------------------
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=2,
                    use_bias=True,
                    activation="linear",
                    input_shape=(3,),
                )
            ]
        )

        # Force build
        _ = model(np.zeros((1, 3), dtype=np.float32))

        # Set deterministic weights
        kernel = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
        bias = np.array([0.25, -2.0], dtype=np.float32)
        model.layers[0].set_weights([kernel, bias])

        # -------------------------
        # Convert to KeyDNN
        # -------------------------
        kd = from_keras(model, device="cpu")

        # Sequential or list fallback
        if isinstance(kd, (list, tuple)):
            self.assertEqual(len(kd), 1)
            kd_dense = kd[0]
        else:
            kd_dense = kd.modules[0] if hasattr(kd, "modules") else kd[0]

        # -------------------------
        # Forward parity check
        # -------------------------
        x_np = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )

        y_keras = tf.keras.backend.eval(model(x_np))

        x_kd = self._make_tensor(x_np)
        y_kd = kd_dense.forward(x_kd)
        y_kd_np = np.asarray(y_kd.to_numpy())

        np.testing.assert_allclose(y_kd_np, y_keras, rtol=1e-6, atol=1e-6)

    def test_from_keras_rejects_non_linear_activation(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=4,
                    activation="relu",
                    input_shape=(3,),
                )
            ]
        )
        _ = model(np.zeros((1, 3), dtype=np.float32))

        with self.assertRaises(Exception):
            _ = from_keras(model, device="cpu", allow_non_linear_activation=False)

    def test_from_keras_two_dense_layers_forward_parity(self):
        tf = self.tf

        # Keras: Dense(4) -> Dense(2)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=4,
                    use_bias=True,
                    activation="linear",
                    input_shape=(3,),
                ),
                tf.keras.layers.Dense(
                    units=2,
                    use_bias=True,
                    activation="linear",
                ),
            ]
        )
        _ = model(np.zeros((1, 3), dtype=np.float32))  # build

        # Set deterministic weights
        k1 = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        )  # (3,4)
        b1 = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float32)

        k2 = np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
                [4.0, -4.0],
            ],
            dtype=np.float32,
        )  # (4,2)
        b2 = np.array([0.25, 1.5], dtype=np.float32)

        model.layers[0].set_weights([k1, b1])
        model.layers[1].set_weights([k2, b2])

        # Convert
        kd = from_keras(model, device="cpu")

        # Normalize to list of KeyDNN layers
        if isinstance(kd, (list, tuple)):
            kd_layers = list(kd)
        else:
            if hasattr(kd, "modules"):
                kd_layers = list(kd.modules)
            elif hasattr(kd, "__iter__"):
                kd_layers = list(kd)
            else:
                kd_layers = [kd]

        self.assertEqual(len(kd_layers), 2)

        # Forward parity check (apply sequentially)
        x_np = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        )
        y_keras = tf.keras.backend.eval(model(x_np))

        x_kd = self._make_tensor(x_np)
        y_kd = kd_layers[0].forward(x_kd)
        y_kd = kd_layers[1].forward(y_kd)
        y_kd_np = np.asarray(y_kd.to_numpy())

        np.testing.assert_allclose(y_kd_np, y_keras, rtol=1e-6, atol=1e-6)

    def test_from_keras_dense_without_bias_forward_parity(self):
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=2,
                    use_bias=False,
                    activation="linear",
                    input_shape=(3,),
                )
            ]
        )
        _ = model(np.zeros((1, 3), dtype=np.float32))  # build

        kernel = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
        model.layers[0].set_weights([kernel])  # no bias

        kd = from_keras(model, device="cpu")

        if isinstance(kd, (list, tuple)):
            kd_dense = kd[0]
        else:
            kd_dense = kd.modules[0] if hasattr(kd, "modules") else kd[0]

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        x_kd = self._make_tensor(x_np)
        y_kd = kd_dense.forward(x_kd)
        y_kd_np = np.asarray(y_kd.to_numpy())

        np.testing.assert_allclose(y_kd_np, y_keras, rtol=1e-6, atol=1e-6)

    def test_from_keras_path_loads_and_converts(self):
        """
        Save a small Keras model to disk and import from path to validate
        the model_or_path branch in from_keras.
        """
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=2,
                    use_bias=True,
                    activation="linear",
                    input_shape=(3,),
                )
            ]
        )
        _ = model(np.zeros((1, 3), dtype=np.float32))  # build

        kernel = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
        bias = np.array([0.25, -2.0], dtype=np.float32)
        model.layers[0].set_weights([kernel, bias])

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.keras")
            try:
                model.save(path)
            except Exception as e:
                self.skipTest(f"Saving Keras model failed in this environment: {e}")

            kd = from_keras(path, device="cpu")

        if isinstance(kd, (list, tuple)):
            kd_dense = kd[0]
        else:
            kd_dense = kd.modules[0] if hasattr(kd, "modules") else kd[0]

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        y_keras = tf.keras.backend.eval(model(x_np))

        x_kd = self._make_tensor(x_np)
        y_kd = kd_dense.forward(x_kd)
        y_kd_np = np.asarray(y_kd.to_numpy())

        np.testing.assert_allclose(y_kd_np, y_keras, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
