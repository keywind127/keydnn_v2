# tests/presentation/interops/keras/test_importer_real_keras.py
"""
Real Keras -> KeyDNN importer tests.

These tests validate that `from_keras(...)` can convert real `tf.keras.Sequential`
models end-to-end (no mocks) using the registered converter registry.

Covered layer combinations
--------------------------
- Conv2D / Conv2DTranspose
- Activation / ReLU / LeakyReLU / Sigmoid / Tanh / Softmax
- MaxPooling2D / AveragePooling2D / GlobalAveragePooling2D
- Flatten / Dense / Dropout
- BatchNormalization (axis=1, channels_first)
- LayerNormalization (trailing axis only)

Notes
-----
- TensorFlow is treated as an optional dependency; tests are skipped when
  TensorFlow is not installed.
- Phase 1 constraints are respected:
  - Sequential only
  - channels_first for convolution + pooling + BatchNormalization
  - Conv2D and Dense use activation="linear"; non-linearities are explicit layers
  - LayerNormalization axis is trailing (e.g., -1)
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.presentation.interops.keras.importer import from_keras


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


def _as_list(kd_out):
    """
    Normalize importer output into an ordered list of modules.

    Parameters
    ----------
    kd_out : Any
        Output returned by `from_keras(...)`.

    Returns
    -------
    list
        KeyDNN modules in order.
    """
    if isinstance(kd_out, list):
        return kd_out

    for attr in ("modules", "layers"):
        if hasattr(kd_out, attr):
            v = getattr(kd_out, attr)
            if callable(v):
                v = v()
            if isinstance(v, (list, tuple)):
                return list(v)

    return [kd_out]


def _names(kd_out):
    """
    Return lowercase class names for converted layers.

    Parameters
    ----------
    kd_out : Any
        Output returned by `from_keras(...)`.

    Returns
    -------
    list[str]
        Lowercased module type names.
    """
    return [type(m).__name__.lower() for m in _as_list(kd_out)]


@unittest.skipUnless(
    _tf_available(), "TensorFlow not installed; skipping real Keras importer tests."
)
class TestKerasImporterRealKeras(TestCase):
    """
    End-to-end importer tests using real TensorFlow/Keras models.
    """

    def setUp(self):
        """Import TensorFlow and seed for determinism."""
        self.tf = _get_tf()
        self.tf.random.set_seed(0)
        np.random.seed(0)

    def _convert(self, model, x):
        """
        Run a forward pass to build weights, then convert via `from_keras`.

        Parameters
        ----------
        model : Any
            tf.keras.Sequential model.
        x : np.ndarray
            Input array used to build the model.

        Returns
        -------
        Any
            Converted KeyDNN module/container.
        """
        _ = model(x, training=False)
        return from_keras(model, device=Device("cpu"), dtype=np.float32, strict=True)

    def test_pipeline_conv2d_relu_flatten_dense(self):
        """
        Conv2D -> ReLU -> Flatten -> Dense pipeline should convert end-to-end.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 8, 8)),
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(5, use_bias=True, activation="linear"),
            ]
        )

        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertIn("conv2d", names[0])
        self.assertTrue(any("relu" in n or "activation" in n for n in names))
        self.assertTrue(any("flatten" in n for n in names))
        self.assertTrue(any("dense" in n or "linear" in n for n in names))

    def test_pipeline_conv2d_bn_relu_maxpool_flatten_dense_dropout(self):
        """
        Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense -> Dropout pipeline.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 16, 16)),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=False,
                    activation="linear",
                ),
                tf.keras.layers.BatchNormalization(axis=1, center=True, scale=True),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, use_bias=True, activation="linear"),
                tf.keras.layers.Dropout(0.25),
            ]
        )

        x = np.random.randn(2, 3, 16, 16).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(any("batchnorm" in n for n in names))
        self.assertTrue(any("maxpool" in n for n in names))
        self.assertTrue(any("dropout" in n for n in names))

    def test_pipeline_conv2d_avgpool_globalavgpool_dense_softmax(self):
        """
        Conv2D -> AvgPool -> GlobalAvgPool -> Dense -> Softmax pipeline.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 12, 12)),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                ),
                tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4, use_bias=True, activation="linear"),
                tf.keras.layers.Softmax(),
            ]
        )

        x = np.random.randn(2, 3, 12, 12).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(any("averagepool" in n or "avgpool" in n for n in names))
        self.assertTrue(
            any("globalaveragepool" in n or "globalavg" in n for n in names)
        )
        self.assertTrue(any("softmax" in n for n in names))

    def test_pipeline_flatten_dense_leakyrelu_dense_sigmoid(self):
        """
        Flatten -> Dense -> LeakyReLU -> Dense -> Sigmoid pipeline.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(2, 3, 4)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, use_bias=True, activation="linear"),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(3, use_bias=True, activation="linear"),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )

        x = np.random.randn(2, 2, 3, 4).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(any("leaky" in n for n in names))
        self.assertTrue(any("sigmoid" in n for n in names))

    def test_pipeline_flatten_dense_tanh(self):
        """
        Flatten -> Dense -> Tanh pipeline.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 5)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(7, use_bias=True, activation="linear"),
                tf.keras.layers.Activation("tanh"),
            ]
        )

        x = np.random.randn(2, 3, 5).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(any("tanh" in n for n in names))

    def test_pipeline_layernorm_then_dense(self):
        """
        LayerNorm (trailing axis) -> Dense pipeline.

        This uses a 2D input so axis=-1 is unambiguously trailing and avoids
        any channels_first semantics.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(12,)),
                tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True),
                tf.keras.layers.Dense(6, use_bias=True, activation="linear"),
            ]
        )

        x = np.random.randn(2, 12).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(any("layernorm" in n for n in names))
        self.assertTrue(any("dense" in n or "linear" in n for n in names))

    def test_pipeline_conv2d_transpose_relu(self):
        """
        Conv2DTranspose -> ReLU pipeline should convert end-to-end.

        Notes
        -----
        Keras Conv2DTranspose defaults to channels_last; set data_format explicitly.
        """
        tf = self.tf
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4, 6, 6)),
                tf.keras.layers.Conv2DTranspose(
                    filters=3,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.ReLU(),
            ]
        )

        x = np.random.randn(2, 4, 6, 6).astype(np.float32)
        kd = self._convert(model, x)

        names = _names(kd)
        self.assertTrue(
            any(
                "conv2dtranspose" in n
                or "conv2d_transpose" in n
                or "conv2dtranspose" in n
                for n in names
            )
        )
        self.assertTrue(any("relu" in n or "activation" in n for n in names))


if __name__ == "__main__":
    unittest.main()
