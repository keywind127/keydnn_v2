"""
Numerical sanity unit tests for Keras -> KeyDNN importer.

These tests run small forward-pass comparisons between:
- a real `tf.keras.Sequential` model, and
- the KeyDNN model returned by `from_keras(...)`

Covered pipelines
-----------------
1) Conv2D -> ReLU -> Flatten -> Dense
2) Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense -> Dropout
3) Conv2D -> AvgPool -> GlobalAvgPool -> Dense -> Softmax
4) Flatten -> Dense -> LeakyReLU -> Dense -> Sigmoid
5) Flatten -> Dense -> Tanh
6) LayerNorm -> Dense
7) Conv2DTranspose -> ReLU

Notes
-----
- TensorFlow is an optional dependency; tests are skipped when unavailable.
- These tests validate numerical forward parity only (no backward).
- Phase 1 constraints are respected:
  - Sequential only
  - channels_first for convolution/pooling/batchnorm
  - explicit activations (no fused Conv2D/Dense activation)
  - LayerNormalization uses trailing axis only
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._module import Module
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


def _as_list(kd_out) -> list[Module]:
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


def _set_eval_mode(kd_out) -> None:
    """
    Put a KeyDNN model/container into inference mode when supported.

    This is best-effort and covers common conventions:
    - `.eval()` if available
    - per-module `.training = False` when present
    """
    if hasattr(kd_out, "eval") and callable(getattr(kd_out, "eval")):
        kd_out.eval()
        return

    for m in _as_list(kd_out):
        if hasattr(m, "training"):
            try:
                setattr(m, "training", False)
            except Exception:
                pass


def _import_tensor_class():
    """
    Import a KeyDNN Tensor class from common public locations.

    Returns
    -------
    Any
        Tensor class.

    Raises
    ------
    ImportError
        If no Tensor class can be imported.
    """
    candidates = [
        "src.keydnn.presentation.tensor",  # if you have a presentation Tensor module
        "src.keydnn.presentation.tensor._tensor",
        "src.keydnn.infrastructure.tensor",  # fallback
        "src.keydnn.infrastructure.tensor._tensor",
    ]

    last_err = None
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["Tensor"])
            if hasattr(mod, "Tensor"):
                return getattr(mod, "Tensor")
        except Exception as e:
            last_err = e

    raise ImportError("Failed to import a KeyDNN Tensor class.") from last_err


def _tensor_from_numpy(x: np.ndarray, *, device: Device):
    """
    Create a KeyDNN Tensor and copy data from numpy.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    device : Device
        Target KeyDNN device.

    Returns
    -------
    Any
        KeyDNN Tensor instance holding `x`.
    """
    Tensor = _import_tensor_class()
    t = Tensor(shape=tuple(x.shape), device=device, requires_grad=False, ctx=None)

    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        t.copy_from_numpy(x)
        return t

    raise RuntimeError("KeyDNN Tensor does not support copy_from_numpy(...).")


def _to_numpy(t) -> np.ndarray:
    """
    Convert KeyDNN Tensor-like output to numpy.

    Parameters
    ----------
    t : Any
        KeyDNN Tensor-like object.

    Returns
    -------
    np.ndarray
        Numpy array view/copy.
    """
    if hasattr(t, "to_numpy") and callable(getattr(t, "to_numpy")):
        return np.asarray(t.to_numpy())

    if hasattr(t, "data") and isinstance(getattr(t, "data"), np.ndarray):
        return np.asarray(t.data)

    raise RuntimeError("Failed to convert KeyDNN output to numpy.")


def _forward_keydnn(kd_out, x_np: np.ndarray, *, device: Device) -> np.ndarray:
    """
    Run KeyDNN forward pass for a converted model/container.

    Parameters
    ----------
    kd_out : Any
        Converted KeyDNN model/container.
    x_np : np.ndarray
        Input array.
    device : Device
        Target KeyDNN device.

    Returns
    -------
    np.ndarray
        Output as numpy array.
    """
    x = _tensor_from_numpy(x_np, device=device)

    if callable(kd_out):
        y = kd_out(x)
        return _to_numpy(y)

    y_any = x
    for m in _as_list(kd_out):
        if not callable(m):
            raise RuntimeError(f"Converted module is not callable: {type(m).__name__}")
        y_any = m(y_any)
    return _to_numpy(y_any)


@unittest.skipUnless(
    _tf_available(), "TensorFlow not installed; skipping numerical importer tests."
)
class TestKerasImporterNumericalSanity(TestCase):
    """
    Numerical sanity tests for importer conversions.

    Each case builds a small Keras Sequential model, converts it via `from_keras`,
    and compares KeyDNN output against Keras output on the same input.
    """

    def setUp(self):
        """Import TensorFlow and seed for determinism."""
        self.tf = _get_tf()
        self.tf.random.set_seed(0)
        np.random.seed(0)
        self.device = Device("cpu")

    def _assert_model_close(
        self, *, model, x: np.ndarray, name: str, rtol=1e-4, atol=1e-4
    ):
        """
        Convert a Keras model and assert KeyDNN forward output matches Keras output.

        Parameters
        ----------
        model : Any
            tf.keras.Sequential model.
        x : np.ndarray
            Input array.
        name : str
            Case name for error context.
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        """
        # Build weights + reference output
        y_ref = model(x, training=False).numpy()

        kd = from_keras(model, device=self.device, dtype=np.float32, strict=True)
        _set_eval_mode(kd)
        y_kd = _forward_keydnn(kd, x, device=self.device)

        self.assertEqual(
            y_ref.shape,
            y_kd.shape,
            f"[{name}] shape mismatch: ref={y_ref.shape}, keydnn={y_kd.shape}",
        )
        np.testing.assert_allclose(y_kd, y_ref, rtol=rtol, atol=atol)

    def test_numerical_sanity_pipelines(self):
        """
        Numerical sanity checks for a set of supported Phase 1 pipelines.
        """
        tf = self.tf

        cases = []

        # 1) Conv2D -> ReLU -> Flatten -> Dense
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
        cases.append(("conv2d_relu_flatten_dense", model, x))

        # 2) Conv2D -> BN -> ReLU -> MaxPool -> Flatten -> Dense -> Dropout
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
        cases.append(("conv2d_bn_relu_maxpool_flatten_dense_dropout", model, x))

        # 3) Conv2D -> AvgPool -> GlobalAvgPool -> Dense -> Softmax
        # 3) Conv2D -> AvgPool -> GlobalAvgPool -> Flatten -> Dense -> Softmax
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
        cases.append(("conv2d_avgpool_globalavgpool_flatten_dense_softmax", model, x))

        # 4) Flatten -> Dense -> LeakyReLU -> Dense -> Sigmoid
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
        cases.append(("flatten_dense_leakyrelu_dense_sigmoid", model, x))

        # 5) Flatten -> Dense -> Tanh
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 5)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(7, use_bias=True, activation="linear"),
                tf.keras.layers.Activation("tanh"),
            ]
        )
        x = np.random.randn(2, 3, 5).astype(np.float32)
        cases.append(("flatten_dense_tanh", model, x))

        # 6) LayerNorm -> Dense (trailing axis)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(12,)),
                tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True),
                tf.keras.layers.Dense(6, use_bias=True, activation="linear"),
            ]
        )
        x = np.random.randn(2, 12).astype(np.float32)
        cases.append(("layernorm_dense", model, x))

        # 7) Conv2DTranspose -> ReLU
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
        cases.append(("conv2d_transpose_relu", model, x))

        for name, m, x in cases:
            with self.subTest(case=name):
                self._assert_model_close(model=m, x=x, name=name, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
