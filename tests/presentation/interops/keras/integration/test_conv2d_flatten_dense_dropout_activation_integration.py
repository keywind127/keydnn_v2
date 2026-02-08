# tests/presentation/interops/keras/integration/test_conv2d_flatten_dense_dropout_activation_integration.py
"""
Integration tests for Keras -> KeyDNN Sequential conversion.

Covered pipeline
----------------
Conv2D -> Activation -> Flatten -> Dense -> Dropout -> Activation

These tests validate that `from_keras(...)`:
- Produces the expected ordered KeyDNN module sequence.
- Correctly loads Conv2D and Dense weights into KeyDNN layout.
- Preserves deterministic forward behavior for stateless layers (Flatten, activations).
- Preserves inference-time Dropout identity behavior.

Notes
-----
- TensorFlow is treated as an optional dependency; tests are skipped when
  TensorFlow is not installed.
- Forward parity for Conv2D is validated against a pure NumPy NCHW reference
  convolution implementation. This avoids backend-specific constraints in
  TensorFlow around NCHW execution on CPU.
- Dropout parity is tested in evaluation mode only (identity mapping), since
  training-time dropout is stochastic and depends on RNG streams.
"""

import unittest
from unittest import TestCase
from typing import List, Tuple, Any

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.presentation.interops.keras import from_keras


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid elementwise in float32.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid output.
    """
    x = x.astype(np.float32, copy=False)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _relu(x: np.ndarray) -> np.ndarray:
    """
    Compute ReLU elementwise in float32.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        ReLU output.
    """
    x = x.astype(np.float32, copy=False)
    return np.maximum(x, 0.0).astype(np.float32, copy=False)


def _conv2d_nchw_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> np.ndarray:
    """
    Reference NCHW Conv2D implementation (no dilation, no groups).

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C_in, H, W).
    w : np.ndarray
        Weight tensor of shape (C_out, C_in, K_h, K_w).
    b : np.ndarray | None
        Optional bias of shape (C_out,).
    stride : Tuple[int, int]
        Stride (s_h, s_w).
    padding : Tuple[int, int]
        Zero padding (p_h, p_w) applied symmetrically.

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C_out, H_out, W_out).
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    b = None if b is None else np.asarray(b, dtype=np.float32)

    n, c_in, h, w_in = x.shape
    c_out, c_in2, k_h, k_w = w.shape
    if c_in2 != c_in:
        raise ValueError(f"w has C_in={c_in2}, expected {c_in}.")

    s_h, s_w = int(stride[0]), int(stride[1])
    p_h, p_w = int(padding[0]), int(padding[1])

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    h_pad = h + 2 * p_h
    w_pad = w_in + 2 * p_w

    h_out = (h_pad - k_h) // s_h + 1
    w_out = (w_pad - k_w) // s_w + 1

    y = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

    for nn in range(n):
        for oc in range(c_out):
            for oh in range(h_out):
                ih0 = oh * s_h
                for ow in range(w_out):
                    iw0 = ow * s_w
                    acc = 0.0
                    for ic in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                acc += float(
                                    x_pad[nn, ic, ih0 + kh, iw0 + kw]
                                    * w[oc, ic, kh, kw]
                                )
                    if b is not None:
                        acc += float(b[oc])
                    y[nn, oc, oh, ow] = np.float32(acc)

    return y


def _flatten_nchw(x: np.ndarray) -> np.ndarray:
    """
    Flatten NCHW tensor to (N, -1) while preserving row-major order.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (N, ...).

    Returns
    -------
    np.ndarray
        Flattened array with shape (N, -1).
    """
    x = np.asarray(x, dtype=np.float32)
    return x.reshape((x.shape[0], -1)).astype(np.float32, copy=False)


def _dense_ref(
    x: np.ndarray, kernel_in_out: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    """
    Reference Dense computation using Keras kernel layout (in_features, out_features).

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N, in_features).
    kernel_in_out : np.ndarray
        Keras kernel of shape (in_features, out_features).
    bias : np.ndarray | None
        Optional bias of shape (out_features,).

    Returns
    -------
    np.ndarray
        Output array of shape (N, out_features).
    """
    x = np.asarray(x, dtype=np.float32)
    k = np.asarray(kernel_in_out, dtype=np.float32)
    y = x @ k
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y.astype(np.float32, copy=False)


def _set_dropout_eval_if_present(layers: List[Any]) -> None:
    """
    Set `.training = False` for any KeyDNN Dropout modules in the layer list.

    Parameters
    ----------
    layers : List[Any]
        KeyDNN layers in execution order.
    """
    for m in layers:
        if type(m).__name__ == "Dropout" and hasattr(m, "training"):
            m.training = False


def _unwrap_layers(kd_model: Any) -> List[Any]:
    """
    Normalize the importer output into a list of KeyDNN layers in order.

    Parameters
    ----------
    kd_model : Any
        Result of `from_keras(...)`.

    Returns
    -------
    List[Any]
        List of KeyDNN layers in execution order.
    """
    if isinstance(kd_model, (list, tuple)):
        return list(kd_model)
    if hasattr(kd_model, "modules"):
        return list(kd_model.modules)
    if hasattr(kd_model, "__iter__"):
        return list(kd_model)
    return [kd_model]


def _make_tensor(arr: np.ndarray):
    """
    Construct a KeyDNN Tensor from numpy using public APIs only.

    Parameters
    ----------
    arr : np.ndarray
        Numpy input.

    Returns
    -------
    Any
        KeyDNN Tensor instance on CPU.
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


def _run_keydnn_sequential(layers: List[Any], x_np: np.ndarray) -> np.ndarray:
    """
    Execute KeyDNN layers sequentially and return numpy output.

    Parameters
    ----------
    layers : List[Any]
        KeyDNN layers in execution order.
    x_np : np.ndarray
        Input numpy array.

    Returns
    -------
    np.ndarray
        Output numpy array.
    """
    y = _make_tensor(x_np)
    for layer in layers:
        y = layer.forward(y)
    return np.asarray(y.to_numpy())


@unittest.skipUnless(
    _tf_available(),
    "TensorFlow not installed; skipping Keras Conv2D integration tests.",
)
class TestKerasConv2DFlattenDenseDropoutActivationIntegration(TestCase):
    """
    Integration tests for conversion and forward parity of a small CNN pipeline.

    The pipeline uses NCHW (`channels_first`) to match KeyDNN Conv2d semantics.
    """

    def setUp(self):
        import tensorflow as tf

        self.tf = tf
        tf.random.set_seed(0)
        np.random.seed(0)

    def _build_model_valid_padding(self):
        """
        Build a Keras Sequential model with Conv2D(valid) -> ReLU -> Flatten -> Dense -> Dropout -> Sigmoid.

        Returns
        -------
        Any
            Built Keras model.
        """
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                    input_shape=(1, 5, 5),
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(3, use_bias=True, activation="linear"),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )

        # Keras 3 may treat the model as built when input_shape is provided.
        # Only build if necessary, and build the whole model (not individual layers).
        if not getattr(model, "built", False):
            model.build((None, 1, 5, 5))

        return model

    def test_conv2d_flatten_dense_dropout_sigmoid_parity_valid_padding_eval(self):
        """
        Converted KeyDNN pipeline should match a NumPy reference for inference behavior.

        Dropout is forced to evaluation mode in KeyDNN (identity), matching the
        inference-time expectation used in the NumPy reference.
        """
        tf = self.tf
        model = self._build_model_valid_padding()

        # -------------------------
        # Set deterministic weights
        # -------------------------
        # Keras Conv2D kernel: (k_h, k_w, in_c, out_c)
        conv_kernel = np.arange(3 * 3 * 1 * 2, dtype=np.float32).reshape((3, 3, 1, 2))
        conv_bias = np.array([0.25, -1.0], dtype=np.float32)
        model.layers[0].set_weights([conv_kernel, conv_bias])

        # Keras Dense kernel: (in_features, out_features)
        dense_kernel = (
            np.arange((2 * 3 * 3) * 3, dtype=np.float32).reshape((18, 3)) * 0.01
        )
        dense_bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        model.layers[3].set_weights([dense_kernel, dense_bias])

        # -------------------------
        # Build NumPy reference
        # -------------------------
        x_np = np.arange(1 * 1 * 5 * 5, dtype=np.float32).reshape((1, 1, 5, 5)) * 0.1

        # Convert Keras conv kernel to KeyDNN layout for reference computation:
        # (k_h, k_w, in_c, out_c) -> (out_c, in_c, k_h, k_w)
        w_kd = conv_kernel.transpose(3, 2, 0, 1).astype(np.float32, copy=False)

        y0 = _conv2d_nchw_ref(x_np, w_kd, conv_bias, stride=(1, 1), padding=(0, 0))
        y1 = _relu(y0)
        y2 = _flatten_nchw(y1)  # (1, 18)
        y3 = _dense_ref(y2, dense_kernel, dense_bias)  # (1, 3)
        # Dropout eval: identity
        y_ref = _sigmoid(y3)

        # -------------------------
        # Convert and run KeyDNN
        # -------------------------
        kd = from_keras(model, device="cpu")
        layers = _unwrap_layers(kd)
        self.assertEqual(len(layers), 6)

        _set_dropout_eval_if_present(layers)
        y_kd = _run_keydnn_sequential(layers, x_np)

        np.testing.assert_allclose(y_kd, y_ref, rtol=1e-6, atol=1e-6)

    def _build_model_same_padding_stride1(self):
        """
        Build a Keras Sequential model with Conv2D(same, stride=1, odd kernel) -> Flatten -> Dense.

        Returns
        -------
        Any
            Built Keras model.
        """
        tf = self.tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    data_format="channels_first",
                    use_bias=False,
                    activation="linear",
                    input_shape=(1, 5, 5),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2, use_bias=False, activation="linear"),
            ]
        )

        if not getattr(model, "built", False):
            model.build((None, 1, 5, 5))

        return model

    def test_conv2d_same_padding_stride1_parity_numpy_ref(self):
        """
        Converted KeyDNN Conv2D(padding='same') should match NumPy reference under Phase 1 constraints.

        This test covers the converter's static SAME mapping: padding=(k//2, k//2).
        """
        model = self._build_model_same_padding_stride1()

        conv_kernel = np.ones(
            (3, 3, 1, 1), dtype=np.float32
        )  # simple averaging-ish sum
        model.layers[0].set_weights([conv_kernel])

        dense_kernel = np.ones((25, 2), dtype=np.float32) * 0.1
        model.layers[2].set_weights([dense_kernel])

        x_np = (np.arange(25, dtype=np.float32).reshape((1, 1, 5, 5)) - 12.0) * 0.05

        w_kd = conv_kernel.transpose(3, 2, 0, 1).astype(np.float32, copy=False)
        y0 = _conv2d_nchw_ref(
            x_np, w_kd, None, stride=(1, 1), padding=(1, 1)
        )  # same => p=1
        y1 = _flatten_nchw(y0)  # (1, 25)
        y_ref = _dense_ref(y1, dense_kernel, None)

        kd = from_keras(model, device="cpu")
        layers = _unwrap_layers(kd)
        self.assertEqual(len(layers), 3)

        y_kd = _run_keydnn_sequential(layers, x_np)

        np.testing.assert_allclose(y_kd, y_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
