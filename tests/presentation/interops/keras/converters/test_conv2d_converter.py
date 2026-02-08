# tests/presentation/interops/keras/converters/test_conv2d_converter.py
"""
Unit tests for the Keras -> KeyDNN Conv2D converter.

These tests verify that:
- Keras Conv2D configuration maps correctly to KeyDNN Conv2d hyperparameters.
- Kernel and bias weights are copied with the correct layout transpose.
- Unsupported Keras configurations are rejected in Phase 1:
  - data_format != "channels_first"
  - non-linear fused activation
  - groups != 1
  - dilation_rate != (1, 1)
  - padding="same" with stride != (1, 1)
  - padding="same" with even kernel sizes

Notes
-----
- TensorFlow is treated as an optional dependency; tests are skipped when
  TensorFlow is not installed.
- Keras layer weight materialization is performed via `layer.build(...)`
  instead of running a forward pass. This avoids backend-specific constraints
  around channels_first execution on CPU.
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

from src.keydnn.presentation.interops.keras.converters.conv2d import (
    Conv2DConverter,
    KerasInteropError,
)


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


class _Ctx:
    """
    Minimal ctx object for converters.

    Attributes
    ----------
    device : Device
        Target KeyDNN device for constructed layers.
    dtype : Any
        Target dtype (reserved for future use by some converters).
    """

    def __init__(self, device: Device, dtype=np.float32):
        self.device = device
        self.dtype = dtype


def _param_to_numpy(param):
    """
    Best-effort conversion of a KeyDNN Parameter/Tensor-like object to numpy.

    Parameters
    ----------
    param : Any
        KeyDNN Parameter/Tensor-like object.

    Returns
    -------
    Optional[np.ndarray]
        Numpy array if conversion is possible, otherwise None.
    """
    if hasattr(param, "to_numpy") and callable(getattr(param, "to_numpy")):
        return np.asarray(param.to_numpy())

    if hasattr(param, "data") and isinstance(getattr(param, "data"), np.ndarray):
        return np.asarray(param.data)

    return None


@unittest.skipUnless(
    _tf_available(), "TensorFlow not installed; skipping Keras interop tests."
)
class TestKerasConv2DConverter(TestCase):
    """
    Test suite for `Conv2DConverter`.

    This suite focuses on configuration mapping, parameter layout correctness,
    and rejection of unsupported configurations.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()

    def _build_keras_conv2d_layer(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        data_format="channels_first",
        activation="linear",
        dilation_rate=(1, 1),
        groups=1,
        input_hw=(8, 8),
    ):
        """
        Create and BUILD a Keras Conv2D layer by calling `layer.build(...)`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (filters).
        kernel_size : tuple, optional
            Kernel size (k_h, k_w).
        strides : tuple, optional
            Stride (s_h, s_w).
        padding : str, optional
            Padding mode ("valid" or "same").
        use_bias : bool, optional
            Whether bias is used.
        data_format : str, optional
            "channels_first" or "channels_last".
        activation : str, optional
            Activation name (expects "linear" for Phase 1).
        dilation_rate : tuple, optional
            Dilation rate (expects (1, 1) for Phase 1).
        groups : int, optional
            Groups (expects 1 for Phase 1).
        input_hw : tuple, optional
            Spatial size (H, W).

        Returns
        -------
        Any
            Built Keras Conv2D layer instance.
        """
        tf = self.tf
        layer = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_bias=use_bias,
            activation=activation,
        )

        h, w = int(input_hw[0]), int(input_hw[1])
        if str(data_format).lower() == "channels_first":
            input_shape = (None, int(in_channels), h, w)
        else:
            input_shape = (None, h, w, int(in_channels))

        layer.build(input_shape)
        return layer

    def test_build_maps_hyperparams_valid_padding(self):
        """
        Converter.build should map Keras Conv2D hyperparameters to KeyDNN Conv2d.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=5,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            use_bias=True,
            data_format="channels_first",
            activation="linear",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        kd = conv.build(k_layer, ctx)

        self.assertEqual(int(getattr(kd, "in_channels")), 3)
        self.assertEqual(int(getattr(kd, "out_channels")), 5)
        self.assertEqual(tuple(getattr(kd, "kernel_size")), (3, 3))
        self.assertEqual(tuple(getattr(kd, "stride")), (2, 2))
        self.assertEqual(tuple(getattr(kd, "padding")), (0, 0))

        self.assertIsNotNone(getattr(kd, "weight", None))
        self.assertEqual(tuple(kd.weight.shape), (5, 3, 3, 3))

        self.assertIsNotNone(getattr(kd, "bias", None))
        self.assertEqual(tuple(kd.bias.shape), (5,))

        self.assertEqual(str(getattr(kd, "weight").device), "cpu")

    def test_build_maps_same_padding_stride1_odd_kernel(self):
        """
        padding='same' should map to static padding (k//2, k//2) when stride=1 and odd kernels.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=2,
            out_channels=4,
            kernel_size=(5, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            data_format="channels_first",
            activation="linear",
        )

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        kd = conv.build(k_layer, ctx)

        self.assertEqual(tuple(getattr(kd, "padding")), (2, 1))
        self.assertIsNone(getattr(kd, "bias", None))

    def test_load_weights_transposes_kernel_and_copies_bias(self):
        """
        load_weights should copy kernel with transpose (3,2,0,1) and bias directly.
        """
        in_c, out_c = 2, 3
        k_h, k_w = 2, 3
        k_layer = self._build_keras_conv2d_layer(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(k_h, k_w),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            data_format="channels_first",
            activation="linear",
        )

        # Keras kernel layout: (k_h, k_w, in_c, out_c)
        kernel = np.arange(k_h * k_w * in_c * out_c, dtype=np.float32).reshape(
            (k_h, k_w, in_c, out_c)
        )
        bias = np.array([0.25, -1.0, 2.5], dtype=np.float32)
        k_layer.set_weights([kernel, bias])

        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        w_np = _param_to_numpy(kd.weight)
        b_np = _param_to_numpy(kd.bias)

        expected_w = kernel.transpose(3, 2, 0, 1)
        expected_b = bias

        if w_np is not None:
            np.testing.assert_allclose(w_np, expected_w, rtol=0, atol=0)
        else:
            self.assertEqual(tuple(kd.weight.shape), expected_w.shape)

        if b_np is not None:
            np.testing.assert_allclose(b_np, expected_b, rtol=0, atol=0)
        else:
            self.assertEqual(tuple(kd.bias.shape), expected_b.shape)

    def test_build_rejects_channels_last(self):
        """
        data_format='channels_last' should be rejected in Phase 1.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=4,
            data_format="channels_last",
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_non_linear_activation(self):
        """
        Fused non-linear activation should be rejected in Phase 1.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=4,
            data_format="channels_first",
            activation="relu",
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_groups(self):
        """
        groups != 1 should be rejected.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=4,
            out_channels=4,
            data_format="channels_first",
            groups=2,
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_dilation(self):
        """
        dilation_rate != (1, 1) should be rejected.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=4,
            data_format="channels_first",
            dilation_rate=(2, 1),
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_same_padding_with_stride_not_one(self):
        """
        padding='same' with stride != (1,1) should be rejected (static padding limitation).
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            data_format="channels_first",
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)

    def test_build_rejects_same_padding_with_even_kernel(self):
        """
        padding='same' with even kernel sizes should be rejected in Phase 1.
        """
        k_layer = self._build_keras_conv2d_layer(
            in_channels=3,
            out_channels=4,
            kernel_size=(4, 3),  # even k_h
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
        )
        ctx = _Ctx(device=Device("cpu"), dtype=np.float32)
        conv = Conv2DConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(k_layer, ctx)


if __name__ == "__main__":
    unittest.main()
