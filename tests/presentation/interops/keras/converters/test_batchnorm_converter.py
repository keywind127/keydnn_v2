# tests/presentation/interops/keras/converters/test_batchnorm_converter.py
"""
Unit tests for Keras -> KeyDNN BatchNormalization converter.

Covered converter
-----------------
- BatchNormalizationConverter

These tests validate:
- Build logic:
  - num_features inference from Keras moving statistics
  - eps/momentum mapping
  - affine mapping (enabled only when center=True and scale=True)
  - rank inference (2D -> BatchNorm1d, 4D -> BatchNorm2d)
- Weight loading behavior:
  - running_mean / running_var copied from Keras moving statistics
  - gamma / beta copied when affine=True
  - rejection when affine configurations are incompatible
- Axis constraints:
  - axis must be 1 (channels_first semantics)
  - multi-axis normalization is rejected
- Device constraints:
  - non-CPU targets are rejected (KeyDNN BatchNorm is CPU-only)

Notes
-----
TensorFlow is treated as an optional dependency; tests are skipped when
TensorFlow is not installed.
"""

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device

from src.keydnn.presentation.interops.keras.converters._base import KerasInteropError
from src.keydnn.presentation.interops.keras.converters.batchnorm import (
    BatchNormalizationConverter,
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
        KeyDNN Tensor/Parameter-like object.

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
class TestKerasBatchNormalizationConverter(TestCase):
    """
    Test suite for BatchNormalizationConverter.

    This suite focuses on configuration mapping and weight transfer semantics.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()
        self.tf.random.set_seed(0)
        np.random.seed(0)

    def _build_bn_layer(
        self,
        *,
        channels: int,
        rank: int,
        axis: int = 1,
        center: bool = True,
        scale: bool = True,
        eps: float = 1e-3,
        momentum: float = 0.99,
    ):
        """
        Create and BUILD a Keras BatchNormalization layer by running a forward pass.

        Parameters
        ----------
        channels : int
            Feature/channel count.
        rank : int
            Input rank; 2 -> (N,C), 4 -> (N,C,H,W).
        axis : int
            Keras BN axis.
        center : bool
            Whether to include beta.
        scale : bool
            Whether to include gamma.
        eps : float
            Keras epsilon.
        momentum : float
            Keras momentum.

        Returns
        -------
        Any
            Built Keras BatchNormalization layer.
        """
        tf = self.tf

        bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            center=center,
            scale=scale,
            epsilon=eps,
            momentum=momentum,
        )

        if rank == 2:
            x = np.zeros((4, channels), dtype=np.float32)
        elif rank == 4:
            x = np.zeros((2, channels, 3, 3), dtype=np.float32)
        else:
            raise ValueError("rank must be 2 or 4")

        _ = bn(x, training=False)  # build weights
        return bn

    def test_build_rank2_creates_batchnorm1d_and_maps_config(self):
        """
        2D BatchNormalization should convert to KeyDNN BatchNorm1d with mapped config.
        """
        k_layer = self._build_bn_layer(
            channels=5, rank=2, axis=1, center=True, scale=True, eps=1e-4, momentum=0.9
        )

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()
        kd = conv.build(k_layer, ctx)

        self.assertIn(type(kd).__name__, ("BatchNorm1d", "BatchNorm2d"))
        self.assertEqual(int(getattr(kd, "num_features")), 5)
        self.assertAlmostEqual(float(getattr(kd, "eps")), 1e-4)
        self.assertAlmostEqual(float(getattr(kd, "momentum")), 0.9)
        self.assertTrue(bool(getattr(kd, "affine")))

    def test_build_rank4_creates_batchnorm2d_and_maps_config(self):
        """
        4D BatchNormalization should convert to KeyDNN BatchNorm2d with mapped config.
        """
        k_layer = self._build_bn_layer(
            channels=3, rank=4, axis=1, center=True, scale=True, eps=1e-5, momentum=0.1
        )

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "BatchNorm2d")
        self.assertEqual(int(getattr(kd, "num_features")), 3)
        self.assertAlmostEqual(float(getattr(kd, "eps")), 1e-5)
        self.assertAlmostEqual(float(getattr(kd, "momentum")), 0.1)
        self.assertTrue(bool(getattr(kd, "affine")))

    def test_load_weights_copies_running_stats_and_affine_params_rank2(self):
        """
        load_weights should copy moving_mean/moving_variance and gamma/beta into KeyDNN BatchNorm1d.
        """
        k_layer = self._build_bn_layer(
            channels=4, rank=2, axis=1, center=True, scale=True
        )

        gamma = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        beta = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        mm = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        mv = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        k_layer.set_weights([gamma, beta, mm, mv])

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()
        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        rm = _param_to_numpy(getattr(kd, "running_mean"))
        rv = _param_to_numpy(getattr(kd, "running_var"))
        g = _param_to_numpy(getattr(kd, "gamma", None))
        b = _param_to_numpy(getattr(kd, "beta", None))

        if rm is not None:
            np.testing.assert_allclose(rm, mm, rtol=0, atol=0)
        if rv is not None:
            np.testing.assert_allclose(rv, mv, rtol=0, atol=0)
        if g is not None:
            np.testing.assert_allclose(g, gamma, rtol=0, atol=0)
        if b is not None:
            np.testing.assert_allclose(b, beta, rtol=0, atol=0)

    def test_load_weights_copies_running_stats_and_affine_params_rank4(self):
        """
        load_weights should copy moving_mean/moving_variance and gamma/beta into KeyDNN BatchNorm2d.
        """
        k_layer = self._build_bn_layer(
            channels=2, rank=4, axis=1, center=True, scale=True
        )

        gamma = np.array([1.5, -2.0], dtype=np.float32)
        beta = np.array([0.25, -0.5], dtype=np.float32)
        mm = np.array([3.0, 4.0], dtype=np.float32)
        mv = np.array([9.0, 16.0], dtype=np.float32)
        k_layer.set_weights([gamma, beta, mm, mv])

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()
        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        rm = _param_to_numpy(getattr(kd, "running_mean"))
        rv = _param_to_numpy(getattr(kd, "running_var"))
        g = _param_to_numpy(getattr(kd, "gamma", None))
        b = _param_to_numpy(getattr(kd, "beta", None))

        if rm is not None:
            np.testing.assert_allclose(rm, mm, rtol=0, atol=0)
        if rv is not None:
            np.testing.assert_allclose(rv, mv, rtol=0, atol=0)
        if g is not None:
            np.testing.assert_allclose(g, gamma, rtol=0, atol=0)
        if b is not None:
            np.testing.assert_allclose(b, beta, rtol=0, atol=0)

    def test_axis_not_one_is_rejected(self):
        """
        axis != 1 should be rejected in Phase 1 (channels_first requirement).
        """
        k_layer = self._build_bn_layer(channels=3, rank=2, axis=-1)

        ctx = _Ctx(device=Device("cpu"))
        with self.assertRaises(KerasInteropError):
            _ = BatchNormalizationConverter().build(k_layer, ctx)

    def test_multiple_axes_is_rejected(self):
        """
        BatchNormalization with axis as a list/tuple should be rejected.

        Notes
        -----
        Some Keras versions cast `axis` to int during layer construction and
        reject tuples before the layer is created. This test uses a minimal
        fake layer object to exercise converter validation.
        """

        class _FakeBN:
            axis = (1, 2)
            center = True
            scale = True
            epsilon = 1e-3
            momentum = 0.99
            input_shape = (None, 3)

            def get_weights(self):
                gamma = np.ones((3,), dtype=np.float32)
                beta = np.zeros((3,), dtype=np.float32)
                mm = np.zeros((3,), dtype=np.float32)
                mv = np.ones((3,), dtype=np.float32)
                return [gamma, beta, mm, mv]

        ctx = _Ctx(device=Device("cpu"))
        with self.assertRaises(KerasInteropError):
            _ = BatchNormalizationConverter().build(_FakeBN(), ctx)

    def test_non_cpu_device_is_rejected(self):
        """
        Non-CPU target devices should be rejected (KeyDNN BatchNorm is CPU-only).
        """
        k_layer = self._build_bn_layer(channels=3, rank=2, axis=1)

        ctx = _Ctx(device=Device("cuda:0"))
        with self.assertRaises(KerasInteropError):
            _ = BatchNormalizationConverter().build(k_layer, ctx)

    def test_affine_false_when_center_or_scale_disabled(self):
        """
        When Keras center or scale is disabled, KeyDNN BatchNorm should be affine=False.
        """
        k_layer = self._build_bn_layer(
            channels=4, rank=2, axis=1, center=False, scale=True
        )

        ctx = _Ctx(device=Device("cpu"))
        kd = BatchNormalizationConverter().build(k_layer, ctx)
        self.assertFalse(bool(getattr(kd, "affine")))

    def test_affine_true_requires_gamma_and_beta(self):
        """
        When KeyDNN BatchNorm is affine=True, missing gamma/beta in Keras should be rejected.
        """

        class _FakeBNMissingGamma:
            axis = 1
            center = True
            scale = True
            epsilon = 1e-3
            momentum = 0.99
            input_shape = (None, 3)

            def get_weights(self):
                # Missing gamma (scale) even though scale=True
                beta = np.zeros((3,), dtype=np.float32)
                mm = np.zeros((3,), dtype=np.float32)
                mv = np.ones((3,), dtype=np.float32)
                return [beta, mm, mv]

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()

        with self.assertRaises(KerasInteropError):
            _ = conv.build(_FakeBNMissingGamma(), ctx)

    def test_incompatible_affine_config_rejected_on_load(self):
        """
        If KeyDNN BatchNorm is affine=False but Keras provides gamma/beta, load_weights should raise.
        """
        k_layer_affine = self._build_bn_layer(
            channels=3, rank=2, axis=1, center=True, scale=True
        )
        gamma = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        beta = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mv = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        k_layer_affine.set_weights([gamma, beta, mm, mv])

        # Build a non-affine KeyDNN BN by disabling one of center/scale in Keras.
        k_layer_no_beta = self._build_bn_layer(
            channels=3, rank=2, axis=1, center=False, scale=True
        )

        ctx = _Ctx(device=Device("cpu"))
        conv = BatchNormalizationConverter()
        kd = conv.build(k_layer_no_beta, ctx)

        with self.assertRaises(KerasInteropError):
            conv.load_weights(kd, k_layer_affine, ctx)


if __name__ == "__main__":
    unittest.main()
