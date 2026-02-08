# tests/presentation/interops/keras/converters/test_layernorm_converter.py
"""
Unit tests for Keras -> KeyDNN LayerNormalization converter.

Covered converter
-----------------
- LayerNormalizationConverter

These tests validate:
- Build logic:
  - normalized_shape inference (from gamma/beta when available; otherwise from input shape)
  - eps mapping
  - affine mapping (enabled only when center=True and scale=True)
- Weight loading behavior:
  - gamma / beta copied when affine=True
  - rejection when affine configurations are incompatible
- Axis constraints (Phase 1):
  - supports only trailing-axis LayerNormalization:
    - axis = -1 or [-1]
    - axis = [-K, ..., -1] for K >= 1
  - rejects non-trailing axis specifications (e.g., axis=1, axis=[1,2])
- Device constraints:
  - non-CPU targets are rejected (KeyDNN LayerNorm is CPU-only)

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
from src.keydnn.presentation.interops.keras.converters.layernorm import (
    LayerNormalizationConverter,
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
class TestKerasLayerNormalizationConverter(TestCase):
    """
    Test suite for LayerNormalizationConverter.

    This suite focuses on configuration mapping, axis constraints, and
    parameter transfer semantics.
    """

    def setUp(self):
        """Import TensorFlow for tests (only executed when TF is available)."""
        self.tf = _get_tf()
        self.tf.random.set_seed(0)
        np.random.seed(0)

    def _build_ln_layer(
        self,
        *,
        input_shape,
        axis=-1,
        center: bool = True,
        scale: bool = True,
        eps: float = 1e-3,
    ):
        """
        Create and BUILD a Keras LayerNormalization layer by running a forward pass.

        Parameters
        ----------
        input_shape : tuple[int, ...]
            Full input shape including batch dimension (N, ...).
        axis : int or list[int]
            Keras LayerNormalization axis.
        center : bool
            Whether to include beta.
        scale : bool
            Whether to include gamma.
        eps : float
            Keras epsilon.

        Returns
        -------
        Any
            Built Keras LayerNormalization layer.
        """
        tf = self.tf
        ln = tf.keras.layers.LayerNormalization(
            axis=axis,
            center=center,
            scale=scale,
            epsilon=eps,
        )

        x = np.zeros(tuple(int(d) for d in input_shape), dtype=np.float32)
        _ = ln(x)  # build weights
        return ln

    def test_build_infers_normalized_shape_from_weights_axis_minus_one(self):
        """
        axis=-1 should map to normalized_shape=(last_dim,) and preserve eps/affine config.
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 4, 7),
            axis=-1,
            center=True,
            scale=True,
            eps=1e-5,
        )

        ctx = _Ctx(device=Device("cpu"))
        conv = LayerNormalizationConverter()
        kd = conv.build(k_layer, ctx)

        self.assertEqual(type(kd).__name__, "LayerNorm")
        self.assertEqual(tuple(getattr(kd, "normalized_shape")), (7,))
        self.assertAlmostEqual(float(getattr(kd, "eps")), 1e-5)
        self.assertTrue(bool(getattr(kd, "affine")))
        self.assertEqual(str(getattr(kd, "device")), "cpu")

        self.assertIsNotNone(getattr(kd, "gamma", None))
        self.assertIsNotNone(getattr(kd, "beta", None))
        self.assertEqual(tuple(getattr(kd, "gamma").shape), (7,))
        self.assertEqual(tuple(getattr(kd, "beta").shape), (7,))

    def test_build_supports_axis_list_trailing(self):
        """
        axis=[-2,-1] should map to normalized_shape=(d_{-2}, d_{-1}).
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 3, 4, 5),
            axis=[-2, -1],
            center=True,
            scale=True,
            eps=1e-3,
        )

        ctx = _Ctx(device=Device("cpu"))
        kd = LayerNormalizationConverter().build(k_layer, ctx)

        self.assertEqual(tuple(getattr(kd, "normalized_shape")), (4, 5))
        self.assertTrue(bool(getattr(kd, "affine")))

    def test_build_affine_false_when_center_or_scale_disabled(self):
        """
        When Keras center or scale is disabled, KeyDNN LayerNorm should be affine=False.
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 6),
            axis=-1,
            center=False,
            scale=True,
            eps=1e-4,
        )

        ctx = _Ctx(device=Device("cpu"))
        kd = LayerNormalizationConverter().build(k_layer, ctx)
        self.assertFalse(bool(getattr(kd, "affine")))
        self.assertIsNone(getattr(kd, "gamma", None))
        self.assertIsNone(getattr(kd, "beta", None))

    def test_load_weights_copies_gamma_beta_when_affine_true(self):
        """
        load_weights should copy Keras gamma/beta into KeyDNN LayerNorm when affine=True.
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 4, 7),
            axis=-1,
            center=True,
            scale=True,
            eps=1e-3,
        )

        gamma = np.linspace(1.0, 2.0, 7, dtype=np.float32)
        beta = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
        k_layer.set_weights([gamma, beta])

        ctx = _Ctx(device=Device("cpu"))
        conv = LayerNormalizationConverter()
        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)

        g = _param_to_numpy(getattr(kd, "gamma", None))
        b = _param_to_numpy(getattr(kd, "beta", None))

        if g is not None:
            np.testing.assert_allclose(g, gamma, rtol=0, atol=0)
        if b is not None:
            np.testing.assert_allclose(b, beta, rtol=0, atol=0)

    def test_axis_non_trailing_is_rejected(self):
        """
        Non-trailing axis specifications should be rejected in Phase 1.

        Notes
        -----
        Some Keras versions do not reliably expose `input_shape` on LayerNormalization
        after building. A minimal fake layer object is used to exercise converter
        validation deterministically.
        """

        class _FakeLN:
            # For input rank 3 (None, 3, 4), axis=1 normalizes a non-trailing dim.
            axis = 1
            center = True
            scale = True
            epsilon = 1e-3
            input_shape = (None, 3, 4)

            def get_weights(self):
                # Provide gamma/beta so normalized_shape inference succeeds.
                gamma = np.ones((4,), dtype=np.float32)
                beta = np.zeros((4,), dtype=np.float32)
                return [gamma, beta]

        ctx = _Ctx(device=Device("cpu"))
        with self.assertRaises(KerasInteropError):
            _ = LayerNormalizationConverter().build(_FakeLN(), ctx)

    def test_axis_non_trailing_list_is_rejected(self):
        """
        axis lists that do not match [-K,...,-1] should be rejected.
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 3, 4, 5),
            axis=[-3, -1],  # skips -2, not contiguous trailing
            center=True,
            scale=True,
        )

        ctx = _Ctx(device=Device("cpu"))
        with self.assertRaises(KerasInteropError):
            _ = LayerNormalizationConverter().build(k_layer, ctx)

    def test_duplicate_axis_is_rejected(self):
        """
        axis with duplicate entries should be rejected.

        Notes
        -----
        Keras itself rejects duplicate axes during execution, so a minimal fake
        layer object is used to exercise converter-side validation.
        """

        class _FakeLN:
            axis = (-1, -1)
            center = True
            scale = True
            epsilon = 1e-3
            input_shape = (None, 4, 7)

            def get_weights(self):
                # Provide gamma/beta to allow shape inference from weights.
                gamma = np.ones((7,), dtype=np.float32)
                beta = np.zeros((7,), dtype=np.float32)
                return [gamma, beta]

        ctx = _Ctx(device=Device("cpu"))
        with self.assertRaises(KerasInteropError):
            _ = LayerNormalizationConverter().build(_FakeLN(), ctx)

    def test_non_cpu_device_is_rejected(self):
        """
        Non-CPU target devices should be rejected (KeyDNN LayerNorm is CPU-only).
        """
        k_layer = self._build_ln_layer(
            input_shape=(2, 4, 7),
            axis=-1,
            center=True,
            scale=True,
        )

        ctx = _Ctx(device=Device("cuda:0"))
        with self.assertRaises(KerasInteropError):
            _ = LayerNormalizationConverter().build(k_layer, ctx)

    def test_incompatible_affine_config_rejected_on_load(self):
        """
        If KeyDNN LayerNorm is affine=False but Keras provides gamma/beta, load_weights should raise.
        """
        k_layer_affine = self._build_ln_layer(
            input_shape=(2, 7),
            axis=-1,
            center=True,
            scale=True,
        )

        gamma = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        k_layer_affine.set_weights([gamma, beta])

        # Build a non-affine KeyDNN LN by disabling center or scale in Keras.
        k_layer_no_beta = self._build_ln_layer(
            input_shape=(2, 7),
            axis=-1,
            center=False,
            scale=True,
        )

        ctx = _Ctx(device=Device("cpu"))
        conv = LayerNormalizationConverter()
        kd = conv.build(k_layer_no_beta, ctx)

        with self.assertRaises(KerasInteropError):
            conv.load_weights(kd, k_layer_affine, ctx)


if __name__ == "__main__":
    unittest.main()
