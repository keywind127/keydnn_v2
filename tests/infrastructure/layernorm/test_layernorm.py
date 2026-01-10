import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.layers._layernorm import LayerNorm


def _tensor_supports_numpy_load() -> bool:
    """Return True if we can create a Tensor with known numeric contents using ONLY public APIs."""
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        pass

    t = Tensor((1, 1), Device("cpu"))
    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        return True
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        return True

    return False


def _make_tensor_from_numpy(
    arr: np.ndarray, device: Device, *, requires_grad: bool
) -> Tensor:
    """
    Construct a Tensor holding arr using public APIs only.
    Prefers data=..., otherwise uses from_numpy/copy_from_numpy.
    """
    arr = np.asarray(arr, dtype=np.float32)

    try:
        t = Tensor(data=arr, device=device, requires_grad=requires_grad)
        return t
    except TypeError:
        t = Tensor(arr.shape, device, requires_grad=requires_grad, ctx=None)
        if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
            t.from_numpy(arr)
            return t
        if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
            t.copy_from_numpy(arr)
            return t

        raise AssertionError(
            "Tensor cannot be loaded with NumPy data via public APIs. "
            "Implement Tensor(data=...) OR Tensor.from_numpy()/copy_from_numpy()."
        )


def _make_layernorm(
    *,
    normalized_shape: tuple[int, ...],
    eps: float,
    affine: bool,
    device: Device,
) -> LayerNorm:
    """
    Construct LayerNorm, supporting both signatures:
      - LayerNorm(..., device=Device("cpu"))  (recommended)
      - LayerNorm(...)                       (if it internally uses CPU)
    """
    try:
        return LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            affine=affine,
            device=device,
        )
    except TypeError:
        return LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            affine=affine,
        )


class TestLayerNormInfrastructure(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_get_config_and_from_config_roundtrip(self):
        ln1 = _make_layernorm(
            normalized_shape=(8,),
            eps=1e-5,
            affine=True,
            device=self.device,
        )
        cfg = ln1.get_config()

        self.assertEqual(cfg["normalized_shape"], [8])
        self.assertAlmostEqual(cfg["eps"], 1e-5)
        self.assertEqual(cfg["affine"], True)
        self.assertIn("device", cfg)

        ln2 = LayerNorm.from_config(cfg)
        self.assertIsInstance(ln2, LayerNorm)
        self.assertEqual(ln2.get_config(), cfg)

    def test_parameters_exist_only_when_affine_true(self):
        ln_aff = _make_layernorm(
            normalized_shape=(4,),
            eps=1e-5,
            affine=True,
            device=self.device,
        )
        params = list(ln_aff.parameters())
        self.assertEqual(len(params), 2)  # gamma + beta

        ln_no = _make_layernorm(
            normalized_shape=(4,),
            eps=1e-5,
            affine=False,
            device=self.device,
        )
        params2 = list(ln_no.parameters())
        self.assertEqual(len(params2), 0)

    def test_rejects_empty_normalized_shape(self):
        """
        LayerNorm(normalized_shape=()) is invalid. Current implementation raises
        at forward-time (not construction-time).
        """
        ln = _make_layernorm(
            normalized_shape=(),
            eps=1e-5,
            affine=True,
            device=self.device,
        )

        x = Tensor((2, 3), self.device)
        with self.assertRaises(ValueError):
            _ = ln.forward(x)


class TestLayerNormForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_rejects_rank_too_small(self):
        ln = _make_layernorm(
            normalized_shape=(3, 4),
            eps=1e-5,
            affine=True,
            device=self.device,
        )

        x = Tensor((3, 4), self.device)  # rank=2, needs rank>=2 for (3,4) OK
        # This one should pass rank check but fail trailing shape if mismatch occurs in other cases.
        # Let's make a rank-1 tensor which must fail.
        x1 = Tensor((12,), self.device)
        with self.assertRaises(ValueError):
            ln.forward(x1)

    def test_forward_rejects_trailing_shape_mismatch(self):
        ln = _make_layernorm(
            normalized_shape=(3,),
            eps=1e-5,
            affine=True,
            device=self.device,
        )

        x = Tensor((4, 5), self.device)  # trailing dim is 5, expected 3
        with self.assertRaises(ValueError):
            ln.forward(x)

    def test_forward_outputs_zero_mean_unit_var_affine_false_2d(self):
        """
        For affine=False, LayerNorm should normalize per-sample over last dims:
          mean ~ 0 and var ~ 1 for each sample.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(10)
        N, D = 32, 16
        x_np = (np.random.randn(N, D).astype(np.float32) * 4.0) + 2.0
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=False)

        ln = _make_layernorm(
            normalized_shape=(D,),
            eps=1e-5,
            affine=False,
            device=self.device,
        )

        y = ln.forward(x).to_numpy()
        mean = y.mean(axis=1)
        var = y.var(axis=1)

        np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=2e-3)
        np.testing.assert_allclose(var, np.ones_like(var), rtol=0, atol=4e-3)

    def test_forward_outputs_zero_mean_unit_var_affine_false_3d(self):
        """
        LayerNorm over last 2 dims for 3D input: (N, T, D), normalized_shape=(T,D)
        should yield per-sample mean/var over axes (1,2).
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(11)
        N, T, D = 8, 5, 7
        x_np = (np.random.randn(N, T, D).astype(np.float32) * 1.5) - 3.0
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=False)

        ln = _make_layernorm(
            normalized_shape=(T, D),
            eps=1e-5,
            affine=False,
            device=self.device,
        )

        y = ln.forward(x).to_numpy()
        mean = y.mean(axis=(1, 2))
        var = y.var(axis=(1, 2))

        np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=2e-3)
        np.testing.assert_allclose(var, np.ones_like(var), rtol=0, atol=5e-3)

    def test_affine_changes_output(self):
        """
        With affine=True (default gamma=1, beta=0) it should match affine=False.
        If we modify gamma/beta, output should change.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(12)
        N, D = 6, 9
        x_np = np.random.randn(N, D).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=False)

        ln_no = _make_layernorm(
            normalized_shape=(D,),
            eps=1e-5,
            affine=False,
            device=self.device,
        )
        y_no = ln_no.forward(x).to_numpy()

        ln_aff = _make_layernorm(
            normalized_shape=(D,),
            eps=1e-5,
            affine=True,
            device=self.device,
        )
        y0 = ln_aff.forward(x).to_numpy()
        np.testing.assert_allclose(y0, y_no, rtol=0, atol=1e-6)

        # tweak gamma/beta => output should differ
        assert ln_aff.gamma is not None and ln_aff.beta is not None

        gamma_np = np.ones((D,), dtype=np.float32) * 2.0
        beta_np = np.ones((D,), dtype=np.float32) * 0.5

        ln_aff.gamma.copy_from(
            _make_tensor_from_numpy(gamma_np, self.device, requires_grad=False)
        )
        ln_aff.beta.copy_from(
            _make_tensor_from_numpy(beta_np, self.device, requires_grad=False)
        )

        y1 = ln_aff.forward(x).to_numpy()
        self.assertFalse(np.allclose(y1, y0))


class TestLayerNormBackward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_backward_populates_grad_for_x_and_affine_params(self):
        """
        x.requires_grad True and affine=True should yield grads for:
        - x
        - gamma
        - beta
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(20)
        N, D = 7, 13
        x_np = np.random.randn(N, D).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=True)

        ln = _make_layernorm(
            normalized_shape=(D,),
            eps=1e-5,
            affine=True,
            device=self.device,
        )

        for p in ln.parameters():
            p.requires_grad = True

        out = ln.forward(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        self.assertTrue(hasattr(ln, "gamma"))
        self.assertTrue(hasattr(ln, "beta"))
        assert ln.gamma is not None and ln.beta is not None

        self.assertIsNotNone(ln.gamma.grad)
        self.assertIsNotNone(ln.beta.grad)

        self.assertEqual(ln.gamma.grad.shape, ln.gamma.shape)
        self.assertEqual(ln.beta.grad.shape, ln.beta.shape)

    def test_backward_runs_when_affine_false(self):
        """
        affine=False should still backprop into x.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(21)
        N, T, D = 4, 6, 5
        x = _make_tensor_from_numpy(
            np.random.randn(N, T, D).astype(np.float32),
            self.device,
            requires_grad=True,
        )

        ln = _make_layernorm(
            normalized_shape=(T, D),
            eps=1e-5,
            affine=False,
            device=self.device,
        )

        y = ln.forward(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_gradcheck_like_finite_difference_small(self):
        """
        Lightweight finite-difference check on x gradient for sanity.
        Uses a tiny tensor to keep runtime small.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric LayerNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(22)
        N, D = 2, 4
        eps_fd = 1e-3

        x0 = np.random.randn(N, D).astype(np.float32)

        ln = _make_layernorm(
            normalized_shape=(D,),
            eps=1e-5,
            affine=False,
            device=self.device,
        )

        # analytic grad
        x = _make_tensor_from_numpy(x0, self.device, requires_grad=True)
        y = ln.forward(x)
        # scalar loss: sum(y)
        y.sum().backward()
        g_analytic = x.grad.to_numpy()

        # numeric grad (central difference)
        g_num = np.zeros_like(x0, dtype=np.float32)
        for i in range(N):
            for j in range(D):
                xp = x0.copy()
                xm = x0.copy()
                xp[i, j] += eps_fd
                xm[i, j] -= eps_fd

                tp = _make_tensor_from_numpy(xp, self.device, requires_grad=False)
                tm = _make_tensor_from_numpy(xm, self.device, requires_grad=False)

                lp = ln.forward(tp).to_numpy().sum()
                lm = ln.forward(tm).to_numpy().sum()
                g_num[i, j] = (lp - lm) / (2.0 * eps_fd)

        np.testing.assert_allclose(g_analytic, g_num, rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    unittest.main()
