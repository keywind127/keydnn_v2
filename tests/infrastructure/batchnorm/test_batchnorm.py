import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor

from src.keydnn.infrastructure.layers._batchnorm import BatchNorm1d


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


def _make_bn1d(
    *, num_features: int, eps: float, momentum: float, affine: bool, device: Device
) -> BatchNorm1d:
    """
    Construct BatchNorm1d, supporting both signatures:
      - BatchNorm1d(..., device=Device("cpu"))  (if you implemented it)
      - BatchNorm1d(...)                       (if it internally uses CPU)
    """
    try:
        return BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            device=device,
        )
    except TypeError:
        return BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )


class TestBatchNorm1dInfrastructure(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_get_config_and_from_config_roundtrip(self):
        bn1 = _make_bn1d(
            num_features=8, eps=1e-5, momentum=0.2, affine=True, device=self.device
        )
        cfg = bn1.get_config()

        self.assertEqual(cfg["num_features"], 8)
        self.assertAlmostEqual(cfg["eps"], 1e-5)
        self.assertAlmostEqual(cfg["momentum"], 0.2)
        self.assertEqual(cfg["affine"], True)

        bn2 = BatchNorm1d.from_config(cfg)
        self.assertIsInstance(bn2, BatchNorm1d)
        self.assertEqual(bn2.get_config(), cfg)

    def test_parameters_exist_only_when_affine_true(self):
        bn_aff = _make_bn1d(
            num_features=4, eps=1e-5, momentum=0.1, affine=True, device=self.device
        )
        params = list(bn_aff.parameters())
        self.assertEqual(len(params), 2)  # gamma + beta

        bn_no = _make_bn1d(
            num_features=4, eps=1e-5, momentum=0.1, affine=False, device=self.device
        )
        params2 = list(bn_no.parameters())
        self.assertEqual(len(params2), 0)


class TestBatchNorm1dForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_rejects_non_2d_input(self):
        bn = _make_bn1d(
            num_features=3, eps=1e-5, momentum=0.1, affine=True, device=self.device
        )

        x1 = Tensor((3,), self.device)  # 1D
        with self.assertRaises(ValueError):
            bn.forward(x1)

        x3 = Tensor((2, 3, 4), self.device)  # 3D
        with self.assertRaises(ValueError):
            bn.forward(x3)

    def test_forward_rejects_feature_mismatch(self):
        bn = _make_bn1d(
            num_features=3, eps=1e-5, momentum=0.1, affine=True, device=self.device
        )
        x = Tensor((4, 5), self.device)  # C=5 mismatch
        with self.assertRaises(ValueError):
            bn.forward(x)

    def test_training_forward_outputs_zero_mean_unit_var_affine_false(self):
        """
        In training mode, for affine=False, output should be normalized:
          mean ~ 0 and var ~ 1 across batch dimension (per feature).
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric BatchNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(0)
        N, C = 64, 6
        x_np = (
            np.random.randn(N, C).astype(np.float32) * 3.0
        ) + 5.0  # non-zero mean, scaled var
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=False)

        bn = _make_bn1d(
            num_features=C, eps=1e-5, momentum=0.1, affine=False, device=self.device
        )
        bn.training = True

        y = bn.forward(x).to_numpy()

        mean = y.mean(axis=0)
        var = y.var(axis=0)

        # Allow small numeric error due to eps and float32
        np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=1e-3)
        np.testing.assert_allclose(var, np.ones_like(var), rtol=0, atol=2e-3)

    def test_running_stats_update_in_training(self):
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric BatchNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(1)
        N, C = 32, 4
        x_np = np.random.randn(N, C).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=False)

        bn = _make_bn1d(
            num_features=C, eps=1e-5, momentum=0.5, affine=True, device=self.device
        )
        bn.training = True

        # Snapshot running stats before
        rm0 = bn.running_mean.to_numpy().copy()
        rv0 = bn.running_var.to_numpy().copy()

        _ = bn.forward(x)

        rm1 = bn.running_mean.to_numpy()
        rv1 = bn.running_var.to_numpy()

        # running stats should change (very likely)
        self.assertFalse(np.allclose(rm0, rm1))
        self.assertFalse(np.allclose(rv0, rv1))
        self.assertEqual(rm1.shape, (C,))
        self.assertEqual(rv1.shape, (C,))

    def test_eval_uses_running_stats_deterministically(self):
        """
        After a training forward updates running stats, eval forward should be deterministic
        and should not depend on current batch stats.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric BatchNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(2)
        N, C = 16, 5
        bn = _make_bn1d(
            num_features=C, eps=1e-5, momentum=0.9, affine=False, device=self.device
        )

        # Training pass to update running stats
        bn.training = True
        x_train = _make_tensor_from_numpy(
            np.random.randn(N, C).astype(np.float32), self.device, requires_grad=False
        )
        _ = bn.forward(x_train)

        # Eval: same input twice => identical output
        bn.training = False
        x_eval = _make_tensor_from_numpy(
            np.random.randn(N, C).astype(np.float32), self.device, requires_grad=False
        )

        y1 = bn.forward(x_eval).to_numpy()
        y2 = bn.forward(x_eval).to_numpy()
        np.testing.assert_allclose(y1, y2, rtol=0, atol=0)


class TestBatchNorm1dBackward(TestCase):
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
            "Cannot run numeric BatchNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(3)
        N, C = 8, 7
        x_np = np.random.randn(N, C).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device, requires_grad=True)

        bn = _make_bn1d(
            num_features=C, eps=1e-5, momentum=0.1, affine=True, device=self.device
        )
        bn.training = True

        # Be explicit: params require grad
        for p in bn.parameters():
            p.requires_grad = True

        out = bn.forward(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # gamma/beta grads
        # (Parameters are Tensor-like in your framework)
        self.assertTrue(hasattr(bn, "gamma"))
        self.assertTrue(hasattr(bn, "beta"))

        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

        self.assertEqual(bn.gamma.grad.shape, bn.gamma.shape)
        self.assertEqual(bn.beta.grad.shape, bn.beta.shape)

    def test_backward_runs_when_affine_false(self):
        """
        affine=False should still backprop into x.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric BatchNorm tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(4)
        N, C = 10, 3
        x = _make_tensor_from_numpy(
            np.random.randn(N, C).astype(np.float32), self.device, requires_grad=True
        )

        bn = _make_bn1d(
            num_features=C, eps=1e-5, momentum=0.1, affine=False, device=self.device
        )
        bn.training = True

        y = bn.forward(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
