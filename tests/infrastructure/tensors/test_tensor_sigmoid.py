import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


class TestTensorSigmoid(unittest.TestCase):
    def test_forward_matches_numpy(self):
        device = Device("cpu")

        x_np = np.array([[-4.0, -1.0, 0.0, 1.0, 4.0]], dtype=np.float32)
        x = Tensor(shape=x_np.shape, device=device, requires_grad=False, ctx=None)
        x.copy_from_numpy(x_np)

        y = x.sigmoid().to_numpy()
        y_ref = 1.0 / (1.0 + np.exp(-x_np))

        np.testing.assert_allclose(y, y_ref, rtol=1e-6, atol=1e-6)

    def test_backward_matches_finite_difference(self):
        device = Device("cpu")
        rng = np.random.default_rng(0)

        x_np = rng.normal(size=(2, 3)).astype(np.float32)
        x = Tensor(shape=x_np.shape, device=device, requires_grad=True, ctx=None)
        x.copy_from_numpy(x_np)

        # loss = sum(sigmoid(x))
        y = x.sigmoid()
        loss = y.sum()
        loss.backward()

        grad = x.grad.to_numpy()
        self.assertEqual(grad.shape, x_np.shape)

        eps = 1e-3
        grad_fd = np.zeros_like(x_np, dtype=np.float32)

        def f(arr: np.ndarray) -> float:
            xx = Tensor(shape=arr.shape, device=device, requires_grad=False, ctx=None)
            xx.copy_from_numpy(arr.astype(np.float32))
            return float(xx.sigmoid().sum().to_numpy())

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                x_pos = x_np.copy()
                x_neg = x_np.copy()
                x_pos[i, j] += eps
                x_neg[i, j] -= eps
                grad_fd[i, j] = (f(x_pos) - f(x_neg)) / (2.0 * eps)

        # finite-diff is noisy in float32; keep tolerances reasonable
        np.testing.assert_allclose(grad, grad_fd, rtol=2e-2, atol=2e-2)
