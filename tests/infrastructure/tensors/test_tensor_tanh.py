import unittest
import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.domain.device._device import Device


class TestTensorTanh(unittest.TestCase):
    def setUp(self):
        self.device = Device("cpu")

    def test_tanh_forward_matches_numpy(self):
        """
        Forward pass: tanh(x) should numerically match numpy.tanh(x).
        """
        x_np = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        x = Tensor(shape=x_np.shape, device=self.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y = x.tanh()
        y_np = y.to_numpy()

        np.testing.assert_allclose(
            y_np,
            np.tanh(x_np),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_tanh_forward_zero_is_zero(self):
        """
        tanh(0) == 0 exactly.
        """
        x = Tensor(shape=(3, 4), device=self.device, requires_grad=False)
        x.copy_from_numpy(np.zeros((3, 4), dtype=np.float32))

        y = x.tanh()
        y_np = y.to_numpy()

        self.assertTrue(np.all(y_np == 0.0))

    def test_tanh_is_odd_function(self):
        """
        tanh(-x) == -tanh(x)
        """
        x_np = np.random.randn(5, 6).astype(np.float32)
        x = Tensor(shape=x_np.shape, device=self.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y1 = x.tanh().to_numpy()

        x_neg = Tensor(shape=x_np.shape, device=self.device, requires_grad=False)
        x_neg.copy_from_numpy(-x_np)
        y2 = x_neg.tanh().to_numpy()

        np.testing.assert_allclose(y1, -y2, rtol=1e-6, atol=1e-6)

    def test_tanh_backward_matches_analytic_gradient(self):
        """
        Backward pass:
            d/dx tanh(x) = 1 - tanh(x)^2
        """
        x_np = np.random.randn(4, 3).astype(np.float32)
        x = Tensor(shape=x_np.shape, device=self.device, requires_grad=True)
        x.copy_from_numpy(x_np)

        y = x.tanh()
        loss = y.sum()  # simple scalar loss
        loss.backward()

        grad_np = x.grad.to_numpy()
        expected = 1.0 - np.tanh(x_np) ** 2

        np.testing.assert_allclose(
            grad_np,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_tanh_backward_zero_input(self):
        """
        At x = 0, derivative of tanh is exactly 1.
        """
        x_np = np.zeros((2, 2), dtype=np.float32)
        x = Tensor(shape=x_np.shape, device=self.device, requires_grad=True)
        x.copy_from_numpy(x_np)

        y = x.tanh()
        y.sum().backward()

        grad_np = x.grad.to_numpy()
        np.testing.assert_allclose(
            grad_np,
            np.ones_like(x_np),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
