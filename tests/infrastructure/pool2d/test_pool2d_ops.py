import unittest
import numpy as np

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


class TestPool2dOps(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_maxpool2d_forward_known_values(self):
        # Input: 1x1x2x2
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        expected = np.array([[[[4.0]]]], dtype=np.float32)

        self.assertEqual(y.shape, (1, 1, 1, 1))
        self.assertTrue(np.allclose(y, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(argmax_idx.shape, y.shape)

    def test_maxpool2d_backward_routes_grad_to_argmax(self):
        # Same input as above, max is at bottom-right (value=4)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        grad_out = np.array([[[[5.0]]]], dtype=np.float32)

        grad_x = maxpool2d_backward_cpu(
            grad_out, argmax_idx, x_shape=x.shape, kernel_size=2, stride=2, padding=0
        )

        expected_grad_x = np.array([[[[0.0, 0.0], [0.0, 5.0]]]], dtype=np.float32)

        self.assertEqual(grad_x.shape, x.shape)
        self.assertTrue(np.allclose(grad_x, expected_grad_x, atol=1e-6, rtol=1e-6))

    def test_avgpool2d_forward_known_values(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        y = avgpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        expected = np.array([[[[2.5]]]], dtype=np.float32)

        self.assertEqual(y.shape, (1, 1, 1, 1))
        self.assertTrue(np.allclose(y, expected, atol=1e-6, rtol=1e-6))

    def test_avgpool2d_backward_uniform_distribution(self):
        x_shape = (1, 1, 2, 2)
        grad_out = np.array([[[[8.0]]]], dtype=np.float32)

        grad_x = avgpool2d_backward_cpu(
            grad_out, x_shape=x_shape, kernel_size=2, stride=2, padding=0
        )

        # 8 distributed over 4 elements => 2 each
        expected = np.array([[[[2.0, 2.0], [2.0, 2.0]]]], dtype=np.float32)

        self.assertEqual(grad_x.shape, x_shape)
        self.assertTrue(np.allclose(grad_x, expected, atol=1e-6, rtol=1e-6))

    def test_global_avgpool2d_forward_backward(self):
        x = np.arange(1, 1 + 2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
        y = global_avgpool2d_forward_cpu(x)

        self.assertEqual(y.shape, (2, 3, 1, 1))

        grad_out = np.ones((2, 3, 1, 1), dtype=np.float32) * 10.0
        grad_x = global_avgpool2d_backward_cpu(grad_out, x_shape=x.shape)

        self.assertEqual(grad_x.shape, x.shape)
        # Each element gets 10/(H*W)
        self.assertTrue(np.allclose(grad_x, 10.0 / (4.0 * 5.0), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
