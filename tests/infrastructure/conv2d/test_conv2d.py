import unittest
import numpy as np

from keydnn.infrastructure.ops.conv2d_cpu import conv2d_forward_cpu, conv2d_backward_cpu


def finite_diff_grad_x(x, w, b, stride, padding, eps=1e-3):
    """
    Finite difference gradient for x under loss L = sum(conv2d(x,w,b)).
    """
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x[idx])

        x[idx] = old + eps
        y1 = float(conv2d_forward_cpu(x, w, b, stride=stride, padding=padding).sum())

        x[idx] = old - eps
        y2 = float(conv2d_forward_cpu(x, w, b, stride=stride, padding=padding).sum())

        x[idx] = old
        grad[idx] = (y1 - y2) / (2.0 * eps)
        it.iternext()
    return grad


def finite_diff_grad_w(x, w, b, stride, padding, eps=1e-3):
    """
    Finite difference gradient for w under loss L = sum(conv2d(x,w,b)).
    """
    grad = np.zeros_like(w, dtype=np.float32)
    it = np.nditer(w, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(w[idx])

        w[idx] = old + eps
        y1 = float(conv2d_forward_cpu(x, w, b, stride=stride, padding=padding).sum())

        w[idx] = old - eps
        y2 = float(conv2d_forward_cpu(x, w, b, stride=stride, padding=padding).sum())

        w[idx] = old
        grad[idx] = (y1 - y2) / (2.0 * eps)
        it.iternext()
    return grad


class TestConv2dCPU(unittest.TestCase):
    def test_forward_shape(self):
        x = np.zeros((2, 3, 10, 9), dtype=np.float32)
        w = np.zeros((4, 3, 3, 3), dtype=np.float32)
        b = np.zeros((4,), dtype=np.float32)

        y = conv2d_forward_cpu(x, w, b, stride=2, padding=1)
        # H_out = floor((10+2-3)/2)+1 = 5
        # W_out = floor((9+2-3)/2)+1  = 5
        self.assertEqual(y.shape, (2, 4, 5, 5))

    def test_forward_known_values_all_ones(self):
        # x = ones, w = ones, stride=1, padding=0 => each 2x2 patch sums to 4
        x = np.ones((1, 1, 3, 3), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)

        y = conv2d_forward_cpu(x, w, b=None, stride=1, padding=0)
        expected = np.array([[[[4.0, 4.0], [4.0, 4.0]]]], dtype=np.float32)

        self.assertTrue(
            np.allclose(y, expected), msg=f"\nGot:\n{y}\nExpected:\n{expected}"
        )

    def test_backward_matches_finite_difference(self):
        # Validate grad_x, grad_w, grad_b using finite differences on a tiny case.
        np.random.seed(0)
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        w = np.random.randn(1, 1, 3, 3).astype(np.float32)
        b = np.random.randn(1).astype(np.float32)

        stride = 1
        padding = 1

        y = conv2d_forward_cpu(x, w, b, stride=stride, padding=padding)
        grad_out = np.ones_like(y, dtype=np.float32)  # L = sum(y)

        grad_x, grad_w, grad_b = conv2d_backward_cpu(
            x, w, b, grad_out, stride=stride, padding=padding
        )

        fd_x = finite_diff_grad_x(
            x.copy(), w.copy(), b.copy(), stride, padding, eps=1e-3
        )
        fd_w = finite_diff_grad_w(
            x.copy(), w.copy(), b.copy(), stride, padding, eps=1e-3
        )
        fd_b = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)

        self.assertTrue(
            np.allclose(grad_x, fd_x, atol=1e-2, rtol=1e-2),
            msg=f"grad_x mismatch\nmax_abs={np.max(np.abs(grad_x - fd_x))}",
        )
        self.assertTrue(
            np.allclose(grad_w, fd_w, atol=1e-2, rtol=1e-2),
            msg=f"grad_w mismatch\nmax_abs={np.max(np.abs(grad_w - fd_w))}",
        )
        self.assertIsNotNone(grad_b)
        self.assertTrue(
            np.allclose(grad_b, fd_b, atol=1e-6, rtol=1e-6),
            msg=f"grad_b mismatch\nGot={grad_b}\nExpected={fd_b}",
        )


if __name__ == "__main__":
    unittest.main()
