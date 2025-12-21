import unittest
from unittest import mock

import numpy as np

from src.keydnn.infrastructure.ops.conv2d_cpu import (
    conv2d_forward_cpu,
    conv2d_backward_cpu,
)


def finite_diff_grad_x(x, w, b, stride, padding, eps=1e-3):
    """
    Finite difference gradient for x under loss L = sum(conv2d(x,w,b)).
    """
    grad = np.zeros_like(x, dtype=x.dtype)
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
    grad = np.zeros_like(w, dtype=w.dtype)
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
    def test_forward_shape_float32(self):
        x = np.zeros((2, 3, 10, 9), dtype=np.float32)
        w = np.zeros((4, 3, 3, 3), dtype=np.float32)
        b = np.zeros((4,), dtype=np.float32)

        y = conv2d_forward_cpu(x, w, b, stride=2, padding=1)
        # H_out = floor((10+2-3)/2)+1 = 5
        # W_out = floor((9+2-3)/2)+1  = 5
        self.assertEqual(y.shape, (2, 4, 5, 5))
        self.assertEqual(y.dtype, np.float32)

    def test_forward_known_values_all_ones_float32(self):
        # x = ones, w = ones, stride=1, padding=0 => each 2x2 patch sums to 4
        x = np.ones((1, 1, 3, 3), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)

        y = conv2d_forward_cpu(x, w, b=None, stride=1, padding=0)
        expected = np.array([[[[4.0, 4.0], [4.0, 4.0]]]], dtype=np.float32)

        self.assertTrue(
            np.allclose(y, expected), msg=f"\nGot:\n{y}\nExpected:\n{expected}"
        )

    def test_backward_matches_finite_difference_float32(self):
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

    def test_forward_and_backward_matches_finite_difference_float64(self):
        # Same as float32 test, but in float64 to validate dtype paths.
        np.random.seed(1)
        x = np.random.randn(1, 1, 4, 4).astype(np.float64)
        w = np.random.randn(1, 1, 3, 3).astype(np.float64)
        b = np.random.randn(1).astype(np.float64)

        stride = 1
        padding = 1

        y = conv2d_forward_cpu(x, w, b, stride=stride, padding=padding)
        self.assertEqual(y.dtype, np.float64)

        grad_out = np.ones_like(y, dtype=np.float64)  # L = sum(y)

        grad_x, grad_w, grad_b = conv2d_backward_cpu(
            x, w, b, grad_out, stride=stride, padding=padding
        )

        # eps a bit smaller for float64
        fd_x = finite_diff_grad_x(
            x.copy(), w.copy(), b.copy(), stride, padding, eps=1e-5
        )
        fd_w = finite_diff_grad_w(
            x.copy(), w.copy(), b.copy(), stride, padding, eps=1e-5
        )
        fd_b = grad_out.sum(axis=(0, 2, 3)).astype(np.float64)

        self.assertTrue(
            np.allclose(grad_x, fd_x, atol=1e-6, rtol=1e-6),
            msg=f"[float64] grad_x mismatch\nmax_abs={np.max(np.abs(grad_x - fd_x))}",
        )
        self.assertTrue(
            np.allclose(grad_w, fd_w, atol=1e-6, rtol=1e-6),
            msg=f"[float64] grad_w mismatch\nmax_abs={np.max(np.abs(grad_w - fd_w))}",
        )
        self.assertIsNotNone(grad_b)
        self.assertTrue(
            np.allclose(grad_b, fd_b, atol=1e-10, rtol=1e-10),
            msg=f"[float64] grad_b mismatch\nGot={grad_b}\nExpected={fd_b}",
        )

    def test_fallback_preserves_dtype_float16(self):
        # float16 is unsupported by native path; must fall back to Python reference loop.
        x = (np.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3) + 1).astype(np.float16)
        w = np.ones((1, 1, 2, 2), dtype=np.float16)

        y = conv2d_forward_cpu(x, w, b=None, stride=1, padding=0)
        self.assertEqual(y.dtype, np.float16)

        # Backward: grad_out float16 too, and outputs should preserve dtype for grad_x / grad_w.
        grad_out = np.ones_like(y, dtype=np.float16)
        grad_x, grad_w, grad_b = conv2d_backward_cpu(
            x, w, None, grad_out, stride=1, padding=0
        )

        self.assertEqual(grad_x.dtype, np.float16)
        self.assertEqual(grad_w.dtype, np.float16)
        self.assertIsNone(grad_b)

    def test_forward_native_dispatch_calls_f32_wrapper_when_available(self):
        # Verify that for float32, we attempt the native path and call conv2d_forward_f32_ctypes.
        # This test does not require an actual shared library.
        x = np.zeros((1, 1, 3, 3), dtype=np.float32)
        w = np.zeros((1, 1, 2, 2), dtype=np.float32)
        b = None

        class _DummyLib:
            pass

        called = {"f32": 0}

        def _fake_load_keydnn_native():
            return _DummyLib()

        def _fake_conv2d_forward_f32_ctypes(
            lib,
            *,
            x_pad,
            w,
            b,
            y,
            **kwargs,
        ):
            called["f32"] += 1
            y[...] = 123.0

        with mock.patch(
            "keydnn.infrastructure.native.python.conv2d_ctypes.load_keydnn_native",
            side_effect=_fake_load_keydnn_native,
        ), mock.patch(
            "keydnn.infrastructure.native.python.conv2d_ctypes.conv2d_forward_f32_ctypes",
            side_effect=_fake_conv2d_forward_f32_ctypes,
        ):
            y = conv2d_forward_cpu(x, w, b, stride=1, padding=0)

        self.assertEqual(called["f32"], 1)
        self.assertTrue(np.all(y == np.float32(123.0)))

    def test_backward_native_dispatch_calls_f32_wrapper_when_available(self):
        # Verify that for float32, we attempt the native path and call conv2d_backward_f32_ctypes.
        x = np.zeros((1, 1, 3, 3), dtype=np.float32)
        w = np.zeros((1, 1, 2, 2), dtype=np.float32)
        b = np.zeros((1,), dtype=np.float32)

        y = conv2d_forward_cpu(x, w, b, stride=1, padding=0)
        grad_out = np.ones_like(y, dtype=np.float32)

        class _DummyLib:
            pass

        called = {"bwd_f32": 0}

        def _fake_load_keydnn_native():
            return _DummyLib()

        def _fake_conv2d_backward_f32_ctypes(
            lib,
            *,
            x_pad,
            w,
            grad_out,
            grad_x_pad,
            grad_w,
            **kwargs,
        ):
            called["bwd_f32"] += 1
            grad_x_pad[...] = 7.0
            grad_w[...] = 9.0

        with mock.patch(
            "keydnn.infrastructure.native.python.conv2d_ctypes.load_keydnn_native",
            side_effect=_fake_load_keydnn_native,
        ), mock.patch(
            "keydnn.infrastructure.native.python.conv2d_ctypes.conv2d_backward_f32_ctypes",
            side_effect=_fake_conv2d_backward_f32_ctypes,
        ):
            grad_x, grad_w_out, grad_b = conv2d_backward_cpu(
                x, w, b, grad_out, stride=1, padding=0
            )

        self.assertEqual(called["bwd_f32"], 1)
        self.assertTrue(np.all(grad_x == np.float32(7.0)))
        self.assertTrue(np.all(grad_w_out == np.float32(9.0)))
        self.assertIsNotNone(grad_b)


if __name__ == "__main__":
    unittest.main()
