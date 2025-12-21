import unittest
import warnings
from unittest.mock import patch

import numpy as np

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


def _ref_maxpool2d_forward_numpy(
    x: np.ndarray,
    *,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference maxpool2d forward that matches KeyDNN semantics:
    - NCHW
    - padding with -inf
    - argmax_idx stores flattened spatial index into padded plane: h * W_pad + w
    """

    def _pair(v):
        if isinstance(v, tuple):
            return v
        return (v, v)

    def _out_hw(H, W, k, s, p):
        k_h, k_w = k
        s_h, s_w = s
        p_h, p_w = p
        H_out = (H + 2 * p_h - k_h) // s_h + 1
        W_out = (W + 2 * p_w - k_w) // s_w + 1
        return H_out, W_out

    k = _pair(kernel_size)
    s = _pair(kernel_size if stride is None else stride)
    p = _pair(padding)

    N, C, H, W = x.shape
    k_h, k_w = k
    p_h, p_w = p
    s_h, s_w = s

    H_out, W_out = _out_hw(H, W, k, s, p)

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    y = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    argmax_idx = np.empty((N, C, H_out, W_out), dtype=np.int64)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    flat_idx = int(np.argmax(patch))
                    y[n, c, i, j] = patch.reshape(-1)[flat_idx]

                    ph = flat_idx // k_w
                    pw = flat_idx % k_w
                    h = h0 + ph
                    w_ = w0 + pw
                    argmax_idx[n, c, i, j] = h * W_pad + w_

    return y, argmax_idx


class TestPool2dOps(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    # -----------------------------
    # Existing deterministic tests
    # -----------------------------
    def test_maxpool2d_forward_known_values(self):
        # Input: 1x1x2x2
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        expected = np.array([[[[4.0]]]], dtype=np.float32)

        self.assertEqual(y.shape, (1, 1, 1, 1))
        self.assertTrue(np.allclose(y, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(argmax_idx.shape, y.shape)
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(argmax_idx.dtype, np.int64)

    def test_maxpool2d_backward_routes_grad_to_argmax(self):
        # Same input as above, max is at bottom-right (value=4)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        _, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
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

    # -----------------------------
    # New: float64 coverage
    # -----------------------------
    def test_maxpool2d_forward_float64_matches_reference(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float64)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        y_ref, idx_ref = _ref_maxpool2d_forward_numpy(
            x, kernel_size=2, stride=2, padding=0
        )

        self.assertEqual(y.dtype, np.float64)
        self.assertEqual(argmax_idx.dtype, np.int64)
        np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)
        np.testing.assert_array_equal(argmax_idx, idx_ref)

    def test_maxpool2d_backward_float64_routes_grad(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float64)
        _, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)

        grad_out = np.array([[[[5.0]]]], dtype=np.float64)
        grad_x = maxpool2d_backward_cpu(
            grad_out, argmax_idx, x_shape=x.shape, kernel_size=2, stride=2, padding=0
        )

        expected_grad_x = np.array([[[[0.0, 0.0], [0.0, 5.0]]]], dtype=np.float64)
        self.assertEqual(grad_x.dtype, np.float64)
        np.testing.assert_allclose(grad_x, expected_grad_x, rtol=0, atol=0)

    # -----------------------------
    # New: randomized stress tests for float32/float64
    # -----------------------------
    def test_maxpool2d_forward_randomized_float32_float64(self):
        rng = np.random.default_rng(123)

        for dtype in (np.float32, np.float64):
            for _ in range(20):
                N = int(rng.integers(1, 4))
                C = int(rng.integers(1, 4))
                H = int(rng.integers(3, 10))
                W = int(rng.integers(3, 10))

                k_h = int(rng.integers(1, min(5, H) + 1))
                k_w = int(rng.integers(1, min(5, W) + 1))
                s_h = int(rng.integers(1, 4))
                s_w = int(rng.integers(1, 4))
                p_h = int(rng.integers(0, 3))
                p_w = int(rng.integers(0, 3))

                x = rng.standard_normal((N, C, H, W)).astype(dtype, copy=False)

                y, idx = maxpool2d_forward_cpu(
                    x,
                    kernel_size=(k_h, k_w),
                    stride=(s_h, s_w),
                    padding=(p_h, p_w),
                )
                y_ref, idx_ref = _ref_maxpool2d_forward_numpy(
                    x,
                    kernel_size=(k_h, k_w),
                    stride=(s_h, s_w),
                    padding=(p_h, p_w),
                )

                self.assertEqual(y.dtype, dtype)
                self.assertEqual(idx.dtype, np.int64)
                np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)
                np.testing.assert_array_equal(idx, idx_ref)

    # -----------------------------
    # New: dtype fallback behavior (unsupported dtypes)
    # -----------------------------
    def test_maxpool2d_forward_dtype_handling(self):
        # Non-float dtypes should be rejected (padding uses -inf).
        x_i32 = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
        with self.assertRaises(TypeError):
            _ = maxpool2d_forward_cpu(x_i32, kernel_size=2, stride=2, padding=0)

        # float16 is allowed (falls back to NumPy path if native doesn't support it).
        x_f16 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
        y16, idx16 = maxpool2d_forward_cpu(x_f16, kernel_size=2, stride=2, padding=0)

        self.assertEqual(y16.dtype, np.float16)
        np.testing.assert_allclose(
            y16.astype(np.float32),
            np.array([[[[4.0]]]], dtype=np.float32),
            rtol=0,
            atol=0,
        )
        self.assertEqual(idx16.dtype, np.int64)

    # -----------------------------
    # New: verify warning on native load failure (simulated)
    # -----------------------------
    def test_maxpool2d_forward_warns_and_falls_back_when_native_missing(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

        # Force load_keydnn_native() to raise OSError so we can assert warning behavior.
        # Patch the function where maxpool2d_forward_cpu imports it from.
        with patch(
            "src.keydnn.infrastructure.native.python.maxpool2d_ctypes.load_keydnn_native",
            side_effect=OSError("simulated missing dll"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                y, idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)

                self.assertEqual(y.dtype, np.float32)
                np.testing.assert_allclose(
                    y, np.array([[[[4.0]]]], dtype=np.float32), rtol=0, atol=0
                )
                self.assertEqual(idx.dtype, np.int64)

                self.assertTrue(
                    any(issubclass(wi.category, RuntimeWarning) for wi in w)
                )
                self.assertTrue(
                    any("falling back to NumPy" in str(wi.message) for wi in w)
                )


if __name__ == "__main__":
    unittest.main()
