from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _conv2d_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> np.ndarray:
    """
    Naive NCHW/OIHW forward reference (matches your CPU semantics).
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for co in range(C_out):
            bias = b[co] if b is not None else x.dtype.type(0)
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    acc = x.dtype.type(0)
                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                acc += (
                                    x_pad[n, ci, h0 + kh, w0 + kw] * w[co, ci, kh, kw]
                                )
                    y[n, co, i, j] = acc + bias
    return y


def _conv2d_backward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Naive backward reference returning (grad_x, grad_w, grad_b).
    grad_b matches your semantics: sum over (N,H_out,W_out) if b is not None.
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    N2, C_out2, H_out, W_out = grad_out.shape
    if N2 != N or C_out2 != C_out:
        raise ValueError("grad_out shape mismatch")

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    grad_x_pad = np.zeros((N, C_in, H_pad, W_pad), dtype=x.dtype)
    grad_w = np.zeros_like(w)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    go = grad_out[n, co, i, j]
                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                grad_w[co, ci, kh, kw] += (
                                    go * x_pad[n, ci, h0 + kh, w0 + kw]
                                )
                                grad_x_pad[n, ci, h0 + kh, w0 + kw] += (
                                    go * w[co, ci, kh, kw]
                                )

    grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    return grad_x, grad_w, grad_b


class TestConv2dCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import conv2d_cuda as ops_conv

        cls.ops_conv = ops_conv

        # Forward candidates
        cls.conv2d_fwd = resolve_func(
            ops_conv,
            candidates=[
                "conv2d_forward_cuda",
                "conv2d_cuda",
                "conv2d_forward",
            ],
        )
        # Backward candidates
        cls.conv2d_bwd = resolve_func(
            ops_conv,
            candidates=[
                "conv2d_backward_cuda",
                "conv2d_backward",
            ],
        )

    def _run_forward(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        # Keep small; this wrapper does CPU padding + GPU compute
        N, C_in, C_out = 2, 3, 4
        H, W = 5, 6
        K_h, K_w = 3, 2
        stride = (1, 2)
        padding = (1, 0)

        x = np.random.randn(N, C_in, H, W).astype(dtype)
        w = np.random.randn(C_out, C_in, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)

        y = self.conv2d_fwd(
            self.env.lib,
            x=x,
            w=w,
            b=b,
            stride=stride,
            padding=padding,
            dtype=dtype,
            sync=True,
            device_index=0,
        )

        self.assertEqual(y.shape, y_ref.shape)
        assert_allclose_by_dtype(y, y_ref, dtype, op="conv2d_forward")

    def test_forward_float32_no_bias(self) -> None:
        self._run_forward(np.float32, with_bias=False)

    def test_forward_float32_with_bias(self) -> None:
        self._run_forward(np.float32, with_bias=True)

    def test_forward_float64_no_bias(self) -> None:
        self._run_forward(np.float64, with_bias=False)

    def test_forward_float64_with_bias(self) -> None:
        self._run_forward(np.float64, with_bias=True)

    def _run_backward(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        # Small sizes; backward uses atomics in native kernel -> tolerate numeric noise
        N, C_in, C_out = 2, 2, 3
        H, W = 6, 5
        K_h, K_w = 3, 3
        stride = (1, 1)
        padding = (1, 1)

        x = np.random.randn(N, C_in, H, W).astype(dtype)
        w = np.random.randn(C_out, C_in, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        # Compute output shape then grad_out
        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        grad_x_ref, grad_w_ref, grad_b_ref = _conv2d_backward_ref(
            x, w, b, grad_out, stride=stride, padding=padding
        )

        grad_x, grad_w, grad_b = self.conv2d_bwd(
            self.env.lib,
            x=x,
            w=w,
            b=b,
            grad_out=grad_out,
            stride=stride,
            padding=padding,
            dtype=dtype,
            sync=True,
            device_index=0,
        )

        self.assertEqual(grad_x.shape, grad_x_ref.shape)
        self.assertEqual(grad_w.shape, grad_w_ref.shape)

        # Backward uses atomicAdd -> order differs; allow slightly looser thresholds
        if dtype == np.float32:
            np.testing.assert_allclose(grad_x, grad_x_ref, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(grad_w, grad_w_ref, rtol=3e-4, atol=3e-4)
        else:
            np.testing.assert_allclose(grad_x, grad_x_ref, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(grad_w, grad_w_ref, rtol=1e-10, atol=1e-10)

        if with_bias:
            self.assertIsNotNone(grad_b)
            self.assertIsNotNone(grad_b_ref)
            assert_allclose_by_dtype(grad_b, grad_b_ref, dtype, op="conv2d_grad_b")
        else:
            self.assertIsNone(grad_b)

    def test_backward_float32_no_bias(self) -> None:
        self._run_backward(np.float32, with_bias=False)

    def test_backward_float32_with_bias(self) -> None:
        self._run_backward(np.float32, with_bias=True)

    def test_backward_float64_no_bias(self) -> None:
        self._run_backward(np.float64, with_bias=False)

    def test_backward_float64_with_bias(self) -> None:
        self._run_backward(np.float64, with_bias=True)

    def test_rejects_in_channels_mismatch(self) -> None:
        dtype = np.float32
        x = np.random.randn(1, 3, 5, 5).astype(dtype)
        w = np.random.randn(4, 2, 3, 3).astype(dtype)  # mismatch C_in=2
        with self.assertRaises(ValueError):
            _ = self.conv2d_fwd(
                self.env.lib,
                x=x,
                w=w,
                b=None,
                stride=1,
                padding=0,
                dtype=dtype,
                sync=True,
                device_index=0,
            )

    def test_backward_rejects_grad_out_shape_mismatch(self) -> None:
        dtype = np.float32
        x = np.random.randn(2, 2, 5, 5).astype(dtype)
        w = np.random.randn(3, 2, 3, 3).astype(dtype)
        b = None

        y_ref = _conv2d_forward_ref(x, w, b, stride=1, padding=0)
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        # Corrupt grad_out (wrong C_out)
        grad_out_bad = grad_out[:, :2, :, :]

        with self.assertRaises(ValueError):
            _ = self.conv2d_bwd(
                self.env.lib,
                x=x,
                w=w,
                b=b,
                grad_out=grad_out_bad,
                stride=1,
                padding=0,
                dtype=dtype,
                sync=True,
                device_index=0,
            )
