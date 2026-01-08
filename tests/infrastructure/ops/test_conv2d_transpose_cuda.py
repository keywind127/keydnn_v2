from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _conv2d_transpose_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """
    Naive NCHW/IOHW transpose-conv forward reference (matches your CPU semantics).

    y[n, co, hi*s_h + kh - p_h, wi*s_w + kw - p_w] += x[n, ci, hi, wi] * w[ci, co, kh, kw]
    Bias is added after accumulation: y[n, co] += b[co]
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    if s_h <= 0 or s_w <= 0:
        raise ValueError("stride must be positive")
    if op_h < 0 or op_w < 0:
        raise ValueError("output_padding must be non-negative")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError("output_padding must be < stride per dim")

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if H_out < 0 or W_out < 0:
        raise ValueError("invalid output size")

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - p_h
                for wi in range(W_in):
                    base_ow = wi * s_w - p_w
                    xv = x[n, ci, hi, wi]
                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                y[n, co, oh, ow] += xv * w[ci, co, kh, kw]

        if b is not None:
            for co in range(C_out):
                y[n, co, :, :] += b[co]

    return y


def _conv2d_transpose_backward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Naive transpose-conv backward reference returning (grad_x, grad_w, grad_b).

    Matches the forward scatter semantics:
      y += x * w  (scatter)
    So:
      grad_x accumulates sum(grad_out * w)
      grad_w accumulates sum(x * grad_out)
    grad_b = sum(grad_out) over (N,H_out,W_out) if b is not None.
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w

    if grad_out.shape != (N, C_out, H_out, W_out):
        raise ValueError("grad_out shape mismatch")

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - p_h
                for wi in range(W_in):
                    base_ow = wi * s_w - p_w
                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                go = grad_out[n, co, oh, ow]
                                grad_x[n, ci, hi, wi] += go * w[ci, co, kh, kw]
                                grad_w[ci, co, kh, kw] += x[n, ci, hi, wi] * go

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    return grad_x, grad_w, grad_b


class TestConv2dTransposeCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import conv2d_transpose_cuda as ops_conv_t

        cls.ops_conv_t = ops_conv_t

        cls.conv_t_fwd = resolve_func(
            ops_conv_t,
            candidates=[
                "conv2d_transpose_forward_cuda",
                "conv2d_transpose_cuda",
                "conv2d_transpose_forward",
            ],
        )
        cls.conv_t_bwd = resolve_func(
            ops_conv_t,
            candidates=[
                "conv2d_transpose_backward_cuda",
                "conv2d_transpose_backward",
            ],
        )

    def _run_forward(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        # Keep small: wrapper does H2D/D2H + kernel
        N, C_in, C_out = 2, 3, 4
        H_in, W_in = 4, 5
        K_h, K_w = 3, 2
        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        x = np.random.randn(N, C_in, H_in, W_in).astype(dtype)
        w = np.random.randn(C_in, C_out, K_h, K_w).astype(dtype)  # IOHW
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )

        y = self.conv_t_fwd(
            self.env.lib,
            x=x,
            w=w,
            b=b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dtype=dtype,
            sync=True,
            device_index=0,
        )

        self.assertEqual(y.shape, y_ref.shape)
        assert_allclose_by_dtype(y, y_ref, dtype, op="conv2d_transpose_forward")

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

        # Small sizes; backward likely uses atomics in native kernel
        N, C_in, C_out = 2, 2, 3
        H_in, W_in = 4, 4
        K_h, K_w = 3, 3
        stride = (1, 1)
        padding = (1, 1)
        output_padding = (0, 0)

        x = np.random.randn(N, C_in, H_in, W_in).astype(dtype)
        w = np.random.randn(C_in, C_out, K_h, K_w).astype(dtype)  # IOHW
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        grad_x_ref, grad_w_ref, grad_b_ref = _conv2d_transpose_backward_ref(
            x,
            w,
            b,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        grad_x, grad_w, grad_b = self.conv_t_bwd(
            self.env.lib,
            x=x,
            w=w,
            b=b,
            grad_out=grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dtype=dtype,
            sync=True,
            device_index=0,
        )

        self.assertEqual(grad_x.shape, grad_x_ref.shape)
        self.assertEqual(grad_w.shape, grad_w_ref.shape)

        # Atomics reorder summation -> allow slightly looser tolerance than forward
        if dtype == np.float32:
            np.testing.assert_allclose(grad_x, grad_x_ref, rtol=4e-4, atol=4e-4)
            np.testing.assert_allclose(grad_w, grad_w_ref, rtol=4e-4, atol=4e-4)
        else:
            np.testing.assert_allclose(grad_x, grad_x_ref, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(grad_w, grad_w_ref, rtol=2e-10, atol=2e-10)

        if with_bias:
            self.assertIsNotNone(grad_b)
            self.assertIsNotNone(grad_b_ref)
            assert_allclose_by_dtype(
                grad_b, grad_b_ref, dtype, op="conv2d_transpose_grad_b"
            )
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
        x = np.random.randn(1, 3, 4, 4).astype(dtype)
        w = np.random.randn(2, 4, 3, 3).astype(dtype)  # C_in=2 mismatch
        with self.assertRaises(ValueError):
            _ = self.conv_t_fwd(
                self.env.lib,
                x=x,
                w=w,
                b=None,
                stride=1,
                padding=0,
                output_padding=0,
                dtype=dtype,
                sync=True,
                device_index=0,
            )

    def test_backward_rejects_grad_out_shape_mismatch(self) -> None:
        dtype = np.float32
        x = np.random.randn(2, 2, 4, 4).astype(dtype)
        w = np.random.randn(2, 3, 3, 3).astype(dtype)
        b = None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=1, padding=1, output_padding=0
        )
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        # Corrupt grad_out (wrong C_out)
        grad_out_bad = grad_out[:, :2, :, :]

        with self.assertRaises(ValueError):
            _ = self.conv_t_bwd(
                self.env.lib,
                x=x,
                w=w,
                b=b,
                grad_out=grad_out_bad,
                stride=1,
                padding=1,
                output_padding=0,
                dtype=dtype,
                sync=True,
                device_index=0,
            )

    def test_rejects_output_padding_ge_stride(self) -> None:
        dtype = np.float32
        x = np.random.randn(1, 1, 3, 3).astype(dtype)
        w = np.random.randn(1, 1, 3, 3).astype(dtype)

        with self.assertRaises(ValueError):
            _ = self.conv_t_fwd(
                self.env.lib,
                x=x,
                w=w,
                b=None,
                stride=(2, 2),
                padding=0,
                output_padding=(2, 0),  # invalid: op_h == stride_h
                dtype=dtype,
                sync=True,
                device_index=0,
            )
