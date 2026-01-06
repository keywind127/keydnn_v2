import unittest
import warnings
from unittest.mock import patch

import numpy as np

from src.keydnn.infrastructure.ops.conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _tol(dtype: np.dtype) -> tuple[float, float]:
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return 1e-5, 1e-6
    if dtype == np.float64:
        return 1e-12, 1e-12
    # fallback (not used by native fast path)
    return 1e-5, 1e-6


def _out_hw_transpose(
    H_in: int,
    W_in: int,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
) -> tuple[int, int]:
    K_h, K_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    op_h, op_w = output_padding
    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    return H_out, W_out


def _ref_conv2d_transpose_forward_numpy(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """
    Reference conv2d_transpose forward matching KeyDNN semantics:

    - x: (N, C_in, H_in, W_in) NCHW
    - w: (C_in, C_out, K_h, K_w) IOHW
    - y: (N, C_out, H_out, W_out)

    Scatter accumulation:
      oh = hi*s_h + kh - p_h
      ow = wi*s_w + kw - p_w
      y[n, co, oh, ow] += x[n, ci, hi, wi] * w[ci, co, kh, kw]

    Bias add after accumulation.
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in2 != C_in:
        raise ValueError("in_channels mismatch in reference")

    H_out, W_out = _out_hw_transpose(
        H_in,
        W_in,
        kernel_size=(K_h, K_w),
        stride=(s_h, s_w),
        padding=(p_h, p_w),
        output_padding=(op_h, op_w),
    )
    if H_out <= 0 or W_out <= 0:
        raise ValueError("invalid output size in reference")

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


def _ref_conv2d_transpose_backward_numpy(
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference conv2d_transpose backward matching KeyDNN semantics:

    Returns:
      grad_x: same shape as x
      grad_w: same shape as w
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in2 != C_in:
        raise ValueError("in_channels mismatch in reference")

    H_out, W_out = _out_hw_transpose(
        H_in,
        W_in,
        kernel_size=(K_h, K_w),
        stride=(s_h, s_w),
        padding=(p_h, p_w),
        output_padding=(op_h, op_w),
    )
    if grad_out.shape != (N, C_out, H_out, W_out):
        raise ValueError("grad_out shape mismatch in reference")

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

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

                                go = grad_out[n, co, oh, ow]
                                grad_x[n, ci, hi, wi] += go * w[ci, co, kh, kw]
                                grad_w[ci, co, kh, kw] += xv * go

    return grad_x, grad_w


class TestConv2DTransposeOps(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    # -----------------------------
    # Deterministic forward tests
    # -----------------------------
    def test_forward_known_values_stride1_pad0(self) -> None:
        # Simple hand-checkable case:
        # x: 1x1x1x1, w: 1x1x2x2, stride=1, padding=0 => y is w * x
        x = np.array([[[[2.0]]]], dtype=np.float32)
        w = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32
        )  # (C_in=1,C_out=1,2,2)
        b = np.array([0.5], dtype=np.float32)

        y = conv2d_transpose_forward_cpu(x, w, b, stride=1, padding=0, output_padding=0)
        expected = np.array([[[[2.0, 4.0], [6.0, 8.0]]]], dtype=np.float32) + 0.5

        self.assertEqual(y.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)

    def test_forward_matches_reference_float32_float64(self) -> None:
        rng = np.random.default_rng(123)

        for dtype in (np.float32, np.float64):
            N, C_in, C_out = 2, 3, 4
            H_in, W_in = 5, 4
            K_h, K_w = 3, 2
            stride = (2, 1)
            padding = (1, 0)
            output_padding = (0, 0)

            x = rng.standard_normal((N, C_in, H_in, W_in)).astype(dtype, copy=False)
            w = rng.standard_normal((C_in, C_out, K_h, K_w)).astype(dtype, copy=False)
            b = rng.standard_normal((C_out,)).astype(dtype, copy=False)

            y = conv2d_transpose_forward_cpu(
                x, w, b, stride=stride, padding=padding, output_padding=output_padding
            )
            y_ref = _ref_conv2d_transpose_forward_numpy(
                x, w, b, stride=stride, padding=padding, output_padding=output_padding
            )

            self.assertEqual(y.dtype, dtype)
            np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)

    def test_forward_no_bias(self) -> None:
        rng = np.random.default_rng(7)

        N, C_in, C_out = 1, 2, 2
        H_in, W_in = 4, 3
        K_h, K_w = 3, 3
        stride = (1, 2)
        padding = (1, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)

        y = conv2d_transpose_forward_cpu(
            x, w, None, stride=stride, padding=padding, output_padding=output_padding
        )
        y_ref = _ref_conv2d_transpose_forward_numpy(
            x, w, None, stride=stride, padding=padding, output_padding=output_padding
        )
        np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)

    # -----------------------------
    # Backward tests
    # -----------------------------
    def test_backward_matches_reference_small(self) -> None:
        rng = np.random.default_rng(9)

        N, C_in, C_out = 2, 2, 3
        H_in, W_in = 4, 5
        K_h, K_w = 2, 3
        stride = (2, 2)
        padding = (0, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)

        H_out, W_out = _out_hw_transpose(
            H_in,
            W_in,
            kernel_size=(K_h, K_w),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        grad_out = rng.standard_normal((N, C_out, H_out, W_out), dtype=np.float32)

        grad_x, grad_w, grad_b = conv2d_transpose_backward_cpu(
            x,
            w,
            b=np.zeros((C_out,), dtype=np.float32),
            grad_out=grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        gx_ref, gw_ref = _ref_conv2d_transpose_backward_numpy(
            x,
            w,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        gb_ref = grad_out.sum(axis=(0, 2, 3)).astype(np.float32, copy=False)

        # grad_x is typically stable; keep strict unless you see drift here too.
        np.testing.assert_allclose(grad_x, gx_ref, rtol=0, atol=0)

        # grad_w is a reduction; allow tiny FP drift (native/OpenMP order differs).
        rtol, atol = _tol(grad_w.dtype)
        np.testing.assert_allclose(grad_w, gw_ref, rtol=rtol, atol=atol)

        np.testing.assert_allclose(grad_b, gb_ref, rtol=0, atol=0)

    # -----------------------------
    # Randomized stress tests
    # -----------------------------
    def test_randomized_shapes_forward_backward_float32_float64(self) -> None:
        rng = np.random.default_rng(2026)

        for dtype in (np.float32, np.float64):
            trials = 0
            while trials < 20:
                N = int(rng.integers(1, 4))
                C_in = int(rng.integers(1, 5))
                C_out = int(rng.integers(1, 6))
                H_in = int(rng.integers(2, 8))
                W_in = int(rng.integers(2, 8))

                K_h = int(rng.integers(1, 5))
                K_w = int(rng.integers(1, 5))

                s_h = int(rng.integers(1, 4))
                s_w = int(rng.integers(1, 4))

                p_h = int(rng.integers(0, K_h))
                p_w = int(rng.integers(0, K_w))

                # output_padding must be < stride per dim
                op_h = int(rng.integers(0, max(1, s_h)))
                op_w = int(rng.integers(0, max(1, s_w)))
                if op_h >= s_h:
                    op_h = 0
                if op_w >= s_w:
                    op_w = 0

                stride = (s_h, s_w)
                padding = (p_h, p_w)
                output_padding = (op_h, op_w)

                # -----------------------------
                # Guard: skip invalid output sizes
                # -----------------------------
                H_out, W_out = _out_hw_transpose(
                    H_in,
                    W_in,
                    kernel_size=(K_h, K_w),
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                if H_out <= 0 or W_out <= 0:
                    continue

                x = rng.standard_normal((N, C_in, H_in, W_in)).astype(dtype, copy=False)
                w = rng.standard_normal((C_in, C_out, K_h, K_w)).astype(
                    dtype, copy=False
                )

                use_bias = bool(rng.integers(0, 2))
                b = (
                    rng.standard_normal((C_out,)).astype(dtype, copy=False)
                    if use_bias
                    else None
                )

                # Forward
                y = conv2d_transpose_forward_cpu(
                    x,
                    w,
                    b,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                y_ref = _ref_conv2d_transpose_forward_numpy(
                    x,
                    w,
                    b,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)

                # Backward
                grad_out = rng.standard_normal((N, C_out, H_out, W_out)).astype(
                    dtype, copy=False
                )

                grad_x, grad_w, grad_b = conv2d_transpose_backward_cpu(
                    x,
                    w,
                    b=b,
                    grad_out=grad_out,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                gx_ref, gw_ref = _ref_conv2d_transpose_backward_numpy(
                    x,
                    w,
                    grad_out,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )

                np.testing.assert_allclose(grad_x, gx_ref, rtol=0, atol=0)

                rtol, atol = _tol(grad_w.dtype)
                np.testing.assert_allclose(grad_w, gw_ref, rtol=rtol, atol=atol)

                if b is None:
                    self.assertIsNone(grad_b)
                else:
                    gb_ref = grad_out.sum(axis=(0, 2, 3)).astype(dtype, copy=False)
                    np.testing.assert_allclose(grad_b, gb_ref, rtol=0, atol=0)

                trials += 1

    # -----------------------------
    # Dtype behavior / fallback
    # -----------------------------
    def test_forward_float16_falls_back_to_numpy(self) -> None:
        # float16 is allowed; native path (if present) supports only f32/f64,
        # so this should still work via reference loops.
        rng = np.random.default_rng(0)

        x = rng.standard_normal((1, 1, 3, 3)).astype(np.float16)
        w = rng.standard_normal((1, 1, 3, 3)).astype(np.float16)
        b = np.array([0.25], dtype=np.float16)

        y = conv2d_transpose_forward_cpu(x, w, b, stride=1, padding=1, output_padding=0)
        y_ref = _ref_conv2d_transpose_forward_numpy(
            x, w, b, stride=1, padding=1, output_padding=0
        )

        self.assertEqual(y.dtype, np.float16)
        np.testing.assert_allclose(
            y.astype(np.float32), y_ref.astype(np.float32), rtol=0, atol=0
        )

    def test_backward_float16_falls_back_to_numpy(self) -> None:
        rng = np.random.default_rng(1)

        x = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
        w = rng.standard_normal((2, 3, 2, 2)).astype(np.float16)

        y = conv2d_transpose_forward_cpu(
            x, w, None, stride=(2, 1), padding=(0, 0), output_padding=0
        )
        grad_out = rng.standard_normal(y.shape).astype(np.float16)

        grad_x, grad_w, grad_b = conv2d_transpose_backward_cpu(
            x,
            w,
            b=None,
            grad_out=grad_out,
            stride=(2, 1),
            padding=(0, 0),
            output_padding=0,
        )

        gx_ref, gw_ref = _ref_conv2d_transpose_backward_numpy(
            x, w, grad_out, stride=(2, 1), padding=(0, 0), output_padding=0
        )
        self.assertIsNone(grad_b)
        np.testing.assert_allclose(
            grad_x.astype(np.float32), gx_ref.astype(np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            grad_w.astype(np.float32), gw_ref.astype(np.float32), rtol=0, atol=0
        )

    # -----------------------------
    # Native load failure -> warn + fallback (only for f32/f64, output_padding==0)
    # -----------------------------
    def test_forward_warns_and_falls_back_when_native_missing(self) -> None:
        rng = np.random.default_rng(2)

        x = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
        w = rng.standard_normal((2, 4, 2, 2), dtype=np.float32)
        b = rng.standard_normal((4,), dtype=np.float32)

        expected = _ref_conv2d_transpose_forward_numpy(
            x, w, b, stride=(2, 2), padding=(1, 1), output_padding=0
        )

        with patch(
            "src.keydnn.infrastructure.native.python.conv2d_transpose_ctypes.load_keydnn_native",
            side_effect=OSError("simulated missing dll"),
        ):
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always", RuntimeWarning)
                y = conv2d_transpose_forward_cpu(
                    x, w, b, stride=(2, 2), padding=(1, 1), output_padding=0
                )

                np.testing.assert_allclose(y, expected, rtol=0, atol=0)
                self.assertTrue(
                    any(issubclass(w.category, RuntimeWarning) for w in warn_list)
                )
                self.assertTrue(
                    any("falling back to NumPy" in str(w.message) for w in warn_list)
                )

    def test_backward_warns_and_falls_back_when_native_missing(self) -> None:
        rng = np.random.default_rng(3)

        x = rng.standard_normal((2, 2, 4, 3), dtype=np.float64)
        w = rng.standard_normal((2, 3, 2, 2), dtype=np.float64)

        y = conv2d_transpose_forward_cpu(
            x, w, None, stride=(1, 2), padding=(0, 1), output_padding=0
        )
        grad_out = rng.standard_normal(y.shape).astype(np.float64, copy=False)

        gx_ref, gw_ref = _ref_conv2d_transpose_backward_numpy(
            x, w, grad_out, stride=(1, 2), padding=(0, 1), output_padding=0
        )

        with patch(
            "src.keydnn.infrastructure.native.python.conv2d_transpose_ctypes.load_keydnn_native",
            side_effect=OSError("simulated missing so"),
        ):
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always", RuntimeWarning)
                grad_x, grad_w, grad_b = conv2d_transpose_backward_cpu(
                    x,
                    w,
                    b=None,
                    grad_out=grad_out,
                    stride=(1, 2),
                    padding=(0, 1),
                    output_padding=0,
                )

                np.testing.assert_allclose(grad_x, gx_ref, rtol=0, atol=0)

                rtol, atol = _tol(grad_w.dtype)
                np.testing.assert_allclose(grad_w, gw_ref, rtol=rtol, atol=atol)

                self.assertIsNone(grad_b)

                self.assertTrue(
                    any(issubclass(w.category, RuntimeWarning) for w in warn_list)
                )
                self.assertTrue(
                    any("falling back to NumPy" in str(w.message) for w in warn_list)
                )

    # -----------------------------
    # output_padding behavior:
    # - supported by pure NumPy loops
    # - native fast-path should be skipped (op != 0)
    # -----------------------------
    def test_forward_output_padding_nonzero(self) -> None:
        rng = np.random.default_rng(4)

        x = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
        w = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
        b = rng.standard_normal((2,), dtype=np.float32)

        stride = (2, 2)
        padding = (1, 1)
        output_padding = (1, 0)  # nonzero => must use Python path

        y = conv2d_transpose_forward_cpu(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        y_ref = _ref_conv2d_transpose_forward_numpy(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)

    def test_backward_output_padding_nonzero(self) -> None:
        rng = np.random.default_rng(5)

        x = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
        w = rng.standard_normal((2, 1, 2, 2), dtype=np.float32)

        stride = (3, 2)
        padding = (0, 1)
        output_padding = (2, 1)  # must be < stride

        y = conv2d_transpose_forward_cpu(
            x, w, None, stride=stride, padding=padding, output_padding=output_padding
        )
        grad_out = rng.standard_normal(y.shape, dtype=np.float32)

        grad_x, grad_w, grad_b = conv2d_transpose_backward_cpu(
            x,
            w,
            b=None,
            grad_out=grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        gx_ref, gw_ref = _ref_conv2d_transpose_backward_numpy(
            x,
            w,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.assertIsNone(grad_b)
        np.testing.assert_allclose(grad_x, gx_ref, rtol=0, atol=0)

        rtol, atol = _tol(grad_w.dtype)
        np.testing.assert_allclose(grad_w, gw_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
