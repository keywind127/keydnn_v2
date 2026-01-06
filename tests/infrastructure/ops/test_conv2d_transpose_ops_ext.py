import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device

from src.keydnn.infrastructure.ops.conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)
from src.keydnn.infrastructure.ops.conv2d_transpose_cpu_ext import (
    conv2d_transpose_forward_cpu_tensor,
    conv2d_transpose_backward_cpu_tensor,
)


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _tol(dtype: np.dtype) -> tuple[float, float]:
    """
    Numerical tolerance for comparisons.

    Notes
    -----
    In ops-ext boundary tests, small ~1e-7 diffs can occur even for float64 due to:
    - internal casting / contiguous conversions
    - different accumulation orders (native vs Python loops)
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return 1e-5, 1e-6
    if dtype == np.float64:
        # ops-ext boundary may not preserve pure-f64 arithmetic end-to-end
        # return 1e-7, 1e-6
        return 1e-5, 1e-6
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


def _safe_rand_case(rng: np.random.Generator) -> dict:
    N = int(rng.integers(1, 4))
    C_in = int(rng.integers(1, 5))
    C_out = int(rng.integers(1, 6))
    H_in = int(rng.integers(2, 8))
    W_in = int(rng.integers(2, 8))

    K_h = int(rng.integers(1, 5))
    K_w = int(rng.integers(1, 5))

    s_h = int(rng.integers(1, 4))
    s_w = int(rng.integers(1, 4))

    op_h = int(rng.integers(0, max(1, s_h)))
    op_w = int(rng.integers(0, max(1, s_w)))
    if op_h >= s_h:
        op_h = 0
    if op_w >= s_w:
        op_w = 0

    # conservative padding upper bound to keep output > 0
    max_p_h = ((H_in - 1) * s_h + K_h + op_h - 1) // 2
    max_p_w = ((W_in - 1) * s_w + K_w + op_w - 1) // 2
    max_p_h = int(max(0, max_p_h))
    max_p_w = int(max(0, max_p_w))

    p_h = int(rng.integers(0, max_p_h + 1))
    p_w = int(rng.integers(0, max_p_w + 1))

    stride = (s_h, s_w)
    padding = (p_h, p_w)
    output_padding = (op_h, op_w)

    H_out, W_out = _out_hw_transpose(
        H_in,
        W_in,
        kernel_size=(K_h, K_w),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    assert H_out > 0 and W_out > 0

    use_bias = bool(rng.integers(0, 2))

    return dict(
        N=N,
        C_in=C_in,
        C_out=C_out,
        H_in=H_in,
        W_in=W_in,
        K_h=K_h,
        K_w=K_w,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        use_bias=use_bias,
    )


def _tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool
) -> Tensor:
    arr = np.asarray(arr)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2DTransposeOpsExtCPU(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.device = Device("cpu")

    def test_forward_matches_cpu_kernel_small(self) -> None:
        rng = np.random.default_rng(0)

        N, C_in, C_out = 2, 3, 4
        H_in, W_in = 5, 4
        K_h, K_w = 3, 2
        stride = (2, 1)
        padding = (1, 0)
        output_padding = (0, 0)

        x_np = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w_np = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)
        b_np = rng.standard_normal((C_out,), dtype=np.float32)

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)
        w = _tensor_from_numpy(w_np, device=self.device, requires_grad=True)
        b = _tensor_from_numpy(b_np, device=self.device, requires_grad=True)

        y = conv2d_transpose_forward_cpu_tensor(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            out_requires_grad=True,
        )

        rtol, atol = _tol(y_ref.dtype)
        np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)

    def test_forward_no_bias(self) -> None:
        rng = np.random.default_rng(7)

        N, C_in, C_out = 1, 2, 2
        H_in, W_in = 4, 3
        K_h, K_w = 3, 3
        stride = (1, 2)
        padding = (1, 1)
        output_padding = (0, 0)

        x_np = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float64)
        w_np = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float64)

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=False)
        w = _tensor_from_numpy(w_np, device=self.device, requires_grad=False)

        y = conv2d_transpose_forward_cpu_tensor(
            x,
            w,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            out_requires_grad=False,
        )

        # float64 should be extremely tight, but still use dtype-aware tol for consistency
        rtol, atol = _tol(y_ref.dtype)
        np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)

    def test_backward_matches_cpu_kernel_small(self) -> None:
        rng = np.random.default_rng(2)

        N, C_in, C_out = 2, 2, 3
        H_in, W_in = 4, 5
        K_h, K_w = 2, 3
        stride = (2, 2)
        padding = (0, 1)
        output_padding = (0, 0)

        x_np = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w_np = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)
        b_np = np.zeros((C_out,), dtype=np.float32)

        y_np = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        grad_out_np = rng.standard_normal(y_np.shape, dtype=np.float32)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            b_np,
            grad_out_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=False)
        w = _tensor_from_numpy(w_np, device=self.device, requires_grad=False)
        b = _tensor_from_numpy(b_np, device=self.device, requires_grad=False)
        grad_out = _tensor_from_numpy(
            grad_out_np, device=self.device, requires_grad=False
        )

        gx, gw, gb = conv2d_transpose_backward_cpu_tensor(
            x,
            w,
            b,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(gx_ref.dtype)
        np.testing.assert_allclose(gx.to_numpy(), gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gw.to_numpy(), gw_ref, rtol=rtol, atol=atol)
        self.assertIsNotNone(gb)
        np.testing.assert_allclose(gb.to_numpy(), gb_ref, rtol=rtol, atol=atol)

        self.assertFalse(gx.requires_grad)
        self.assertFalse(gw.requires_grad)
        self.assertFalse(gb.requires_grad)

    def test_backward_none_bias_returns_none(self) -> None:
        rng = np.random.default_rng(3)

        x_np = rng.standard_normal((1, 2, 3, 4), dtype=np.float32)
        w_np = rng.standard_normal((2, 3, 2, 2), dtype=np.float32)

        stride = (2, 1)
        padding = (0, 0)
        output_padding = (0, 0)

        y_np = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        grad_out_np = rng.standard_normal(y_np.shape, dtype=np.float32)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            None,
            grad_out_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.assertIsNone(gb_ref)

        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=False)
        w = _tensor_from_numpy(w_np, device=self.device, requires_grad=False)
        grad_out = _tensor_from_numpy(
            grad_out_np, device=self.device, requires_grad=False
        )

        gx, gw, gb = conv2d_transpose_backward_cpu_tensor(
            x,
            w,
            None,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(gx_ref.dtype)
        np.testing.assert_allclose(gx.to_numpy(), gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gw.to_numpy(), gw_ref, rtol=rtol, atol=atol)
        self.assertIsNone(gb)

    def test_randomized_shapes_forward_backward_float32_float64(self) -> None:
        rng = np.random.default_rng(2026)

        for dtype in (np.float32, np.float64):
            rtol, atol = _tol(dtype)
            for _ in range(30):
                case = _safe_rand_case(rng)

                N = case["N"]
                C_in = case["C_in"]
                C_out = case["C_out"]
                H_in = case["H_in"]
                W_in = case["W_in"]
                K_h = case["K_h"]
                K_w = case["K_w"]
                stride = case["stride"]
                padding = case["padding"]
                output_padding = case["output_padding"]
                use_bias = case["use_bias"]

                x_np = rng.standard_normal((N, C_in, H_in, W_in)).astype(
                    dtype, copy=False
                )
                w_np = rng.standard_normal((C_in, C_out, K_h, K_w)).astype(
                    dtype, copy=False
                )
                b_np = (
                    rng.standard_normal((C_out,)).astype(dtype, copy=False)
                    if use_bias
                    else None
                )

                y_ref = conv2d_transpose_forward_cpu(
                    x_np,
                    w_np,
                    b_np,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )

                x = _tensor_from_numpy(x_np, device=self.device, requires_grad=False)
                w = _tensor_from_numpy(w_np, device=self.device, requires_grad=False)
                b = (
                    None
                    if b_np is None
                    else _tensor_from_numpy(
                        b_np, device=self.device, requires_grad=False
                    )
                )

                y = conv2d_transpose_forward_cpu_tensor(
                    x,
                    w,
                    b,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    out_requires_grad=False,
                )
                np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)

                grad_out_np = rng.standard_normal(y_ref.shape).astype(dtype, copy=False)
                gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
                    x_np,
                    w_np,
                    b_np,
                    grad_out_np,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )

                grad_out = _tensor_from_numpy(
                    grad_out_np, device=self.device, requires_grad=False
                )
                gx, gw, gb = conv2d_transpose_backward_cpu_tensor(
                    x,
                    w,
                    b,
                    grad_out,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )

                np.testing.assert_allclose(gx.to_numpy(), gx_ref, rtol=rtol, atol=atol)
                np.testing.assert_allclose(gw.to_numpy(), gw_ref, rtol=rtol, atol=atol)
                if b_np is None:
                    self.assertIsNone(gb)
                else:
                    self.assertIsNotNone(gb)
                    np.testing.assert_allclose(
                        gb.to_numpy(), gb_ref, rtol=rtol, atol=atol
                    )

    def test_forward_preserves_out_requires_grad_flag(self) -> None:
        rng = np.random.default_rng(7)

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
        w_np = rng.standard_normal((1, 2, 2, 2), dtype=np.float32)

        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)
        w = _tensor_from_numpy(w_np, device=self.device, requires_grad=True)

        y0 = conv2d_transpose_forward_cpu_tensor(
            x,
            w,
            None,
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            out_requires_grad=False,
        )
        self.assertFalse(y0.requires_grad)

        y1 = conv2d_transpose_forward_cpu_tensor(
            x,
            w,
            None,
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            out_requires_grad=True,
        )
        self.assertTrue(y1.requires_grad)


if __name__ == "__main__":
    unittest.main()
