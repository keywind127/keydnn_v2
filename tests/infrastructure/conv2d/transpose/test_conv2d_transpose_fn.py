import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution.transpose._conv2d_transpose_function import (
    Conv2dTransposeFn,
)
from src.keydnn.infrastructure.ops.conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _tol(dtype: np.dtype) -> tuple[float, float]:
    """
    Tolerances for comparing autograd path vs kernel reference.

    Note: even on CPU, small floating-point accumulation-order differences can
    appear depending on which fast-path is taken (native vs numpy loops).
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return 1e-5, 1e-6
    if dtype == np.float64:
        # Still use a small-but-nonzero tol because the native C++ kernel path
        # may not be bit-identical to the pure numpy reference in all cases.
        return 1e-10, 1e-12
    return 1e-5, 1e-6


class TestConv2dTransposeFn(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_matches_cpu_kernel(self) -> None:
        np.random.seed(0)

        x_np = np.random.randn(2, 3, 5, 4).astype(np.float32)
        w_np = np.random.randn(3, 4, 3, 2).astype(np.float32)  # (C_in, C_out, Kh, Kw)
        b_np = np.random.randn(4).astype(np.float32)

        x = tensor_from_numpy(x_np, self.device, False)
        w = tensor_from_numpy(w_np, self.device, False)
        b = tensor_from_numpy(b_np, self.device, False)

        stride = (2, 1)
        padding = (1, 0)
        output_padding = (0, 0)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(y_ref.dtype)
        self.assertEqual(y.shape, y_ref.shape)
        np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)

    def test_forward_no_bias_matches_cpu_kernel(self) -> None:
        rng = np.random.default_rng(7)

        x_np = rng.standard_normal((1, 2, 4, 3), dtype=np.float32)
        w_np = rng.standard_normal((2, 3, 2, 3), dtype=np.float32)

        x = tensor_from_numpy(x_np, self.device, False)
        w = tensor_from_numpy(w_np, self.device, False)

        stride = (1, 2)
        padding = (1, 1)
        output_padding = (0, 0)

        ctx = Context(parents=(x, w), backward_fn=lambda _: (None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(y_ref.dtype)
        self.assertEqual(y.shape, y_ref.shape)
        np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)

    def test_backward_via_autograd_matches_cpu_kernel(self) -> None:
        """
        out = conv2d_transpose(x,w,b); loss = out.sum(); loss.backward()
        Compare x.grad/w.grad/b.grad with conv2d_transpose_backward_cpu under grad_out=ones.
        """
        rng = np.random.default_rng(1)

        x_np = rng.standard_normal((1, 2, 4, 3), dtype=np.float32)
        w_np = rng.standard_normal((2, 3, 2, 2), dtype=np.float32)
        b_np = rng.standard_normal((3,), dtype=np.float32)

        x = tensor_from_numpy(x_np, self.device, True)
        w = tensor_from_numpy(w_np, self.device, True)
        b = tensor_from_numpy(b_np, self.device, True)

        stride = (2, 1)
        padding = (0, 1)
        output_padding = (0, 0)

        ctx = Context(
            parents=(x, w, b),
            backward_fn=lambda grad_out: Conv2dTransposeFn.backward(ctx, grad_out),
        )
        out = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        out.requires_grad = True
        out._set_ctx(ctx)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)

        # Reference grads with grad_out=ones (matching loss=sum)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        grad_out_np = np.ones_like(y_ref, dtype=np.float32)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            b_np,
            grad_out_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(np.float32)
        np.testing.assert_allclose(x.grad.to_numpy(), gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(w.grad.to_numpy(), gw_ref, rtol=rtol, atol=atol)

        # Bias grad tends to be a straightforward reduction; keep slightly tighter
        np.testing.assert_allclose(b.grad.to_numpy(), gb_ref, rtol=1e-6, atol=1e-6)

    def test_backward_none_bias_returns_none(self) -> None:
        rng = np.random.default_rng(3)

        x_np = rng.standard_normal((2, 2, 3, 3), dtype=np.float32)
        w_np = rng.standard_normal((2, 1, 3, 2), dtype=np.float32)

        x = tensor_from_numpy(x_np, self.device, True)
        w = tensor_from_numpy(w_np, self.device, True)

        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        ctx = Context(
            parents=(x, w),
            backward_fn=lambda grad_out: Conv2dTransposeFn.backward(ctx, grad_out),
        )
        out = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        out.requires_grad = True
        out._set_ctx(ctx)

        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        # no bias => no b.grad to check
        # (Conv2dTransposeFn.backward returns (grad_x, grad_w) only)

    def test_forward_respects_out_requires_grad_flag(self) -> None:
        rng = np.random.default_rng(11)

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
        w_np = rng.standard_normal((1, 2, 2, 2), dtype=np.float32)
        b_np = rng.standard_normal((2,), dtype=np.float32)

        # Case A: parents do not require grad -> output should not require grad
        x0 = tensor_from_numpy(x_np, self.device, False)
        w0 = tensor_from_numpy(w_np, self.device, False)
        b0 = tensor_from_numpy(b_np, self.device, False)

        ctx0 = Context(parents=(x0, w0, b0), backward_fn=lambda _: (None, None, None))
        y0 = Conv2dTransposeFn.forward(
            ctx0, x0, w0, b0, stride=1, padding=0, output_padding=0
        )
        self.assertFalse(y0.requires_grad)

        # Case B: one parent requires grad -> output should require grad
        x1 = tensor_from_numpy(x_np, self.device, True)
        w1 = tensor_from_numpy(w_np, self.device, False)
        b1 = tensor_from_numpy(b_np, self.device, False)

        ctx1 = Context(parents=(x1, w1, b1), backward_fn=lambda _: (None, None, None))
        y1 = Conv2dTransposeFn.forward(
            ctx1, x1, w1, b1, stride=1, padding=0, output_padding=0
        )
        self.assertTrue(y1.requires_grad)

    def test_forward_rejects_mismatched_devices(self) -> None:
        rng = np.random.default_rng(0)

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
        w_np = rng.standard_normal((1, 1, 2, 2), dtype=np.float32)

        x = tensor_from_numpy(x_np, Device("cpu"), False)
        w = tensor_from_numpy(w_np, Device("cpu"), False)

        # Fake: different device string should raise
        # (We can't allocate actual CUDA tensors here, so emulate mismatch by swapping)
        # If your Device supports "cuda:0" but Tensor can't allocate it in this test env,
        # you can remove this test. Keeping it guarded:
        try:
            cuda_dev = Device("cuda:0")
        except Exception:
            self.skipTest("CUDA device string not supported in this environment")

        w_other = Tensor(shape=w.shape, device=cuda_dev, requires_grad=False, ctx=None)
        w_other.copy_from_numpy(w_np)

        ctx = Context(parents=(x, w_other), backward_fn=lambda _: (None, None))
        with self.assertRaises(RuntimeError):
            _ = Conv2dTransposeFn.forward(ctx, x, w_other, None, stride=1, padding=0)

    def test_forward_output_padding_nonzero_works_cpu(self) -> None:
        rng = np.random.default_rng(4)

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=np.float32)
        w_np = rng.standard_normal((1, 2, 3, 3), dtype=np.float32)
        b_np = rng.standard_normal((2,), dtype=np.float32)

        x = tensor_from_numpy(x_np, self.device, False)
        w = tensor_from_numpy(w_np, self.device, False)
        b = tensor_from_numpy(b_np, self.device, False)

        stride = (2, 2)
        padding = (1, 1)
        output_padding = (1, 0)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol(y_ref.dtype)
        np.testing.assert_allclose(y.to_numpy(), y_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
