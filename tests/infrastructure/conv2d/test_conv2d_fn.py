import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.convolution._conv2d_function import Conv2dFn
from src.keydnn.infrastructure.ops.conv2d_cpu import conv2d_forward_cpu, conv2d_backward_cpu


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dFn(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_matches_cpu(self):
        np.random.seed(0)
        x_np = np.random.randn(2, 3, 6, 5).astype(np.float32)
        w_np = np.random.randn(4, 3, 3, 3).astype(np.float32)
        b_np = np.random.randn(4).astype(np.float32)

        x = tensor_from_numpy(x_np, self.device, False)
        w = tensor_from_numpy(w_np, self.device, False)
        b = tensor_from_numpy(b_np, self.device, False)

        stride = (2, 1)
        padding = (1, 2)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dFn.forward(ctx, x, w, b, stride=stride, padding=padding)

        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=stride, padding=padding)
        self.assertEqual(y.shape, y_ref.shape)
        self.assertTrue(np.allclose(y.to_numpy(), y_ref, atol=1e-6, rtol=1e-6))

    def test_backward_via_autograd_matches_cpu(self):
        """
        out = conv2d(x,w,b); loss = out.sum(); loss.backward()
        Compare x.grad/w.grad/b.grad with conv2d_backward_cpu under grad_out=ones.
        """
        np.random.seed(1)
        x_np = np.random.randn(1, 2, 5, 4).astype(np.float32)
        w_np = np.random.randn(3, 2, 3, 2).astype(np.float32)
        b_np = np.random.randn(3).astype(np.float32)

        x = tensor_from_numpy(x_np, self.device, True)
        w = tensor_from_numpy(w_np, self.device, True)
        b = tensor_from_numpy(b_np, self.device, True)

        stride = (1, 2)
        padding = (1, 0)

        ctx = Context(
            parents=(x, w, b),
            backward_fn=lambda grad_out: Conv2dFn.backward(ctx, grad_out),
        )
        out = Conv2dFn.forward(ctx, x, w, b, stride=stride, padding=padding)
        out.requires_grad = True
        out._set_ctx(ctx)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)

        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=stride, padding=padding)
        grad_out_np = np.ones_like(y_ref, dtype=np.float32)
        gx_ref, gw_ref, gb_ref = conv2d_backward_cpu(
            x_np, w_np, b_np, grad_out_np, stride=stride, padding=padding
        )

        self.assertTrue(np.allclose(x.grad.to_numpy(), gx_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(np.allclose(w.grad.to_numpy(), gw_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(np.allclose(b.grad.to_numpy(), gb_ref, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
