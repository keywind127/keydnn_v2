import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor, Context

from src.keydnn.infrastructure.pooling._pooling_function import (
    MaxPool2dFn,
    AvgPool2dFn,
    GlobalAvgPool2dFn,
)

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestPool2dFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_maxpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(1, 2, 5, 6).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = MaxPool2dFn.forward(ctx, x, kernel_size=2, stride=2, padding=0)

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy(grad_out_np, self.device, requires_grad=False)

        (grad_x,) = MaxPool2dFn.backward(ctx, grad_out)

        # CPU reference
        y_ref, argmax_idx = maxpool2d_forward_cpu(
            x_np, kernel_size=2, stride=2, padding=0
        )
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = maxpool2d_backward_cpu(
            grad_out_np,
            argmax_idx,
            x_shape=x_np.shape,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.assertIsNotNone(grad_x)
        self.assertTrue(
            np.allclose(grad_x.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )

    def test_avgpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(2, 3, 6, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = AvgPool2dFn.forward(
            ctx, x, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0)
        )

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy(grad_out_np, self.device, requires_grad=False)

        (grad_x,) = AvgPool2dFn.backward(ctx, grad_out)

        # CPU reference
        y_ref = avgpool2d_forward_cpu(
            x_np, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0)
        )
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = avgpool2d_backward_cpu(
            grad_out_np,
            x_shape=x_np.shape,
            kernel_size=(2, 3),
            stride=(2, 1),
            padding=(1, 0),
        )

        self.assertIsNotNone(grad_x)
        self.assertTrue(
            np.allclose(grad_x.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )

    def test_global_avgpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(2, 4, 3, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = GlobalAvgPool2dFn.forward(ctx, x)

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy(grad_out_np, self.device, requires_grad=False)

        (grad_x,) = GlobalAvgPool2dFn.backward(ctx, grad_out)

        y_ref = global_avgpool2d_forward_cpu(x_np)
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = global_avgpool2d_backward_cpu(grad_out_np, x_shape=x_np.shape)

        self.assertIsNotNone(grad_x)
        self.assertTrue(
            np.allclose(grad_x.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
