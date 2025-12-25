import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
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


class TestPool2dModule(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_maxpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(1, 2, 5, 6).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        y = pool.forward(x)

        # loss = sum(y) => grad_out = ones
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

        grad_out_np = np.ones(y.shape, dtype=np.float32)
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

        self.assertTrue(
            np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )

    def test_avgpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(2, 3, 6, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        pool = AvgPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(1, 0))
        y = pool.forward(x)

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

        grad_out_np = np.ones(y.shape, dtype=np.float32)
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

        self.assertTrue(
            np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )

    def test_global_avgpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(2, 4, 3, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        pool = GlobalAvgPool2d()
        y = pool.forward(x)

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

        grad_out_np = np.ones(y.shape, dtype=np.float32)
        y_ref = global_avgpool2d_forward_cpu(x_np)
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = global_avgpool2d_backward_cpu(grad_out_np, x_shape=x_np.shape)

        self.assertTrue(
            np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-6, rtol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
