import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dNoBias(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_conv2d_no_bias_forward_and_backward(self):
        x_np = np.random.randn(2, 3, 7, 6).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=self.device,
        )

        y = conv.forward(x)
        self.assertEqual(y.shape, (2, 4, 7, 6))

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNone(conv.bias, "bias should be None when bias=False")

        # shape checks
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)

        # finite grad checks
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))


if __name__ == "__main__":
    unittest.main()
