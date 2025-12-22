import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure.ops.conv2d_cpu import conv2d_forward_cpu


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dHandcheckForward(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_forward_all_ones_matches_expected(self):
        # x: ones, w: ones, bias=None, stride=1, padding=0
        # For 2x2 kernel over 3x3 ones, each output = 4.
        x_np = np.ones((1, 1, 3, 3), dtype=np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=False)

        conv = Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 2),
            stride=1,
            padding=0,
            bias=False,
            device=self.device,
        )

        conv.weight.copy_from_numpy(np.ones((1, 1, 2, 2), dtype=np.float32))

        y = conv.forward(x)
        expected = np.array([[[[4.0, 4.0], [4.0, 4.0]]]], dtype=np.float32)

        self.assertEqual(y.shape, expected.shape)
        self.assertTrue(np.allclose(y.to_numpy(), expected, atol=1e-6, rtol=1e-6))

    def test_forward_impulse_matches_kernel_placement(self):
        # Impulse input: single 1 in center, padding=1, kernel=3x3
        # With stride=1, output should equal the kernel "seen" at each location.
        x_np = np.zeros((1, 1, 5, 5), dtype=np.float32)
        x_np[0, 0, 2, 2] = 1.0
        x = tensor_from_numpy(x_np, self.device, requires_grad=False)

        conv = Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
            device=self.device,
        )

        # Use a clearly identifiable kernel
        k = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        conv.weight.copy_from_numpy(k)

        y = conv.forward(x)

        # Reference using CPU op directly (hand-check style but robust)
        y_ref = conv2d_forward_cpu(x_np, k, b=None, stride=(1, 1), padding=(1, 1))

        self.assertEqual(y.shape, y_ref.shape)
        self.assertTrue(np.allclose(y.to_numpy(), y_ref, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
