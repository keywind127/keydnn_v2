import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
)


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dPoolingCompatibility(unittest.TestCase):
    """
    Prove Conv2d modules are compatible with pooling modules by verifying:
    - forward executes and output shapes are correct
    - backward executes and gradients propagate to:
        x, conv.weight, conv.bias (if enabled)
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def _run_chain(self, pool_module, *, expect_shape: tuple[int, ...]) -> None:
        # Input
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        # Conv keeps spatial size with padding=1, k=3
        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )

        # Forward
        y = conv.forward(x)
        z = pool_module.forward(y)

        self.assertEqual(z.shape, expect_shape)
        self.assertTrue(np.all(np.isfinite(z.to_numpy())))

        # Backward: scalar loss
        loss = z.sum()
        loss.backward()

        # Gradients must flow through pooling back into conv + x
        self.assertIsNotNone(x.grad, "x.grad should not be None")
        self.assertIsNotNone(conv.weight.grad, "conv.weight.grad should not be None")
        if conv.bias is not None:
            self.assertIsNotNone(conv.bias.grad, "conv.bias.grad should not be None")

        # Shapes must match
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        if conv.bias is not None:
            self.assertEqual(conv.bias.grad.shape, conv.bias.shape)

        # Finite gradients
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))
        if conv.bias is not None:
            self.assertTrue(np.all(np.isfinite(conv.bias.grad.to_numpy())))

    def test_conv2d_maxpool2d_forward_backward(self):
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Conv output: (2,4,8,8) -> MaxPool k=2,s=2 => (2,4,4,4)
        self._run_chain(pool, expect_shape=(2, 4, 4, 4))

    def test_conv2d_avgpool2d_forward_backward(self):
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        self._run_chain(pool, expect_shape=(2, 4, 4, 4))

    def test_conv2d_globalavgpool2d_forward_backward(self):
        pool = GlobalAvgPool2d()
        # (2,4,8,8) -> (2,4,1,1)
        self._run_chain(pool, expect_shape=(2, 4, 1, 1))


if __name__ == "__main__":
    unittest.main()
