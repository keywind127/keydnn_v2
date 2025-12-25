import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
)
from src.keydnn.infrastructure._activations import Sigmoid


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dPoolingActivationCompatibility(unittest.TestCase):
    """
    Prove Conv2d -> Pool -> Activation chains are compatible in both forward/backward.
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def _run_chain(self, pool_module, *, expect_shape: tuple[int, ...]) -> None:
        x_np = np.random.randn(1, 2, 8, 8).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        conv = Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )
        act = Sigmoid()

        y = conv.forward(x)
        p = pool_module.forward(y)
        z = act.forward(p)

        self.assertEqual(z.shape, expect_shape)
        z_np = z.to_numpy()
        self.assertTrue(np.all(z_np >= 0.0))
        self.assertTrue(np.all(z_np <= 1.0))

        loss = z.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        if conv.bias is not None:
            self.assertIsNotNone(conv.bias.grad)

        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))
        if conv.bias is not None:
            self.assertTrue(np.all(np.isfinite(conv.bias.grad.to_numpy())))

    def test_conv2d_maxpool_sigmoid_chain(self):
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (1,3,8,8)->(1,3,4,4)
        self._run_chain(pool, expect_shape=(1, 3, 4, 4))

    def test_conv2d_avgpool_sigmoid_chain(self):
        pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        self._run_chain(pool, expect_shape=(1, 3, 4, 4))

    def test_conv2d_globalavgpool_sigmoid_chain(self):
        pool = GlobalAvgPool2d()
        self._run_chain(pool, expect_shape=(1, 3, 1, 1))


if __name__ == "__main__":
    unittest.main()
