import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure._activations import Sigmoid


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestGradAccumulationAndZeroGrad(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_backward_accumulates_and_zero_grad_resets(self):
        x_np = np.random.randn(1, 2, 6, 6).astype(np.float32)
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

        # First backward
        y1 = act.forward(conv.forward(x))
        loss1 = y1.sum()
        loss1.backward()

        gx1 = x.grad.to_numpy().copy()
        gw1 = conv.weight.grad.to_numpy().copy()
        gb1 = conv.bias.grad.to_numpy().copy() if conv.bias is not None else None

        # Second backward without zeroing grads => should accumulate
        y2 = act.forward(conv.forward(x))
        loss2 = y2.sum()
        loss2.backward()

        gx2 = x.grad.to_numpy()
        gw2 = conv.weight.grad.to_numpy()
        gb2 = conv.bias.grad.to_numpy() if conv.bias is not None else None

        self.assertTrue(np.allclose(gx2, 2.0 * gx1, atol=1e-6, rtol=1e-6))
        self.assertTrue(np.allclose(gw2, 2.0 * gw1, atol=1e-6, rtol=1e-6))
        if gb1 is not None and gb2 is not None:
            self.assertTrue(np.allclose(gb2, 2.0 * gb1, atol=1e-6, rtol=1e-6))

        # Now zero grads and ensure they reset
        x.zero_grad()
        conv.weight.zero_grad()
        if conv.bias is not None:
            conv.bias.zero_grad()

        self.assertIsNone(x.grad)
        self.assertIsNone(conv.weight.grad)
        if conv.bias is not None:
            self.assertIsNone(conv.bias.grad)


if __name__ == "__main__":
    unittest.main()
