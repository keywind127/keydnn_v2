import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from src.keydnn.infrastructure.flatten._flatten_module import Flatten
from src.keydnn.infrastructure.fully_connected._linear import Linear
from src.keydnn.infrastructure.activations._modules import ReLU, Softmax


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestCNNEndToEndChain(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_cnn_chain_forward_and_backward(self):
        """
        Build and execute a minimal CNN graph:

            x -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> Softmax -> sum -> backward

        and verify gradients propagate through all layers.
        """

        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )
        relu = ReLU()
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        flatten = Flatten()

        fc = Linear(in_features=64, out_features=5, device=self.device)
        softmax = Softmax()

        y = conv.forward(x)
        y = relu.forward(y)
        y = pool.forward(y)
        y = flatten.forward(y)
        y = fc.forward(y)
        y = softmax.forward(y)

        self.assertEqual(y.shape, (2, 5))
        y_np = y.to_numpy()
        self.assertTrue(np.all(np.isfinite(y_np)))
        self.assertTrue(np.all(y_np >= 0.0))

        row_sums = y_np.sum(axis=1)
        self.assertTrue(
            np.allclose(row_sums, np.ones((2,), dtype=np.float32), atol=1e-5, rtol=1e-5)
        )

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))

        self.assertIsNotNone(conv.weight.grad)
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))
        if conv.bias is not None:
            self.assertIsNotNone(conv.bias.grad)
            self.assertTrue(np.all(np.isfinite(conv.bias.grad.to_numpy())))

        self.assertIsNotNone(fc.weight.grad)
        self.assertTrue(np.all(np.isfinite(fc.weight.grad.to_numpy())))
        if fc.bias is not None:
            self.assertIsNotNone(fc.bias.grad)
            self.assertTrue(np.all(np.isfinite(fc.bias.grad.to_numpy())))


if __name__ == "__main__":
    unittest.main()
