import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from src.keydnn.infrastructure.flatten._flatten_module import Flatten
from src.keydnn.infrastructure.fully_connected._linear import Linear
from src.keydnn.infrastructure._activations import ReLU, Softmax


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
        # Input: NCHW
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        # Layers
        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,  # keep 8x8
            bias=True,
            device=self.device,
        )
        relu = ReLU()
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)  # 8x8 -> 4x4
        flatten = Flatten()

        # Flattened dim: C * H * W = 4 * 4 * 4 = 64
        fc = Linear(in_features=64, out_features=5, device=self.device)
        softmax = Softmax()

        # Forward
        y = conv.forward(x)  # (2, 4, 8, 8)
        y = relu.forward(y)  # (2, 4, 8, 8)
        y = pool.forward(y)  # (2, 4, 4, 4)
        y = flatten.forward(y)  # (2, 64)
        y = fc.forward(y)  # (2, 5)
        y = softmax.forward(y)  # (2, 5)

        # Forward sanity
        self.assertEqual(y.shape, (2, 5))
        y_np = y.to_numpy()
        self.assertTrue(np.all(np.isfinite(y_np)))
        self.assertTrue(np.all(y_np >= 0.0))

        # Softmax rows should sum to ~1
        row_sums = y_np.sum(axis=1)
        self.assertTrue(
            np.allclose(row_sums, np.ones((2,), dtype=np.float32), atol=1e-5, rtol=1e-5)
        )

        # Backward
        loss = y.sum()
        loss.backward()

        # Grad checks
        self.assertIsNotNone(x.grad)
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))

        self.assertIsNotNone(conv.weight.grad)
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))
        if conv.bias is not None:
            self.assertIsNotNone(conv.bias.grad)
            self.assertTrue(np.all(np.isfinite(conv.bias.grad.to_numpy())))

        # Linear parameters should also receive gradients
        self.assertIsNotNone(fc.weight.grad)
        self.assertTrue(np.all(np.isfinite(fc.weight.grad.to_numpy())))
        if fc.bias is not None:
            self.assertIsNotNone(fc.bias.grad)
            self.assertTrue(np.all(np.isfinite(fc.bias.grad.to_numpy())))


if __name__ == "__main__":
    unittest.main()
