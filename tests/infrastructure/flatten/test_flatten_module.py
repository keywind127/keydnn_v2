import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.flatten._flatten_module import Flatten


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestFlattenModule(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_flatten_module_forward_backward(self):
        x_np = np.random.randn(3, 2, 4, 4).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        flatten = Flatten()
        y = flatten.forward(x)

        self.assertEqual(y.shape, (3, 2 * 4 * 4))
        self.assertTrue(np.allclose(y.to_numpy(), x_np.reshape(3, -1)))

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        expected = np.ones_like(x_np, dtype=np.float32)
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
