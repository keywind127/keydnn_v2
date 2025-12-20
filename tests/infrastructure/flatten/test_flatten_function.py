import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.infrastructure.flatten._flatten_function import FlattenFn


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestFlattenFn(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_flatten_forward_shape(self):
        x_np = np.random.randn(4, 3, 5, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=False)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: (None,))
        y = FlattenFn.forward(ctx, x)

        self.assertEqual(y.shape, (4, 3 * 5 * 5))
        self.assertTrue(np.allclose(y.to_numpy(), x_np.reshape(4, -1)))

    def test_flatten_backward_sum_is_ones(self):
        x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        ctx = Context(
            parents=(x,),
            backward_fn=lambda grad_out: FlattenFn.backward(ctx, grad_out),
        )
        y = FlattenFn.forward(ctx, x)

        # IMPORTANT: mimic Module behavior (attach ctx to the output)
        if x.requires_grad:
            y.requires_grad = True
            y._set_ctx(ctx)

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        expected = np.ones_like(x_np, dtype=np.float32)
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
