from unittest import TestCase
import unittest

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class _TensorFactoryMixin:
    def _tensor_from_numpy(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        t = Tensor(arr.shape, Device("cpu"), requires_grad=requires_grad)
        t.copy_from_numpy(arr)
        return t


class TestTensorSqrtForwardBackward(TestCase, _TensorFactoryMixin):
    def test_sqrt_forward_matches_numpy(self):
        x_np = np.random.rand(3, 4).astype(np.float32) + 1e-3
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sqrt()
        expected = np.sqrt(x_np)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sqrt_backward_matches_closed_form(self):
        # y = sqrt(x), loss = sum(y) => dL/dx = 0.5 / sqrt(x)
        x_np = np.random.rand(2, 3).astype(np.float32) + 1e-3
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.sqrt()
        loss = y.sum()
        loss.backward()

        expected = 0.5 / np.sqrt(x_np)
        self.assertIsNotNone(x.grad)
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sqrt_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.sqrt()


if __name__ == "__main__":
    unittest.main()
