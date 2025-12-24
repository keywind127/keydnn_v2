import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor


class TestTensorZerosOnes(unittest.TestCase):
    def test_zeros_shape_dtype_and_values(self):
        x = Tensor.zeros(shape=(2, 3), device=Device("cpu"), requires_grad=False)
        arr = x.to_numpy()
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, np.zeros((2, 3), dtype=np.float32))

    def test_ones_shape_dtype_and_values(self):
        x = Tensor.ones(shape=(2, 3), device=Device("cpu"), requires_grad=False)
        arr = x.to_numpy()
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, np.ones((2, 3), dtype=np.float32))

    def test_requires_grad_flag(self):
        x = Tensor.ones(shape=(1, 4), device=Device("cpu"), requires_grad=True)
        self.assertTrue(x.requires_grad)
