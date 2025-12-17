from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.domain._device import Device


class TestTensorInfrastructure(TestCase):

    def test_tensor_initialization_properties(self):
        shape = (2, 3)
        device_cpu = Device("cpu")
        device_cuda = Device("cuda:0")

        tensor_cpu = Tensor(shape, device_cpu)
        self.assertEqual(tensor_cpu.shape, shape)
        self.assertEqual(str(tensor_cpu.device), str(device_cpu))

        tensor_cuda = Tensor(shape, device_cuda)
        self.assertEqual(tensor_cuda.shape, shape)
        self.assertEqual(str(tensor_cuda.device), str(device_cuda))

    def test_cpu_tensor_to_numpy_contract(self):
        """CPU tensor should allocate float32 ndarray, correct shape, initialized to zeros."""
        shape = (2, 3)
        tensor = Tensor(shape, Device("cpu"))

        arr = tensor.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(arr == 0.0))

    def test_cpu_tensor_fill_contract(self):
        """fill() should update CPU tensor values without direct storage access."""
        shape = (2, 2)
        tensor = Tensor(shape, Device("cpu"))

        tensor.fill(1.0)

        expected = np.ones(shape, dtype=np.float32)
        self.assertTrue(np.array_equal(tensor.to_numpy(), expected))

    def test_cuda_tensor_debug_repr_contract(self):
        """CUDA placeholder should have a stable debug representation."""
        shape = (2, 3)
        tensor = Tensor(shape, Device("cuda:0"))

        s = tensor.debug_storage_repr()
        self.assertIsInstance(s, str)
        self.assertIn("CUDA Tensor on device", s)
        self.assertIn("0", s)
        self.assertIn(str(shape), s)

    def test_to_numpy_on_cuda_raises(self):
        tensor = Tensor((2, 3), Device("cuda:0"))
        with self.assertRaises(RuntimeError):
            tensor.to_numpy()

    def test_fill_on_cuda_raises(self):
        tensor = Tensor((2, 3), Device("cuda:0"))
        with self.assertRaises(RuntimeError):
            tensor.fill(1.0)

    def test_invalid_shape_negative_dimension_raises(self):
        """numpy.zeros rejects negative dimensions (current behavior)."""
        with self.assertRaises(ValueError):
            Tensor((2, -1), Device("cpu"))

    def test_invalid_shape_non_int_dimension_raises(self):
        """numpy.zeros rejects non-integer shape entries (current behavior)."""
        with self.assertRaises((TypeError, ValueError)):
            Tensor((2, 3.5), Device("cpu"))

    def test_unsupported_device_type_raises(self):
        """Non-Device should raise ValueError from initialization match-case."""
        with self.assertRaises(ValueError):
            Tensor((2, 3), object())


if __name__ == "__main__":
    unittest.main()
