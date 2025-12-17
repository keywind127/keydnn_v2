from unittest import TestCase
import unittest

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.domain._device import Device


class TestTensorInfrastructure(TestCase):

    def test_tensor_initialization(self):
        shape = (2, 3)
        device_cpu = Device("cpu")
        device_cuda = Device("cuda:0")

        tensor_cpu = Tensor(shape, device_cpu)
        self.assertEqual(tensor_cpu.shape, shape)
        self.assertEqual(str(tensor_cpu.device), str(device_cpu))

        tensor_cuda = Tensor(shape, device_cuda)
        self.assertEqual(tensor_cuda.shape, shape)
        self.assertEqual(str(tensor_cuda.device), str(device_cuda))

    def test_tensor_operations(self):
        shape = (2, 2)
        device = Device("cpu")

        tensor = Tensor(shape, device)

        # Example operation: fill tensor with ones
        if tensor.device.is_cpu():
            tensor._data.fill(1)

        expected_data = [[1.0, 1.0], [1.0, 1.0]]
        self.assertTrue((tensor._data == expected_data).all())


if __name__ == "__main__":
    unittest.main()
