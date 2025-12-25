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


class TestTensorCopyFrom(TestCase, _TensorFactoryMixin):
    def test_copy_from_copies_values_cpu(self):
        src_np = np.random.randn(2, 3).astype(np.float32)
        src = self._tensor_from_numpy(src_np, requires_grad=False)

        dst = Tensor((2, 3), Device("cpu"), requires_grad=False)
        dst.fill(0.0)

        dst.copy_from(src)

        self.assertEqual(dst.shape, src.shape)
        self.assertTrue(np.array_equal(dst.to_numpy(), src_np))

        # Ensure it's a copy (mutating src should not change dst)
        src2_np = np.random.randn(2, 3).astype(np.float32)
        src.copy_from_numpy(src2_np)
        self.assertTrue(np.array_equal(dst.to_numpy(), src_np))
        self.assertTrue(np.array_equal(src.to_numpy(), src2_np))

    def test_copy_from_shape_mismatch_raises(self):
        src = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=False)
        dst = Tensor((2, 4), Device("cpu"), requires_grad=False)

        with self.assertRaises(ValueError):
            dst.copy_from(src)

    def test_copy_from_device_mismatch_raises(self):
        src = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=False)
        dst = Tensor((2, 3), Device("cuda:0"), requires_grad=False)

        with self.assertRaises(Exception):
            dst.copy_from(src)

    def test_copy_from_rejects_non_tensor(self):
        dst = Tensor((2, 3), Device("cpu"), requires_grad=False)
        with self.assertRaises(TypeError):
            dst.copy_from("not a tensor")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
