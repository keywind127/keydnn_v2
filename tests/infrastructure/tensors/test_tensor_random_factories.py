from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class TestTensorFull(TestCase):
    def setUp(self) -> None:
        self.cpu = Device("cpu")
        self.cuda = Device("cuda:0")

    def test_full_forward_matches_numpy(self):
        t = Tensor.full((2, 3), 1.25, device=self.cpu, requires_grad=False)
        self.assertEqual(t.shape, (2, 3))

        arr = t.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.allclose(arr, np.full((2, 3), 1.25, dtype=np.float32)))

    def test_full_scalar_tensor(self):
        t = Tensor.full((), 3.14, device=self.cpu, requires_grad=False)
        self.assertEqual(t.shape, ())

        v = float(np.asarray(t.to_numpy()))
        self.assertAlmostEqual(v, 3.14, places=6)

    def test_full_requires_grad_flag(self):
        t1 = Tensor.full((4,), 2.0, device=self.cpu, requires_grad=False)
        self.assertFalse(t1.requires_grad)

        t2 = Tensor.full((4,), 2.0, device=self.cpu, requires_grad=True)
        self.assertTrue(t2.requires_grad)

    def test_full_on_cuda_raises(self):
        with self.assertRaises(RuntimeError):
            _ = Tensor.full((2, 3), 1.0, device=self.cuda, requires_grad=False)


class TestTensorRand(TestCase):
    def setUp(self) -> None:
        self.cpu = Device("cpu")
        self.cuda = Device("cuda:0")

    def test_rand_shape_dtype_range(self):
        t = Tensor.rand((5, 7), device=self.cpu, requires_grad=False)
        self.assertEqual(t.shape, (5, 7))

        arr = t.to_numpy()
        self.assertEqual(arr.dtype, np.float32)

        # Typical contract: Uniform[0,1)
        self.assertTrue(np.all(arr >= 0.0))
        self.assertTrue(np.all(arr < 1.0))

    def test_rand_is_not_all_equal_most_of_the_time(self):
        # Weak statistical sanity check: random tensor should have some variation.
        t = Tensor.rand((50,), device=self.cpu, requires_grad=False)
        arr = t.to_numpy()
        self.assertGreater(np.std(arr), 0.0)

    def test_rand_requires_grad_flag(self):
        t1 = Tensor.rand((3, 3), device=self.cpu, requires_grad=False)
        self.assertFalse(t1.requires_grad)

        t2 = Tensor.rand((3, 3), device=self.cpu, requires_grad=True)
        self.assertTrue(t2.requires_grad)

    def test_rand_on_cuda_raises(self):
        with self.assertRaises(Exception):
            _ = Tensor.rand((2, 3), device=self.cuda, requires_grad=False)


if __name__ == "__main__":
    unittest.main()
