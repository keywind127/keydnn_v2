# tests/infrastructure/tensors/test_tensor_full_cuda.py
from __future__ import annotations

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


class _CudaTestMixin:
    """
    CUDA helpers for tests.

    We rely on Tensor._get_cuda_lib() to decide CUDA availability, and
    Tensor.to_numpy() for device->host validation (so we don't depend on
    low-level ctypes memcpy wrappers here).
    """

    def _skip_if_no_cuda(self) -> None:
        try:
            _ = Tensor._get_cuda_lib()
        except Exception as e:
            self.skipTest(f"CUDA not available (failed to load native CUDA lib): {e!r}")


class TestTensorFullCuda(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_full_cuda_fills_values_float32(self) -> None:
        dev = Device("cuda:0")
        fill_value = 3.25

        t = Tensor.full((4, 7), fill_value, device=dev, requires_grad=False)

        self.assertTrue(t.device.is_cuda())
        self.assertEqual(t.shape, (4, 7))

        # Contract: CUDA full should allocate (fill() ensures allocation)
        self.assertNotEqual(int(t.data), 0)

        got = t.to_numpy()
        self.assertIsInstance(got, np.ndarray)
        self.assertEqual(got.shape, (4, 7))
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.allclose(got, np.full((4, 7), fill_value, dtype=np.float32)))

    def test_full_cuda_scalar_shape(self) -> None:
        dev = Device("cuda:0")
        fill_value = -1.5

        t = Tensor.full((), fill_value, device=dev, requires_grad=False)

        self.assertTrue(t.device.is_cuda())
        self.assertEqual(t.shape, ())
        self.assertNotEqual(int(t.data), 0)

        got = t.to_numpy()
        self.assertEqual(got.shape, ())
        self.assertEqual(got.dtype, np.float32)
        # scalar ndarray comparison
        self.assertAlmostEqual(float(got), float(np.float32(fill_value)), places=6)

    def test_full_cuda_requires_grad_flag_preserved(self) -> None:
        dev = Device("cuda:0")
        t = Tensor.full((2, 2), 1.0, device=dev, requires_grad=True)

        self.assertTrue(t.device.is_cuda())
        self.assertTrue(bool(getattr(t, "requires_grad", False)))

        # Should still be filled correctly
        got = t.to_numpy()
        self.assertTrue(np.allclose(got, np.ones((2, 2), dtype=np.float32)))

    def test_full_cuda_zero_numel_tensor(self) -> None:
        """
        Edge case: tensors with a zero dimension.
        This should not crash; to_numpy should round-trip to the same shape.
        """
        dev = Device("cuda:0")
        t = Tensor.full((0, 5), 7.0, device=dev, requires_grad=False)

        self.assertTrue(t.device.is_cuda())
        self.assertEqual(t.shape, (0, 5))

        got = t.to_numpy()
        self.assertEqual(got.shape, (0, 5))
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got.size, 0)

    def test_full_cuda_multiple_calls_different_values(self) -> None:
        dev = Device("cuda:0")

        t1 = Tensor.full((3, 3), 0.0, device=dev, requires_grad=False)
        t2 = Tensor.full((3, 3), 2.0, device=dev, requires_grad=False)

        self.assertTrue(np.allclose(t1.to_numpy(), np.zeros((3, 3), dtype=np.float32)))
        self.assertTrue(
            np.allclose(t2.to_numpy(), np.full((3, 3), 2.0, dtype=np.float32))
        )


if __name__ == "__main__":
    unittest.main()
