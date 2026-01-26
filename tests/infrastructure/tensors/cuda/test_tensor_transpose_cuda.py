from __future__ import annotations

from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class _CudaTensorFactoryMixin:
    """
    Test helper mixin for creating CUDA tensors from NumPy arrays.
    """

    @classmethod
    def _try_get_cuda_lib(cls):

        try:
            lib = Tensor._get_cuda_lib()
            return lib
        except Exception:
            return None

    def _cuda_tensor_from_numpy(
        self,
        arr: np.ndarray,
        *,
        requires_grad: bool,
        device_str: str = "cuda:0",
    ) -> Tensor:
        arr = np.asarray(arr, dtype=np.float32, order="C")

        dev = Device(device_str)
        t = Tensor(arr.shape, dev, requires_grad=requires_grad, dtype=np.float32)

        t._ensure_cuda_alloc(dtype=np.float32)
        self.assertNotEqual(
            int(t.data), 0, "CUDA tensor devptr should be non-zero after alloc"
        )

        from src.keydnn.infrastructure.ops.memcpy_cuda import memcpy_htod

        lib = Tensor._get_cuda_lib()
        memcpy_htod(
            lib, dst_dev=int(t.data), src_host=arr, nbytes=int(arr.nbytes), sync=True
        )
        return t


class TestTensorTransposeCudaForwardBackward(TestCase, _CudaTensorFactoryMixin):
    @classmethod
    def setUpClass(cls) -> None:
        cls._cuda_lib = cls._try_get_cuda_lib()
        cls._cuda_available = cls._cuda_lib is not None

    def setUp(self) -> None:
        if not self._cuda_available:
            self.skipTest(
                "CUDA native library not available; skipping CUDA transpose tests."
            )

    def test_transpose_cuda_forward_matches_numpy(self):
        x_np = np.random.randn(2, 5).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.T
        expected = x_np.T

        self.assertTrue(y.device.is_cuda(), "transpose output should be CUDA tensor")
        self.assertEqual(y.shape, (5, 2))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_transpose_cuda_rejects_non_2d(self):

        x = Tensor((2, 3, 4), Device("cuda:0"), requires_grad=False, dtype=np.float32)

        with self.assertRaises(ValueError):
            _ = x.T

    def test_transpose_cuda_backward_routes_grad(self):
        """
        For y = x.T, loss = sum(y),
        grad_y should be ones(shape(y)), so grad_x = grad_y.T = ones(shape(x)).
        """
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.T
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.device.is_cuda(), "x.grad should be CUDA tensor")
        self.assertEqual(x.grad.shape, x.shape)

        expected = np.ones_like(x_np, dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_transpose_cuda_requires_allocated_input_devptr(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False, dtype=np.float32)

        self.assertEqual(int(x.data), 0)

        with self.assertRaises(RuntimeError):
            _ = x.T


if __name__ == "__main__":
    unittest.main()
