# tests/infrastructure/tensors/test_tensor_reshape_cuda.py
from __future__ import annotations

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


class _CudaTestMixin:
    """
    CUDA helpers for tests.

    We stage host<->device copies using the native maxpool2d_ctypes wrapper
    (known-good on Windows). Do NOT use ops/memcpy_cuda here to avoid ctypes
    signature / bound-method issues.
    """

    def _skip_if_no_cuda(self) -> None:
        try:
            _ = Tensor._get_cuda_lib()
        except Exception as e:
            self.skipTest(f"CUDA not available (failed to load native CUDA lib): {e!r}")

    def _cuda_tensor_from_numpy(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        arr = np.ascontiguousarray(arr)
        dtype = np.dtype(arr.dtype)

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()
        dev = Device("cuda:0")

        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, int(getattr(dev, "index", 0) or 0))

        nbytes = int(arr.nbytes)
        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        # IMPORTANT: use the compatibility alias; do not pass raw numpy scalars
        # Signature: cudaMemcpyHtoD(lib, dst_dev, src_host, nbytes)
        m.cudaMemcpyHtoD(lib, int(dev_ptr), arr, nbytes)

        t = Tensor._from_devptr(
            dev_ptr=int(dev_ptr),
            shape=arr.shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )
        return t

    def _cuda_to_numpy(self, t: Tensor) -> np.ndarray:
        """
        Copy device tensor to host using maxpool2d_ctypes wrappers.
        """
        if not t.device.is_cuda():
            raise TypeError("expected CUDA tensor")

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()
        dev = t.device
        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, int(getattr(dev, "index", 0) or 0))

        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        nbytes = int(out.nbytes)
        src_dev = int(t.data)

        # Signature: cudaMemcpyDtoH(lib, dst_host, src_dev, nbytes)
        m.cudaMemcpyDtoH(lib, out, src_dev, nbytes)
        return out


class TestTensorReshapeCudaForward(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_reshape_cuda_forward_matches_numpy(self):
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.reshape((6, 4))
        self.assertEqual(y.shape, (6, 4))

        got = self._cuda_to_numpy(y)
        expected = x_np.reshape(6, 4)
        self.assertTrue(np.allclose(got, expected, rtol=1e-5, atol=1e-6))

    def test_reshape_cuda_supports_infer_minus_one(self):
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.reshape((-1, 4))
        self.assertEqual(y.shape, (6, 4))

        got = self._cuda_to_numpy(y)
        expected = x_np.reshape(-1, 4)
        self.assertTrue(np.allclose(got, expected, rtol=1e-5, atol=1e-6))

    def test_reshape_cuda_invalid_raises(self):
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = x.reshape((5, 5))  # numel mismatch

    def test_reshape_cuda_requires_allocated_input(self):
        # raw CUDA tensor without allocation should have data==0 (per your contract)
        x = Tensor((2, 3, 4), Device("cuda:0"), requires_grad=False, dtype=np.float32)
        self.assertEqual(int(x.data), 0)

        with self.assertRaises(RuntimeError):
            _ = x.reshape((6, 4))


class TestTensorReshapeCudaBackward(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_reshape_cuda_backward_populates_grad_and_matches_ones(self):
        """
        y = reshape(x), loss = sum(y)
        grad_out is ones in y-shape => grad_x should be ones in x-shape.
        """
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.reshape((6, 4))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        got = self._cuda_to_numpy(x.grad)
        expected = np.ones_like(x_np, dtype=np.float32)
        self.assertTrue(np.allclose(got, expected, rtol=1e-5, atol=1e-6))

    def test_reshape_cuda_backward_chain_multiple_reshapes(self):
        """
        x -> reshape -> reshape -> sum -> backward
        should still give grad=ones in x-shape.
        """
        x_np = np.random.randn(2, 2, 3).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.reshape((4, 3)).reshape((12,))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        got = self._cuda_to_numpy(x.grad)
        expected = np.ones_like(x_np, dtype=np.float32)
        self.assertTrue(np.allclose(got, expected, rtol=1e-5, atol=1e-6))

    def test_reshape_cuda_backward_wrong_device_grad_out_raises(self):
        """
        If your reshape backward checks for same-device grad_out,
        this ensures it fails fast when grad_out is CPU.
        """
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.reshape((6, 4))

        grad_out_cpu = Tensor(
            y.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        grad_out_cpu.copy_from_numpy(np.ones(y.shape, dtype=np.float32))

        with self.assertRaises((ValueError, RuntimeError)):
            y.backward(grad_out_cpu)


if __name__ == "__main__":
    unittest.main()
