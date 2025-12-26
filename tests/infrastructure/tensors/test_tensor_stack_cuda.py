# tests/infrastructure/tensors/test_tensor_stack_cuda.py
from __future__ import annotations

import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


class _CudaTestMixin:
    """
    CUDA helpers for tests.

    IMPORTANT:
    We deliberately stage host->device using native ctypes wrappers
    (maxpool2d_ctypes) instead of ops/memcpy_cuda, to avoid ctypes signature
    mismatch issues on Windows that can lead to invalid device pointers and
    probe failures in ops/... CUDA code paths.
    """

    def _skip_if_no_cuda(self) -> None:
        try:
            _ = Tensor._get_cuda_lib()
        except Exception as e:
            self.skipTest(f"CUDA not available (failed to load native CUDA lib): {e!r}")

    def _cuda_tensor_from_numpy(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        """
        Create a CUDA tensor initialized from a NumPy array by:
        - cuda_malloc
        - cudaMemcpyHtoD
        - Tensor._from_devptr(...)
        """
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

        m.cudaMemcpyHtoD(lib, int(dev_ptr), arr, nbytes)

        return Tensor._from_devptr(
            dev_ptr=dev_ptr,
            shape=arr.shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )

    def _cuda_alloc_empty(
        self, shape: tuple[int, ...], *, dtype: np.dtype, requires_grad: bool
    ) -> Tensor:
        """
        Allocate an uninitialized device buffer for a tensor of given shape/dtype.
        """
        dtype = np.dtype(dtype)
        numel = int(np.prod(shape)) if len(shape) > 0 else 1
        nbytes = int(numel) * int(dtype.itemsize)

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()

        dev = Device("cuda:0")
        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, int(getattr(dev, "index", 0) or 0))

        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        return Tensor._from_devptr(
            dev_ptr=dev_ptr,
            shape=shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )


class TestTensorStackCudaForward(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_stack_cuda_forward_axis0_matches_numpy(self):
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.zeros((2, 3), dtype=np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        s = Tensor.stack([a, b], axis=0)
        self.assertEqual(s.device.type, Device("cuda:0").type)
        self.assertEqual(s.shape, (2, 2, 3))

        expected = np.stack([a_np, b_np], axis=0)
        self.assertTrue(np.array_equal(s.to_numpy(), expected))

    def test_stack_cuda_forward_axis1_matches_numpy(self):
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = 2.0 * np.ones((2, 3), dtype=np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        s = Tensor.stack([a, b], axis=1)
        self.assertEqual(s.shape, (2, 2, 3))

        expected = np.stack([a_np, b_np], axis=1)
        self.assertTrue(np.array_equal(s.to_numpy(), expected))

    def test_stack_cuda_rejects_empty_list(self):
        with self.assertRaises(Exception):
            _ = Tensor.stack([], axis=0)

    def test_stack_cuda_rejects_mismatched_shapes(self):
        a = self._cuda_tensor_from_numpy(
            np.zeros((2, 3), dtype=np.float32), requires_grad=False
        )
        b = self._cuda_tensor_from_numpy(
            np.zeros((2, 4), dtype=np.float32), requires_grad=False
        )
        with self.assertRaises(Exception):
            _ = Tensor.stack([a, b], axis=0)

    def test_stack_cuda_mixed_device_raises(self):
        a = self._cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), requires_grad=False
        )
        b = Tensor((2, 3), Device("cpu"), requires_grad=False, dtype=np.float32)
        b.copy_from_numpy(np.zeros((2, 3), dtype=np.float32))

        with self.assertRaises(Exception):
            _ = Tensor.stack([a, b], axis=0)

    def test_stack_cuda_rejects_axis_out_of_range(self):
        a = self._cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), requires_grad=False
        )
        b = self._cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), requires_grad=False
        )
        with self.assertRaises(Exception):
            _ = Tensor.stack([a, b], axis=99)


class TestTensorStackCudaBackward(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_stack_cuda_backward_splits_grad_to_inputs_axis0(self):
        """
        For loss = stack([a,b], axis=0).sum(),
        grads to a and b should be all-ones with their original shapes.
        """
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=True)

        s = Tensor.stack([a, b], axis=0)
        loss = s.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

        ones = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.array_equal(a.grad.to_numpy(), ones))
        self.assertTrue(np.array_equal(b.grad.to_numpy(), ones))

    def test_stack_cuda_backward_splits_grad_to_inputs_axis1(self):
        """
        Same as axis0 test, but along axis=1.
        """
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=True)

        s = Tensor.stack([a, b], axis=1)
        loss = s.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

        ones = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.array_equal(a.grad.to_numpy(), ones))
        self.assertTrue(np.array_equal(b.grad.to_numpy(), ones))

    def test_stack_cuda_backward_only_first_requires_grad(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        s = Tensor.stack([a, b], axis=0)
        loss = s.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNone(b.grad)

        ones = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.array_equal(a.grad.to_numpy(), ones))

    def test_stack_cuda_backward_grad_out_wrong_device_raises(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=True)

        s = Tensor.stack([a, b], axis=0)  # CUDA tensor

        # Wrong-device grad_out (CPU). Tensor.backward should reject this.
        grad_out_cpu = Tensor(
            s.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        grad_out_cpu.copy_from_numpy(np.ones(s.shape, dtype=np.float32))

        with self.assertRaises(ValueError) as cm:
            s.backward(grad_out_cpu)

        self.assertIn("same device", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
