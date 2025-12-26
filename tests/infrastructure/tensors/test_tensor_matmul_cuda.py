# tests/infrastructure/tensors/test_tensor_matmul_cuda.py
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
    probe failures in ops/matmul_cuda._probe_dev_range.
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

        # Use the same native module used elsewhere in Tensor CUDA code
        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()

        # Ensure correct device is selected (index parsed from Device("cuda:0") => 0)
        dev = Device("cuda:0")
        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, int(getattr(dev, "index", 0) or 0))

        nbytes = int(arr.nbytes)
        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        # Copy host -> device using known-good wrapper
        m.cudaMemcpyHtoD(lib, int(dev_ptr), arr, nbytes)

        # Wrap as Tensor backed by that devptr
        t = Tensor._from_devptr(
            dev_ptr=dev_ptr,
            shape=arr.shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )
        return t

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


class _CudaTensorFactoryMixin:
    def _cuda_tensor_from_numpy(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        """
        Create a CUDA tensor initialized from a NumPy array.

        Notes
        -----
        - copy_from_numpy() is CPU-only in this codebase.
        - We therefore stage through CPU, then transfer to CUDA using whatever
          API the Tensor implementation exposes (to/cuda/from_numpy).
        """
        arr = np.asarray(arr, dtype=np.float32)

        # Stage on CPU first
        t_cpu = Tensor(
            arr.shape, Device("cpu"), requires_grad=requires_grad, dtype=np.float32
        )
        t_cpu.copy_from_numpy(arr)

        # Transfer to CUDA using the available API
        if hasattr(t_cpu, "to"):
            t_cuda = t_cpu.to(Device("cuda:0"))
            return t_cuda

        if hasattr(t_cpu, "cuda"):
            t_cuda = t_cpu.cuda()  # type: ignore[attr-defined]
            return t_cuda

        if hasattr(Tensor, "from_numpy"):
            # Common alternative API
            return Tensor.from_numpy(arr, device=Device("cuda:0"), requires_grad=requires_grad)  # type: ignore[attr-defined]

        raise RuntimeError(
            "No supported CPU->CUDA transfer API found. "
            "Expected Tensor.to(Device('cuda:0')) or Tensor.cuda() or Tensor.from_numpy(..., device=...)."
        )


class TestTensorMatmulCudaForward(TestCase, _CudaTestMixin, _CudaTensorFactoryMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_matmul_cuda_forward_matches_numpy_float32(self):
        A_np = np.random.randn(8, 16).astype(np.float32)
        B_np = np.random.randn(16, 5).astype(np.float32)

        A = self._cuda_tensor_from_numpy(A_np, requires_grad=False)
        B = self._cuda_tensor_from_numpy(B_np, requires_grad=False)

        Y = A @ B
        expected = A_np @ B_np

        self.assertEqual(Y.shape, (8, 5))
        self.assertTrue(np.allclose(Y.to_numpy(), expected, rtol=1e-4, atol=1e-5))

    def test_matmul_cuda_rejects_non_2d(self):
        A = self._cuda_alloc_empty((2, 3, 4), dtype=np.float32, requires_grad=False)
        B = self._cuda_alloc_empty((4, 5), dtype=np.float32, requires_grad=False)
        with self.assertRaises(ValueError):
            _ = A @ B

    def test_matmul_cuda_shape_mismatch_raises(self):
        A = self._cuda_alloc_empty((3, 4), dtype=np.float32, requires_grad=False)
        B = self._cuda_alloc_empty((5, 2), dtype=np.float32, requires_grad=False)
        with self.assertRaises(ValueError):
            _ = A @ B

    def test_matmul_cuda_mixed_device_raises(self):
        A = self._cuda_alloc_empty((3, 4), dtype=np.float32, requires_grad=False)

        B = Tensor((4, 2), Device("cpu"), requires_grad=False, dtype=np.float32)
        B.copy_from_numpy(np.random.randn(4, 2).astype(np.float32))

        with self.assertRaises(Exception):
            _ = A @ B

    # def test_matmul_cuda_requires_allocated_inputs(self):
    #     # Intentionally create CUDA tensors without allocating device memory.
    #     A = Tensor((3, 4), Device("cuda:0"), requires_grad=False, dtype=np.float32)
    #     B = Tensor((4, 2), Device("cuda:0"), requires_grad=False, dtype=np.float32)

    #     self.assertEqual(int(A.data), 0)
    #     self.assertEqual(int(B.data), 0)

    #     with self.assertRaises(RuntimeError):
    #         _ = A @ B

    def test_matmul_cuda_requires_allocated_inputs(self):
        # Intentionally create CUDA tensors without allocating device memory.
        A = Tensor((3, 4), Device("cuda:0"), requires_grad=False, dtype=np.float32)
        B = Tensor((4, 2), Device("cuda:0"), requires_grad=False, dtype=np.float32)

        # Contract: raw CUDA Tensor(...) should not allocate unless explicitly requested.
        self.assertEqual(int(A.data), 0)
        self.assertEqual(int(B.data), 0)

        # CUDA matmul requires allocated device buffers for both inputs.
        with self.assertRaises(RuntimeError):
            _ = A @ B


class TestTensorMatmulCudaBackward(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_matmul_cuda_backward_grads_match_closed_form(self):
        """
        For Y = A @ B, loss = sum(Y),
        grad_out = ones(N, M)
        dA = grad_out @ B^T
        dB = A^T @ grad_out
        """
        A_np = np.random.randn(6, 7).astype(np.float32)
        B_np = np.random.randn(7, 4).astype(np.float32)

        A = self._cuda_tensor_from_numpy(A_np, requires_grad=True)
        B = self._cuda_tensor_from_numpy(B_np, requires_grad=True)

        Y = A @ B
        loss = Y.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)

        grad_out = np.ones((6, 4), dtype=np.float32)
        expected_dA = grad_out @ B_np.T
        expected_dB = A_np.T @ grad_out

        self.assertEqual(A.grad.shape, A.shape)
        self.assertEqual(B.grad.shape, B.shape)

        self.assertTrue(
            np.allclose(A.grad.to_numpy(), expected_dA, rtol=1e-4, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(B.grad.to_numpy(), expected_dB, rtol=1e-4, atol=1e-5)
        )

    def test_matmul_cuda_backward_only_left_requires_grad(self):
        A_np = np.random.randn(5, 3).astype(np.float32)
        B_np = np.random.randn(3, 2).astype(np.float32)

        A = self._cuda_tensor_from_numpy(A_np, requires_grad=True)
        B = self._cuda_tensor_from_numpy(B_np, requires_grad=False)

        Y = A @ B
        loss = Y.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNone(B.grad)

        grad_out = np.ones((5, 2), dtype=np.float32)
        expected_dA = grad_out @ B_np.T
        self.assertTrue(
            np.allclose(A.grad.to_numpy(), expected_dA, rtol=1e-4, atol=1e-5)
        )

    def test_matmul_cuda_backward_only_right_requires_grad(self):
        A_np = np.random.randn(2, 6).astype(np.float32)
        B_np = np.random.randn(6, 5).astype(np.float32)

        A = self._cuda_tensor_from_numpy(A_np, requires_grad=False)
        B = self._cuda_tensor_from_numpy(B_np, requires_grad=True)

        Y = A @ B
        loss = Y.sum()
        loss.backward()

        self.assertIsNone(A.grad)
        self.assertIsNotNone(B.grad)

        grad_out = np.ones((2, 5), dtype=np.float32)
        expected_dB = A_np.T @ grad_out
        self.assertTrue(
            np.allclose(B.grad.to_numpy(), expected_dB, rtol=1e-4, atol=1e-5)
        )

    def test_matmul_cuda_backward_grad_out_wrong_device_raises(self):
        A_np = np.random.randn(3, 4).astype(np.float32)
        B_np = np.random.randn(4, 2).astype(np.float32)

        A = self._cuda_tensor_from_numpy(A_np, requires_grad=True)
        B = self._cuda_tensor_from_numpy(B_np, requires_grad=True)

        Y = A @ B  # CUDA forward

        # Wrong-device grad_out (CPU). Tensor.backward should reject this immediately.
        grad_out_cpu = Tensor(
            Y.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        grad_out_cpu.copy_from_numpy(np.ones(Y.shape, dtype=np.float32))

        with self.assertRaises(ValueError) as cm:
            Y.backward(grad_out_cpu)

        self.assertIn("same device", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
