# tests/infrastructure/tensors/test_tensor_copy_from_cuda.py
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
    We intentionally use memcpy_cuda_ctypes (GetProcAddress + CFUNCTYPE) for
    H2D/D2H/D2D to avoid ctypes argtypes collisions across modules.

    maxpool2d_ctypes binds keydnn_cuda_memcpy_* with one signature (c_uint64),
    while ops/memcpy_cuda binds the same symbols with c_void_p. Mixing them in
    the same process leads to ctypes.ArgumentError "wrong type".
    """

    def _skip_if_no_cuda(self) -> None:
        try:
            _ = Tensor._get_cuda_lib()
        except Exception as e:
            self.skipTest(f"CUDA not available (failed to load native CUDA lib): {e!r}")

    @staticmethod
    def _dev_index(dev: Device) -> int:
        return int(getattr(dev, "index", 0) or 0)

    def _cuda_tensor_from_numpy(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        arr = np.ascontiguousarray(arr)
        dtype = np.dtype(arr.dtype)

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m
        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,
        )

        lib = Tensor._get_cuda_lib()
        dev = Device("cuda:0")

        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, self._dev_index(dev))

        nbytes = int(arr.nbytes)
        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        # Copy host -> device using CFUNCTYPE wrapper (no argtypes collision).
        mc.cuda_memcpy_h2d(lib, int(dev_ptr), arr, nbytes)

        return Tensor._from_devptr(
            dev_ptr=int(dev_ptr),
            shape=arr.shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )

    def _cuda_alloc_empty(
        self, shape: tuple[int, ...], *, dtype: np.dtype, requires_grad: bool
    ) -> Tensor:
        dtype = np.dtype(dtype)
        numel = int(np.prod(shape)) if len(shape) > 0 else 1
        nbytes = int(numel) * int(dtype.itemsize)

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()
        dev = Device("cuda:0")

        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, self._dev_index(dev))

        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        return Tensor._from_devptr(
            dev_ptr=int(dev_ptr),
            shape=shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )

    def _cuda_to_numpy(self, t: Tensor) -> np.ndarray:
        """
        Read back a CUDA tensor into a NumPy array using memcpy_cuda_ctypes.
        """
        if not t.device.is_cuda():
            raise TypeError(f"_cuda_to_numpy expects CUDA tensor, got {t.device}")

        dtype = np.dtype(t.dtype)
        out = np.empty(t.shape, dtype=dtype)

        nbytes = int(t.numel()) * int(dtype.itemsize)
        if nbytes == 0:
            return out

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m
        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,
        )

        lib = Tensor._get_cuda_lib()
        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, self._dev_index(t.device))

        src_dev = int(t.data)
        if src_dev == 0 and t.numel() != 0:
            raise RuntimeError("Attempted to read back CUDA tensor with data == 0")

        mc.cuda_memcpy_d2h(lib, out, src_dev, nbytes)
        return out


class TestTensorCopyFromCudaSameDevice(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_copy_from_cuda_dtod_copies_values_float32(self):
        src_np = np.random.randn(4, 7).astype(np.float32)

        src = self._cuda_tensor_from_numpy(src_np, requires_grad=False)
        dst = self._cuda_alloc_empty(
            src_np.shape, dtype=np.float32, requires_grad=False
        )

        dst.copy_from(src)

        got = self._cuda_to_numpy(dst)
        self.assertTrue(np.allclose(got, src_np, rtol=0.0, atol=0.0))

    def test_copy_from_cuda_allocates_destination_if_unallocated(self):
        src_np = np.random.randn(3, 5).astype(np.float32)
        src = self._cuda_tensor_from_numpy(src_np, requires_grad=False)

        # Raw CUDA Tensor(...) should not allocate unless explicitly requested.
        dst = Tensor(
            src_np.shape, Device("cuda:0"), requires_grad=False, dtype=np.float32
        )
        self.assertEqual(int(dst.data), 0)

        dst.copy_from(src)

        self.assertNotEqual(int(dst.data), 0)
        got = self._cuda_to_numpy(dst)
        self.assertTrue(np.allclose(got, src_np, rtol=0.0, atol=0.0))

    def test_copy_from_cuda_dtype_mismatch_raises(self):
        src_np = np.random.randn(2, 3).astype(np.float32)
        src = self._cuda_tensor_from_numpy(src_np, requires_grad=False)

        dst = self._cuda_alloc_empty(
            src_np.shape, dtype=np.float64, requires_grad=False
        )

        with self.assertRaises(TypeError):
            dst.copy_from(src)

    def test_copy_from_cuda_shape_mismatch_raises(self):
        src_np = np.random.randn(2, 3).astype(np.float32)
        src = self._cuda_tensor_from_numpy(src_np, requires_grad=False)

        dst = self._cuda_alloc_empty((2, 4), dtype=np.float32, requires_grad=False)
        with self.assertRaises(ValueError):
            dst.copy_from(src)


class TestTensorCopyFromCudaCrossDevice(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()

    def test_copy_from_cross_device_cuda_to_cpu_requires_flag(self):
        src_np = np.random.randn(
            6,
        ).astype(np.float32)
        src_cuda = self._cuda_tensor_from_numpy(src_np, requires_grad=False)

        dst_cpu = Tensor(
            src_np.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        dst_cpu.copy_from_numpy(np.zeros_like(src_np))

        with self.assertRaises(RuntimeError):
            dst_cpu.copy_from(src_cuda)  # allow_cross_device defaults to False

    def test_copy_from_cross_device_cpu_to_cuda_requires_flag(self):
        src_np = np.random.randn(2, 2).astype(np.float32)

        src_cpu = Tensor(
            src_np.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        src_cpu.copy_from_numpy(src_np)

        dst_cuda = Tensor(
            src_np.shape, Device("cuda:0"), requires_grad=False, dtype=np.float32
        )

        with self.assertRaises(RuntimeError):
            dst_cuda.copy_from(src_cpu)  # allow_cross_device defaults to False

    def test_copy_from_cross_device_cuda_to_cpu_when_enabled(self):
        src_np = np.random.randn(4, 3).astype(np.float32)
        src_cuda = self._cuda_tensor_from_numpy(src_np, requires_grad=False)

        dst_cpu = Tensor(
            src_np.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        dst_cpu.copy_from_numpy(np.zeros_like(src_np))

        dst_cpu.copy_from(src_cuda, allow_cross_device=True)

        got = dst_cpu.to_numpy()
        self.assertTrue(np.allclose(got, src_np, rtol=0.0, atol=0.0))

    def test_copy_from_cross_device_cpu_to_cuda_when_enabled(self):
        src_np = np.random.randn(3, 4).astype(np.float32)

        src_cpu = Tensor(
            src_np.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        src_cpu.copy_from_numpy(src_np)

        dst_cuda = Tensor(
            src_np.shape, Device("cuda:0"), requires_grad=False, dtype=np.float32
        )
        self.assertEqual(int(dst_cuda.data), 0)

        dst_cuda.copy_from(src_cpu, allow_cross_device=True)

        self.assertNotEqual(int(dst_cuda.data), 0)
        got = self._cuda_to_numpy(dst_cuda)
        self.assertTrue(np.allclose(got, src_np, rtol=0.0, atol=0.0))

    def test_copy_from_cross_device_dtype_mismatch_raises_even_when_enabled(self):
        src_np = np.random.randn(2, 3).astype(np.float32)

        src_cpu = Tensor(
            src_np.shape, Device("cpu"), requires_grad=False, dtype=np.float32
        )
        src_cpu.copy_from_numpy(src_np)

        dst_cuda = Tensor(
            src_np.shape, Device("cuda:0"), requires_grad=False, dtype=np.float64
        )

        with self.assertRaises(TypeError):
            dst_cuda.copy_from(src_cpu, allow_cross_device=True)


if __name__ == "__main__":
    unittest.main()
