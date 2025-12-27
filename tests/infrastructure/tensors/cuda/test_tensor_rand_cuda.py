# tests/infrastructure/tensors/cuda/test_tensor_rand_cuda.py
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
    - Use maxpool2d_ctypes for malloc/free/memset/set_device (stable).
    - Use memcpy_ctypes (CFUNCTYPE) for H2D/D2H to avoid CDLL.argtypes collisions
      with other binders (e.g., ops/memcpy_cuda.py).
    """

    def _skip_if_no_cuda(self) -> None:
        try:
            _ = Tensor._get_cuda_lib()
        except Exception as e:
            self.skipTest(f"CUDA not available (failed to load native CUDA lib): {e!r}")

    @staticmethod
    def _numel(shape: tuple[int, ...]) -> int:
        if len(shape) == 0:
            return 1
        n = 1
        for d in shape:
            n *= int(d)
        return int(n)

    def _cuda_set_device0(self, lib) -> None:
        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        if hasattr(m, "cuda_set_device"):
            m.cuda_set_device(lib, 0)

    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool) -> Tensor:
        """
        Create a CUDA tensor initialized from a NumPy array, using:
        - cuda_malloc from maxpool2d_ctypes
        - cuda_memcpy_h2d from memcpy_ctypes (CFUNCTYPE)
        """
        arr = np.ascontiguousarray(arr)
        dtype = np.dtype(arr.dtype)

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m
        from src.keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc

        lib = Tensor._get_cuda_lib()
        dev = Device("cuda:0")

        self._cuda_set_device0(lib)

        nbytes = int(arr.nbytes)
        dev_ptr = int(m.cuda_malloc(lib, nbytes))
        if dev_ptr == 0:
            raise RuntimeError("cuda_malloc returned nullptr")

        # CFUNCTYPE wrapper: mc.cuda_memcpy_h2d(lib, dst_dev, src_host, nbytes)
        mc.cuda_memcpy_h2d(lib, int(dev_ptr), arr, int(nbytes))

        return Tensor._from_devptr(
            dev_ptr=int(dev_ptr),
            shape=arr.shape,
            device=dev,
            requires_grad=requires_grad,
            ctx=None,
            dtype=dtype,
        )

    def _cuda_to_numpy(self, t: Tensor) -> np.ndarray:
        """
        Copy CUDA tensor -> NumPy using memcpy_ctypes to avoid CDLL.argtypes collisions.
        """
        if not t.device.is_cuda():
            raise TypeError("expected CUDA tensor")

        dtype = np.dtype(t.dtype)
        out = np.empty(t.shape, dtype=dtype)

        if t.numel() == 0:
            return out

        src_dev = int(t.data)
        if src_dev == 0:
            raise RuntimeError("CUDA tensor has no allocated devptr")

        from src.keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc

        lib = Tensor._get_cuda_lib()
        self._cuda_set_device0(lib)

        nbytes = int(out.nbytes)
        # CFUNCTYPE wrapper: mc.cuda_memcpy_d2h(lib, dst_host, src_dev, nbytes)
        mc.cuda_memcpy_d2h(lib, out, int(src_dev), int(nbytes))
        return out

    def _fill_cuda_bytes(self, t: Tensor, value_byte: int) -> None:
        """
        Byte-fill a CUDA tensor's device buffer (useful to detect accidental mutation).
        """
        if not t.device.is_cuda():
            raise TypeError("expected CUDA tensor")
        if t.numel() == 0:
            return

        dev_ptr = int(t.data)
        if dev_ptr == 0:
            return

        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = Tensor._get_cuda_lib()
        self._cuda_set_device0(lib)

        nbytes = int(t.numel()) * int(np.dtype(t.dtype).itemsize)
        m.cuda_memset(lib, int(dev_ptr), int(value_byte), int(nbytes))


class TestTensorRandCuda(TestCase, _CudaTestMixin):
    def setUp(self) -> None:
        self._skip_if_no_cuda()
        np.random.seed(1234)

    def test_rand_cuda_returns_cuda_tensor_with_correct_shape_dtype_range(self):
        dev = Device("cuda:0")
        shape = (32, 17)

        t = Tensor.rand(shape, device=dev, requires_grad=False)

        self.assertTrue(t.device.is_cuda())
        self.assertEqual(t.shape, shape)
        self.assertEqual(np.dtype(t.dtype), np.dtype(np.float32))

        got = self._cuda_to_numpy(t)
        self.assertEqual(got.shape, shape)
        self.assertEqual(got.dtype, np.float32)

        self.assertTrue(np.all(got >= 0.0))
        self.assertTrue(np.all(got < 1.0 + 1e-7))

        # should not be constant
        self.assertGreater(float(np.std(got)), 0.0)

    def test_rand_cuda_allocates_device_memory(self):
        dev = Device("cuda:0")
        t = Tensor.rand((8, 9), device=dev, requires_grad=False)

        # Contract: CUDA rand should result in an allocated devptr
        self.assertNotEqual(int(t.data), 0)

        got = self._cuda_to_numpy(t)
        self.assertEqual(got.shape, (8, 9))

    def test_rand_cuda_requires_grad_flag_respected(self):
        dev = Device("cuda:0")
        t = Tensor.rand((4, 5), device=dev, requires_grad=True)
        self.assertTrue(bool(getattr(t, "requires_grad", False)))

    def test_rand_cuda_zero_numel_ok(self):
        dev = Device("cuda:0")
        t = Tensor.rand((0, 10), device=dev, requires_grad=False)

        self.assertTrue(t.device.is_cuda())
        self.assertEqual(t.shape, (0, 10))

        got = self._cuda_to_numpy(t)
        self.assertEqual(got.shape, (0, 10))

    def test_rand_cuda_different_calls_produce_different_values_most_of_time(self):
        dev = Device("cuda:0")
        a = Tensor.rand((16, 16), device=dev, requires_grad=False)
        b = Tensor.rand((16, 16), device=dev, requires_grad=False)

        a_np = self._cuda_to_numpy(a)
        b_np = self._cuda_to_numpy(b)

        self.assertFalse(np.array_equal(a_np, b_np))

    def test_rand_cuda_does_not_mutate_existing_tensor(self):
        """
        Regression: Tensor.rand must not reuse/overwrite an existing buffer.
        """
        dev = Device("cuda:0")

        # allocate a CUDA tensor and fill bytes with a known pattern
        t0 = Tensor((32,), dev, requires_grad=False, dtype=np.float32)
        if int(t0.data) == 0:
            t0._ensure_cuda_alloc(dtype=np.dtype(np.float32))

        self._fill_cuda_bytes(t0, 0x7F)
        before = self._cuda_to_numpy(t0).copy()

        # call rand (should not affect t0)
        _ = Tensor.rand((32,), device=dev, requires_grad=False)

        after = self._cuda_to_numpy(t0)
        self.assertTrue(np.array_equal(before, after))


if __name__ == "__main__":
    unittest.main()
