import unittest
import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.domain.device._device import Device


class TestTensorDataAndDevPtr(unittest.TestCase):
    """
    Tests for Tensor.data, Tensor._from_devptr, and device-specific storage
    initialization semantics.

    These tests assert:
    - CPU tensors allocate NumPy storage on construction.
    - CUDA tensors default to a null DevPtr (0) on construction.
    - Tensor.data returns ndarray on CPU and int on CUDA.
    - Tensor._from_devptr constructs a CUDA tensor backed by a provided DevPtr.
    - __repr__ reflects CPU vs CUDA storage correctly.
    """

    def test_cpu_tensor_initializes_numpy_storage(self):
        """CPU Tensor should allocate a float32 NumPy array of zeros."""
        t = Tensor((2, 3), Device("cpu"))
        self.assertIsInstance(t.data, np.ndarray)
        self.assertEqual(t.data.dtype, np.float32)
        self.assertEqual(t.data.shape, (2, 3))
        self.assertTrue(np.all(t.data == 0))

    def test_cuda_tensor_initializes_null_devptr(self):
        """CUDA Tensor should not allocate device memory and should default to DevPtr(0)."""
        t = Tensor((2, 3), Device("cuda:0"))
        self.assertIsInstance(t.data, int)
        self.assertEqual(t.data, 0)

    def test_data_property_cpu_returns_ndarray_cuda_returns_int(self):
        """Tensor.data should return ndarray on CPU and int (DevPtr) on CUDA."""
        cpu_t = Tensor((1, 1), Device("cpu"))
        cuda_t = Tensor((1, 1), Device("cuda:0"))

        self.assertIsInstance(cpu_t.data, np.ndarray)
        self.assertIsInstance(cuda_t.data, int)

    def test_from_devptr_requires_cuda_device(self):
        """Tensor._from_devptr should reject non-CUDA devices."""
        with self.assertRaises(ValueError):
            Tensor._from_devptr(
                123,
                shape=(2, 3),
                device=Device("cpu"),
            )

    def test_from_devptr_rejects_none_devptr(self):
        """Tensor._from_devptr should reject dev_ptr=None."""
        with self.assertRaises(ValueError):
            Tensor._from_devptr(
                None,  # type: ignore[arg-type]
                shape=(2, 3),
                device=Device("cuda:0"),
            )

    def test_from_devptr_sets_shape_device_and_data(self):
        """Tensor._from_devptr should store provided devptr, shape, and device."""
        dev_ptr = 12345678
        device = Device("cuda:0")
        t = Tensor._from_devptr(
            dev_ptr,
            shape=(4, 5, 6),
            device=device,
            requires_grad=False,
            ctx=None,
        )

        self.assertEqual(t.shape, (4, 5, 6))
        self.assertEqual(t.device, device)
        self.assertIsInstance(t.data, int)
        self.assertEqual(t.data, dev_ptr)

    def test_from_devptr_preserves_autograd_fields(self):
        """Tensor._from_devptr should preserve requires_grad and ctx fields."""
        dev_ptr = 777
        device = Device("cuda:0")
        ctx_obj = object()

        t = Tensor._from_devptr(
            dev_ptr,
            shape=(1,),
            device=device,
            requires_grad=True,
            ctx=ctx_obj,  # Context is optional; for this test we just use a sentinel.
        )

        # These are internal fields in your snippet; we verify via getattr to avoid
        # coupling to any additional API you may have.
        self.assertTrue(getattr(t, "_requires_grad"))
        self.assertIs(getattr(t, "_ctx"), ctx_obj)
        self.assertIsNone(getattr(t, "_grad"))

    def test_repr_cpu_includes_dtype(self):
        """CPU __repr__ should include dtype information."""
        t = Tensor((2, 3), Device("cpu"))
        s = repr(t)
        self.assertIn("Tensor(shape=", s)
        self.assertIn("device=", s)
        self.assertIn("dtype=", s)
        self.assertIn("float32", s)

    def test_repr_cuda_includes_devptr(self):
        """CUDA __repr__ should include DevPtr(<int>)."""
        t0 = Tensor((2, 3), Device("cuda:0"))
        s0 = repr(t0)
        self.assertIn("Tensor(shape=", s0)
        self.assertIn("device=", s0)
        self.assertIn("DevPtr(", s0)
        self.assertIn("DevPtr(0)", s0)

        t1 = Tensor._from_devptr(999, shape=(2, 3), device=Device("cuda:0"))
        s1 = repr(t1)
        self.assertIn("DevPtr(999)", s1)


if __name__ == "__main__":
    unittest.main()
