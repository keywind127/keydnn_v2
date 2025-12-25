import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_d2h,
    cuda_memset,
    cuda_synchronize,
)

from src.keydnn.infrastructure.native_cuda.python.fill_ctypes import (
    cuda_fill,
)


class TestCudaFillCtypes(unittest.TestCase):
    """
    Unit tests for CUDA fill ctypes wrapper (fill_ctypes.py).

    These tests validate:
    - cuda_fill writes the requested scalar value for float32/float64
    - cuda_fill rejects unsupported dtypes
    - numel=0 is accepted when y_dev is a valid device pointer (no crash)
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = load_keydnn_cuda_native()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        try:
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA device unavailable / cannot set device: {e}")

    def _run_fill_case(self, dtype: np.dtype, value: float, numel: int) -> None:
        if dtype not in (np.float32, np.float64):
            raise ValueError("dtype must be float32/float64 in _run_fill_case")

        host = np.empty((numel,), dtype=dtype)
        nbytes = int(host.nbytes)

        y_dev = cuda_malloc(self.lib, nbytes)
        try:
            cuda_memset(self.lib, y_dev, 0xA5, nbytes)

            cuda_fill(self.lib, y_dev=y_dev, numel=numel, value=value, dtype=dtype)
            cuda_synchronize(self.lib)

            cuda_memcpy_d2h(self.lib, host, y_dev)
        finally:
            cuda_free(self.lib, y_dev)

        expected = np.full((numel,), value, dtype=dtype)
        np.testing.assert_allclose(host, expected, rtol=0.0, atol=0.0)

    # ----------------------------
    # Tests: correctness
    # ----------------------------

    def test_fill_f32_sets_all_elements(self) -> None:
        self._run_fill_case(np.float32, value=1.0, numel=1024)

    def test_fill_f64_sets_all_elements(self) -> None:
        self._run_fill_case(np.float64, value=1.0, numel=1024)

    def test_fill_f32_nontrivial_value(self) -> None:
        self._run_fill_case(np.float32, value=-3.25, numel=333)

    def test_fill_f64_nontrivial_value(self) -> None:
        self._run_fill_case(np.float64, value=2.5, numel=333)

    def test_fill_numel_zero_is_noop_with_valid_ptr(self) -> None:
        """
        Your native fill currently returns -1 if y_dev==0, even when numel==0.

        So this test asserts a weaker (but realistic) contract:
        - If numel==0 and y_dev is a valid pointer, the call succeeds
          and does not modify memory.
        """
        for dtype in (np.float32, np.float64):
            # Allocate 1 element to obtain a valid pointer, but request numel=0 fill.
            host_before = np.empty((1,), dtype=dtype)
            host_after = np.empty((1,), dtype=dtype)

            nbytes = int(host_before.nbytes)
            y_dev = cuda_malloc(self.lib, nbytes)
            try:
                # poison
                cuda_memset(self.lib, y_dev, 0xA5, nbytes)
                cuda_synchronize(self.lib)

                cuda_memcpy_d2h(self.lib, host_before, y_dev)

                # Should not raise; should not write anything.
                cuda_fill(self.lib, y_dev=y_dev, numel=0, value=1.0, dtype=dtype)
                cuda_synchronize(self.lib)

                cuda_memcpy_d2h(self.lib, host_after, y_dev)
            finally:
                cuda_free(self.lib, y_dev)

            np.testing.assert_array_equal(host_after, host_before)

    # ----------------------------
    # Tests: input validation
    # ----------------------------

    def test_fill_rejects_unsupported_dtype(self) -> None:
        host = np.empty((4,), dtype=np.float32)
        y_dev = cuda_malloc(self.lib, host.nbytes)
        try:
            with self.assertRaises(TypeError):
                cuda_fill(self.lib, y_dev=y_dev, numel=4, value=1.0, dtype=np.int32)
        finally:
            cuda_free(self.lib, y_dev)


if __name__ == "__main__":
    unittest.main()
