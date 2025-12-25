import unittest

import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_h2d,
    cuda_memcpy_d2h,
    cuda_memset,
)
from src.keydnn.infrastructure.ops.fill_cuda import (
    fill_cuda,
    zeros_cuda,
    ones_cuda,
)


class TestCudaFillOps(unittest.TestCase):
    """
    Unit tests for infrastructure CUDA fill ops (fill_cuda / zeros_cuda / ones_cuda).

    These tests validate:
    - fill_cuda writes correct constant values for float32/float64
    - zeros_cuda produces all-zeros (memset path)
    - ones_cuda produces all-ones (scalar fill path)
    - dtype validation rejects unsupported dtypes
    - sync flag does not change correctness (smoke)
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Load CUDA DLL once and set device 0. Skip suite if unavailable.
        try:
            cls.lib = load_keydnn_cuda_native()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        try:
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA device unavailable / cannot set device: {e}")

    def _run_fill_case(self, dtype: np.dtype, *, value: float, numel: int) -> None:
        # Allocate output on host/device, initialize device with a sentinel, then fill.
        y = np.empty((numel,), dtype=dtype)
        y_dev = cuda_malloc(
            self.lib, y.nbytes if y.nbytes > 0 else np.dtype(dtype).itemsize
        )

        try:
            # Initialize device buffer to non-zero bytes so we can see a change.
            # (For numel==0, still a minimal allocation so memset is valid.)
            cuda_memset(self.lib, y_dev, 0x7F, max(1, y.nbytes))

            fill_cuda(
                self.lib,
                y_dev=y_dev,
                numel=numel,
                value=value,
                dtype=dtype,
                sync=True,
            )

            if numel > 0:
                cuda_memcpy_d2h(self.lib, y, y_dev)
                y_ref = np.full((numel,), value, dtype=dtype)
                atol = 1e-6 if dtype == np.float32 else 1e-12
                rtol = 0.0
                np.testing.assert_allclose(y, y_ref, rtol=rtol, atol=atol)
        finally:
            cuda_free(self.lib, y_dev)

    def _run_zeros_case(self, dtype: np.dtype, *, numel: int) -> None:
        y = np.empty((numel,), dtype=dtype)
        y_dev = cuda_malloc(
            self.lib, y.nbytes if y.nbytes > 0 else np.dtype(dtype).itemsize
        )

        try:
            cuda_memset(self.lib, y_dev, 0x7F, max(1, y.nbytes))

            zeros_cuda(
                self.lib,
                y_dev=y_dev,
                numel=numel,
                dtype=dtype,
                sync=True,
            )

            if numel > 0:
                cuda_memcpy_d2h(self.lib, y, y_dev)
                y_ref = np.zeros((numel,), dtype=dtype)
                # Exact zeros expected
                np.testing.assert_array_equal(y, y_ref)
        finally:
            cuda_free(self.lib, y_dev)

    def _run_ones_case(self, dtype: np.dtype, *, numel: int) -> None:
        y = np.empty((numel,), dtype=dtype)
        y_dev = cuda_malloc(
            self.lib, y.nbytes if y.nbytes > 0 else np.dtype(dtype).itemsize
        )

        try:
            cuda_memset(self.lib, y_dev, 0x00, max(1, y.nbytes))

            ones_cuda(
                self.lib,
                y_dev=y_dev,
                numel=numel,
                dtype=dtype,
                sync=True,
            )

            if numel > 0:
                cuda_memcpy_d2h(self.lib, y, y_dev)
                y_ref = np.ones((numel,), dtype=dtype)
                atol = 1e-6 if dtype == np.float32 else 1e-12
                rtol = 0.0
                np.testing.assert_allclose(y, y_ref, rtol=rtol, atol=atol)
        finally:
            cuda_free(self.lib, y_dev)

    # ----------------------------
    # Tests: correctness
    # ----------------------------

    def test_fill_f32_matches_numpy(self) -> None:
        self._run_fill_case(np.float32, value=3.25, numel=1024)

    def test_fill_f64_matches_numpy(self) -> None:
        self._run_fill_case(np.float64, value=-1.5, numel=2048)

    def test_zeros_f32_produces_all_zeros(self) -> None:
        self._run_zeros_case(np.float32, numel=4096)

    def test_zeros_f64_produces_all_zeros(self) -> None:
        self._run_zeros_case(np.float64, numel=4096)

    def test_ones_f32_produces_all_ones(self) -> None:
        self._run_ones_case(np.float32, numel=4096)

    def test_ones_f64_produces_all_ones(self) -> None:
        self._run_ones_case(np.float64, numel=4096)

    # ----------------------------
    # Tests: dtype validation
    # ----------------------------

    def test_fill_rejects_unsupported_dtype(self) -> None:
        y_dev = cuda_malloc(self.lib, 4)
        try:
            with self.assertRaises(TypeError):
                fill_cuda(
                    self.lib,
                    y_dev=y_dev,
                    numel=1,
                    value=1.0,
                    dtype=np.int32,  # unsupported
                    sync=True,
                )
        finally:
            cuda_free(self.lib, y_dev)

    def test_zeros_rejects_unsupported_dtype(self) -> None:
        y_dev = cuda_malloc(self.lib, 4)
        try:
            with self.assertRaises(TypeError):
                zeros_cuda(
                    self.lib,
                    y_dev=y_dev,
                    numel=1,
                    dtype=np.int32,  # unsupported
                    sync=True,
                )
        finally:
            cuda_free(self.lib, y_dev)

    def test_ones_rejects_unsupported_dtype(self) -> None:
        y_dev = cuda_malloc(self.lib, 4)
        try:
            with self.assertRaises(TypeError):
                ones_cuda(
                    self.lib,
                    y_dev=y_dev,
                    numel=1,
                    dtype=np.int32,  # unsupported
                    sync=True,
                )
        finally:
            cuda_free(self.lib, y_dev)

    # ----------------------------
    # Tests: sync flag smoke
    # ----------------------------

    def test_fill_sync_false_still_correct(self) -> None:
        # If your fill_ctypes / ops are correct, disabling sync should still be correct
        # as long as the subsequent D2H copy synchronizes implicitly (or driver does).
        # This is a smoke test, not a strict async semantics test.
        dtype = np.float32
        numel = 1024
        value = 2.0

        y = np.empty((numel,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y.nbytes)

        try:
            cuda_memset(self.lib, y_dev, 0x7F, y.nbytes)

            fill_cuda(
                self.lib,
                y_dev=y_dev,
                numel=numel,
                value=value,
                dtype=dtype,
                sync=False,
            )

            cuda_memcpy_d2h(self.lib, y, y_dev)
            np.testing.assert_allclose(
                y, np.full((numel,), value, dtype=dtype), rtol=0.0, atol=1e-6
            )
        finally:
            cuda_free(self.lib, y_dev)

    # ----------------------------
    # Edge cases
    # ----------------------------

    def test_fill_numel_zero_does_not_crash_with_valid_ptr(self) -> None:
        # IMPORTANT: Do not pass y_dev=0 unless native contract explicitly allows it.
        # We allocate a minimal buffer and fill 0 elements; should be a no-op.
        self._run_fill_case(np.float32, value=1.0, numel=0)

    def test_zeros_numel_zero_does_not_crash_with_valid_ptr(self) -> None:
        self._run_zeros_case(np.float64, numel=0)

    def test_ones_numel_zero_does_not_crash_with_valid_ptr(self) -> None:
        self._run_ones_case(np.float32, numel=0)


if __name__ == "__main__":
    unittest.main()
