from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_synchronize,
    cudaMemcpyHtoD,
    cudaMemcpyDtoH,
)

from src.keydnn.infrastructure.native_cuda.python.ops.transpose_ctypes import (
    transpose2d_cuda,
)


class _CudaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = load_keydnn_cuda_native()
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            cls.lib = None
            cls._skip_reason = f"CUDA native library not available: {e!r}"

    def setUp(self) -> None:
        if getattr(self, "lib", None) is None:
            self.skipTest(getattr(self, "_skip_reason", "CUDA not available"))


class TestTransposeCtypes(_CudaTestCase):
    def _run_transpose(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        rows, cols = 7, 11
        x = (np.random.rand(rows, cols) - 0.5).astype(dtype, copy=False)
        y_ref = x.T.copy()

        nbytes_x = int(x.nbytes)
        nbytes_y = int(y_ref.nbytes)

        x_dev = int(cuda_malloc(lib, nbytes_x))
        y_dev = int(cuda_malloc(lib, nbytes_y))
        try:
            cudaMemcpyHtoD(lib, x_dev, x, nbytes_x)

            transpose2d_cuda(
                lib,
                x_dev=x_dev,
                y_dev=y_dev,
                rows=rows,
                cols=cols,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            y = np.empty_like(y_ref)
            cudaMemcpyDtoH(lib, y, y_dev, nbytes_y)
            np.testing.assert_allclose(y, y_ref, rtol=0, atol=0)
        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, y_dev)

    def test_transpose2d_float32(self) -> None:
        self._run_transpose(np.float32)

    def test_transpose2d_float64(self) -> None:
        self._run_transpose(np.float64)

    def test_transpose2d_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            transpose2d_cuda(
                lib,
                x_dev=123,
                y_dev=456,
                rows=2,
                cols=3,
                dtype=np.int32,
            )

    def test_transpose2d_zero_rows_or_cols_is_ok(self) -> None:
        lib = self.lib
        # If rows==0 or cols==0, kernel should be no-op; we still allocate minimal buffers.
        dtype = np.float32
        x = np.empty((0, 5), dtype=dtype)
        y = np.empty((5, 0), dtype=dtype)

        x_dev = int(cuda_malloc(lib, 1))
        y_dev = int(cuda_malloc(lib, 1))
        try:
            transpose2d_cuda(lib, x_dev=x_dev, y_dev=y_dev, rows=0, cols=5, dtype=dtype)
            transpose2d_cuda(lib, x_dev=x_dev, y_dev=y_dev, rows=5, cols=0, dtype=dtype)
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, y_dev)


if __name__ == "__main__":
    unittest.main()
