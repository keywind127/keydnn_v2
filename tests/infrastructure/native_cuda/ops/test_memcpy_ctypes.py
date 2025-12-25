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

from src.keydnn.infrastructure.native_cuda.python.ops.memcpy_ctypes import (
    cuda_memcpy_d2d,
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


class TestMemcpyCtypes(_CudaTestCase):
    def test_cuda_memcpy_d2d_float32(self) -> None:
        lib = self.lib
        x = (np.random.rand(64) - 0.5).astype(np.float32, copy=False)
        nbytes = int(x.nbytes)

        src = int(cuda_malloc(lib, nbytes))
        dst = int(cuda_malloc(lib, nbytes))
        try:
            cudaMemcpyHtoD(lib, src, x, nbytes)
            cuda_memcpy_d2d(lib, dst_dev=dst, src_dev=src, nbytes=nbytes)
            cuda_synchronize(lib)

            y = np.empty_like(x)
            cudaMemcpyDtoH(lib, y, dst, nbytes)
            np.testing.assert_allclose(y, x, rtol=0, atol=0)
        finally:
            cuda_free(lib, src)
            cuda_free(lib, dst)

    def test_cuda_memcpy_d2d_float64(self) -> None:
        lib = self.lib
        x = (np.random.rand(32) - 0.5).astype(np.float64, copy=False)
        nbytes = int(x.nbytes)

        src = int(cuda_malloc(lib, nbytes))
        dst = int(cuda_malloc(lib, nbytes))
        try:
            cudaMemcpyHtoD(lib, src, x, nbytes)
            cuda_memcpy_d2d(lib, dst_dev=dst, src_dev=src, nbytes=nbytes)
            cuda_synchronize(lib)

            y = np.empty_like(x)
            cudaMemcpyDtoH(lib, y, dst, nbytes)
            np.testing.assert_allclose(y, x, rtol=0, atol=0)
        finally:
            cuda_free(lib, src)
            cuda_free(lib, dst)

    def test_cuda_memcpy_d2d_zero_bytes_is_noop(self) -> None:
        lib = self.lib
        # nbytes==0 should be allowed and not crash, even if pointers are non-zero/zero.
        # We still allocate to stay safe with stricter native contracts.
        src = int(cuda_malloc(lib, 1))
        dst = int(cuda_malloc(lib, 1))
        try:
            cuda_memcpy_d2d(lib, dst_dev=dst, src_dev=src, nbytes=0)
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, src)
            cuda_free(lib, dst)


if __name__ == "__main__":
    unittest.main()
