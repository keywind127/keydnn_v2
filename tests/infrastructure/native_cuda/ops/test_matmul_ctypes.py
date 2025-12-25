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

from src.keydnn.infrastructure.native_cuda.python.ops.matmul_ctypes import matmul_cuda


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


class TestMatmulCtypes(_CudaTestCase):
    def _run_matmul(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        # Small sizes for deterministic + fast tests
        M, K, N = 5, 7, 4

        A = (np.random.rand(M, K) - 0.5).astype(dtype, copy=False)
        B = (np.random.rand(K, N) - 0.5).astype(dtype, copy=False)
        C_ref = (A @ B).astype(dtype, copy=False)

        nbytes_A = int(A.nbytes)
        nbytes_B = int(B.nbytes)
        nbytes_C = int(C_ref.nbytes)

        a_dev = int(cuda_malloc(lib, nbytes_A))
        b_dev = int(cuda_malloc(lib, nbytes_B))
        c_dev = int(cuda_malloc(lib, nbytes_C))
        try:
            cudaMemcpyHtoD(lib, a_dev, A, nbytes_A)
            cudaMemcpyHtoD(lib, b_dev, B, nbytes_B)

            matmul_cuda(
                lib,
                a_dev=a_dev,
                b_dev=b_dev,
                c_dev=c_dev,
                M=M,
                N=N,
                K=K,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            C = np.empty_like(C_ref)
            cudaMemcpyDtoH(lib, C, c_dev, nbytes_C)

            # naive GEMM should match exactly for these sizes
            if dtype == np.float32:
                np.testing.assert_allclose(C, C_ref, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_allclose(C, C_ref, rtol=1e-12, atol=1e-12)

        finally:
            cuda_free(lib, a_dev)
            cuda_free(lib, b_dev)
            cuda_free(lib, c_dev)

    def test_matmul_float32(self) -> None:
        self._run_matmul(np.float32)

    def test_matmul_float64(self) -> None:
        self._run_matmul(np.float64)

    def test_matmul_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            matmul_cuda(
                lib,
                a_dev=1,
                b_dev=2,
                c_dev=3,
                M=1,
                N=1,
                K=1,
                dtype=np.int32,
            )

    def test_matmul_zero_dim_is_ok(self) -> None:
        lib = self.lib
        dtype = np.float32
        # M==0 or N==0 should no-op
        a_dev = int(cuda_malloc(lib, 1))
        b_dev = int(cuda_malloc(lib, 1))
        c_dev = int(cuda_malloc(lib, 1))
        try:
            matmul_cuda(
                lib, a_dev=a_dev, b_dev=b_dev, c_dev=c_dev, M=0, N=3, K=5, dtype=dtype
            )
            matmul_cuda(
                lib, a_dev=a_dev, b_dev=b_dev, c_dev=c_dev, M=3, N=0, K=5, dtype=dtype
            )
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, a_dev)
            cuda_free(lib, b_dev)
            cuda_free(lib, c_dev)


if __name__ == "__main__":
    unittest.main()
