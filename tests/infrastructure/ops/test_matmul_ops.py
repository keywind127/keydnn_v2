from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


class TestMatmulCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import matmul_cuda as ops_mm

        cls.ops_mm = ops_mm

        cls.matmul2d = resolve_func(
            ops_mm,
            candidates=["matmul_cuda", "matmul2d_cuda", "gemm_cuda", "matmul2d"],
        )

    def _malloc(self, nbytes: int) -> int:
        return int(self.env.cuda_malloc(self.env.lib, int(nbytes)))

    def _free(self, dev_ptr: int) -> None:
        self.env.cuda_free(self.env.lib, int(dev_ptr))

    def _run_matmul(self, dtype: np.dtype, A_shape=(5, 7), B_shape=(7, 4)) -> None:
        dtype = np.dtype(dtype)
        A = np.random.randn(*A_shape).astype(dtype)
        B = np.random.randn(*B_shape).astype(dtype)
        C_ref = (A @ B).astype(dtype)

        A_dev = self._malloc(int(A.nbytes))
        B_dev = self._malloc(int(B.nbytes))
        C_dev = self._malloc(int(C_ref.nbytes))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(A_dev), A, int(A.nbytes))
            self.env.cudaMemcpyHtoD(self.env.lib, int(B_dev), B, int(B.nbytes))

            n, k = A_shape
            k2, m = B_shape
            self.assertEqual(k, k2)

            # expected signature:
            # matmul_cuda(lib, a_dev=..., b_dev=..., c_dev=..., n=..., k=..., m=..., dtype=..., sync=True)
            self.matmul2d(
                self.env.lib,
                a_dev=int(A_dev),
                b_dev=int(B_dev),
                c_dev=int(C_dev),
                n=int(n),
                k=int(k),
                m=int(m),
                dtype=dtype,
                sync=True,
            )

            C = np.empty_like(C_ref)
            self.env.cudaMemcpyDtoH(self.env.lib, C, int(C_dev), int(C.nbytes))

            assert_allclose_by_dtype(C, C_ref, dtype, op="matmul")
        finally:
            self._free(A_dev)
            self._free(B_dev)
            self._free(C_dev)

    def test_matmul_float32(self) -> None:
        self._run_matmul(np.float32, A_shape=(8, 16), B_shape=(16, 9))

    def test_matmul_float64(self) -> None:
        self._run_matmul(np.float64, A_shape=(6, 11), B_shape=(11, 5))

    def test_matmul_rejects_shape_mismatch(self) -> None:
        # This should be caught by wrapper checks (k mismatch) or by caller logic.
        dtype = np.float32
        A_shape = (4, 3)
        B_shape = (2, 5)  # mismatch
        A = np.random.randn(*A_shape).astype(dtype)
        B = np.random.randn(*B_shape).astype(dtype)

        A_dev = self._malloc(int(A.nbytes))
        B_dev = self._malloc(int(B.nbytes))
        C_dev = self._malloc(int(4 * 5 * np.dtype(dtype).itemsize))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(A_dev), A, int(A.nbytes))
            self.env.cudaMemcpyHtoD(self.env.lib, int(B_dev), B, int(B.nbytes))

            with self.assertRaises((ValueError, RuntimeError)):
                self.matmul2d(
                    self.env.lib,
                    a_dev=int(A_dev),
                    b_dev=int(B_dev),
                    c_dev=int(C_dev),
                    n=int(A_shape[0]),
                    k=int(A_shape[1]),
                    m=int(B_shape[1]),
                    dtype=dtype,
                    sync=True,
                )
        finally:
            self._free(A_dev)
            self._free(B_dev)
            self._free(C_dev)
