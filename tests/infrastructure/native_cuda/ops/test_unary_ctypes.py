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

from src.keydnn.infrastructure.native_cuda.python.ops.unary_ctypes import (
    exp_cuda,
    mul_cuda,
    mul_scalar_cuda,
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


class TestUnaryCtypes(_CudaTestCase):
    def _run_exp(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        x = (np.random.rand(128) - 0.5).astype(dtype, copy=False)
        y_ref = np.exp(x).astype(dtype, copy=False)

        nbytes = int(x.nbytes)

        x_dev = int(cuda_malloc(lib, nbytes))
        y_dev = int(cuda_malloc(lib, nbytes))
        try:
            cudaMemcpyHtoD(lib, x_dev, x, nbytes)

            exp_cuda(lib, x_dev=x_dev, y_dev=y_dev, numel=int(x.size), dtype=dtype)
            cuda_synchronize(lib)

            y = np.empty_like(x)
            cudaMemcpyDtoH(lib, y, y_dev, nbytes)

            if dtype == np.float32:
                np.testing.assert_allclose(y, y_ref, rtol=2e-6, atol=3e-7)
            else:
                np.testing.assert_allclose(y, y_ref, rtol=1e-14, atol=1e-14)

        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, y_dev)

    def test_exp_float32(self) -> None:
        self._run_exp(np.float32)

    def test_exp_float64(self) -> None:
        self._run_exp(np.float64)

    def test_exp_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            exp_cuda(lib, x_dev=1, y_dev=2, numel=3, dtype=np.int32)

    def _run_mul(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        a = (np.random.rand(256) - 0.5).astype(dtype, copy=False)
        b = (np.random.rand(256) - 0.5).astype(dtype, copy=False)
        y_ref = (a * b).astype(dtype, copy=False)

        nbytes = int(a.nbytes)

        a_dev = int(cuda_malloc(lib, nbytes))
        b_dev = int(cuda_malloc(lib, nbytes))
        y_dev = int(cuda_malloc(lib, nbytes))
        try:
            cudaMemcpyHtoD(lib, a_dev, a, nbytes)
            cudaMemcpyHtoD(lib, b_dev, b, nbytes)

            mul_cuda(
                lib,
                a_dev=a_dev,
                b_dev=b_dev,
                y_dev=y_dev,
                numel=int(a.size),
                dtype=dtype,
            )
            cuda_synchronize(lib)

            y = np.empty_like(a)
            cudaMemcpyDtoH(lib, y, y_dev, nbytes)

            if dtype == np.float32:
                np.testing.assert_allclose(y, y_ref, rtol=2e-6, atol=3e-7)
            else:
                np.testing.assert_allclose(y, y_ref, rtol=1e-14, atol=1e-14)

        finally:
            cuda_free(lib, a_dev)
            cuda_free(lib, b_dev)
            cuda_free(lib, y_dev)

    def test_mul_float32(self) -> None:
        self._run_mul(np.float32)

    def test_mul_float64(self) -> None:
        self._run_mul(np.float64)

    def test_mul_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            mul_cuda(lib, a_dev=1, b_dev=2, y_dev=3, numel=4, dtype=np.int32)

    def test_exp_and_mul_numel_zero_is_ok(self) -> None:
        lib = self.lib
        dtype = np.float32
        # numel==0 should not crash; allocate minimal buffers to satisfy strict contracts.
        x_dev = int(cuda_malloc(lib, 1))
        y_dev = int(cuda_malloc(lib, 1))
        a_dev = int(cuda_malloc(lib, 1))
        b_dev = int(cuda_malloc(lib, 1))
        try:
            exp_cuda(lib, x_dev=x_dev, y_dev=y_dev, numel=0, dtype=dtype)
            mul_cuda(lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, numel=0, dtype=dtype)
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, y_dev)
            cuda_free(lib, a_dev)
            cuda_free(lib, b_dev)

    def _run_mul_scalar(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        a = (np.random.rand(256) - 0.5).astype(dtype, copy=False)
        alpha = dtype.type(0.37)
        y_ref = (a * alpha).astype(dtype, copy=False)

        nbytes = int(a.nbytes)

        a_dev = int(cuda_malloc(lib, nbytes))
        y_dev = int(cuda_malloc(lib, nbytes))
        try:
            cudaMemcpyHtoD(lib, a_dev, a, nbytes)

            mul_scalar_cuda(
                lib,
                a_dev=a_dev,
                alpha=float(alpha),
                y_dev=y_dev,
                numel=int(a.size),
                dtype=dtype,
            )
            cuda_synchronize(lib)

            y = np.empty_like(a)
            cudaMemcpyDtoH(lib, y, y_dev, nbytes)

            if dtype == np.float32:
                np.testing.assert_allclose(y, y_ref, rtol=2e-6, atol=3e-7)
            else:
                np.testing.assert_allclose(y, y_ref, rtol=1e-14, atol=1e-14)

        finally:
            cuda_free(lib, a_dev)
            cuda_free(lib, y_dev)

    def test_mul_scalar_float32(self) -> None:
        self._run_mul_scalar(np.float32)

    def test_mul_scalar_float64(self) -> None:
        self._run_mul_scalar(np.float64)

    def test_mul_scalar_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            mul_scalar_cuda(
                lib,
                a_dev=1,
                alpha=1.0,
                y_dev=2,
                numel=3,
                dtype=np.int32,
            )

    def test_mul_scalar_numel_zero_is_ok(self) -> None:
        lib = self.lib
        dtype = np.float32
        a_dev = int(cuda_malloc(lib, 1))
        y_dev = int(cuda_malloc(lib, 1))
        try:
            mul_scalar_cuda(
                lib,
                a_dev=a_dev,
                alpha=2.0,
                y_dev=y_dev,
                numel=0,
                dtype=dtype,
            )
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, a_dev)
            cuda_free(lib, y_dev)


if __name__ == "__main__":
    unittest.main()
