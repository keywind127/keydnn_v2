from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


class TestUnaryCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import unary_cuda as ops_unary

        cls.ops_unary = ops_unary

        cls.exp_cuda = resolve_func(
            ops_unary,
            candidates=["exp_cuda", "cuda_exp", "exp", "unary_exp_cuda"],
        )

    def _malloc(self, nbytes: int) -> int:
        return int(self.env.cuda_malloc(self.env.lib, int(nbytes)))

    def _free(self, dev_ptr: int) -> None:
        self.env.cuda_free(self.env.lib, int(dev_ptr))

    def _run_exp(self, dtype: np.dtype, numel: int = 128) -> None:
        dtype = np.dtype(dtype)
        x = np.random.randn(numel).astype(dtype)
        y_ref = np.exp(x).astype(dtype)

        x_dev = self._malloc(int(x.nbytes))
        y_dev = self._malloc(int(y_ref.nbytes))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(x_dev), x, int(x.nbytes))

            # expected signature:
            # exp_cuda(lib, x_dev=..., y_dev=..., numel=..., dtype=..., sync=True)
            self.exp_cuda(
                self.env.lib,
                x_dev=int(x_dev),
                y_dev=int(y_dev),
                numel=int(numel),
                dtype=dtype,
                sync=True,
            )

            y = np.empty_like(y_ref)
            self.env.cudaMemcpyDtoH(self.env.lib, y, int(y_dev), int(y.nbytes))

            assert_allclose_by_dtype(y, y_ref, dtype, op="exp")
        finally:
            self._free(x_dev)
            self._free(y_dev)

    def test_exp_float32(self) -> None:
        self._run_exp(np.float32, numel=256)

    def test_exp_float64(self) -> None:
        self._run_exp(np.float64, numel=256)

    def test_exp_rejects_int_dtype(self) -> None:
        x = np.arange(16, dtype=np.int32)
        x_dev = self._malloc(int(x.nbytes))
        y_dev = self._malloc(int(x.nbytes))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(x_dev), x, int(x.nbytes))
            with self.assertRaises((TypeError, ValueError)):
                self.exp_cuda(
                    self.env.lib,
                    x_dev=int(x_dev),
                    y_dev=int(y_dev),
                    numel=16,
                    dtype=np.int32,
                    sync=True,
                )
        finally:
            self._free(x_dev)
            self._free(y_dev)
