from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func


class TestTransposeCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import transpose_cuda as ops_transpose

        cls.ops_transpose = ops_transpose

        cls.transpose2d = resolve_func(
            ops_transpose,
            candidates=[
                "transpose2d_cuda",
                "transpose_cuda",
                "transpose2d",
                "transpose",
            ],
        )

    def _malloc(self, nbytes: int) -> int:
        return int(self.env.cuda_malloc(self.env.lib, int(nbytes)))

    def _free(self, dev_ptr: int) -> None:
        self.env.cuda_free(self.env.lib, int(dev_ptr))

    def _run_case(self, dtype: np.dtype, shape=(5, 7)) -> None:
        dtype = np.dtype(dtype)
        x = np.random.randn(*shape).astype(dtype)
        y_ref = x.T.copy()

        x_dev = self._malloc(int(x.nbytes))
        y_dev = self._malloc(int(y_ref.nbytes))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(x_dev), x, int(x.nbytes))

            # transpose op signature candidates:
            # transpose2d(lib, x_dev=..., y_dev=..., rows=..., cols=..., dtype=..., sync=True)
            self.transpose2d(
                self.env.lib,
                x_dev=int(x_dev),
                y_dev=int(y_dev),
                rows=int(shape[0]),
                cols=int(shape[1]),
                dtype=dtype,
                sync=True,
            )

            y = np.empty_like(y_ref)
            self.env.cudaMemcpyDtoH(self.env.lib, y, int(y_dev), int(y.nbytes))

            np.testing.assert_array_equal(y, y_ref)
        finally:
            self._free(x_dev)
            self._free(y_dev)

    def test_transpose_float32(self) -> None:
        self._run_case(np.float32, shape=(9, 4))

    def test_transpose_float64(self) -> None:
        self._run_case(np.float64, shape=(3, 11))

    def test_transpose_rejects_non_2d(self) -> None:
        # ops module should reject invalid rows/cols or shape semantics
        x = np.random.randn(2, 3, 4).astype(np.float32)
        x_dev = self._malloc(int(x.nbytes))
        y_dev = self._malloc(int(x.nbytes))
        try:
            self.env.cudaMemcpyHtoD(self.env.lib, int(x_dev), x, int(x.nbytes))
            with self.assertRaises((ValueError, TypeError)):
                self.transpose2d(
                    self.env.lib,
                    x_dev=int(x_dev),
                    y_dev=int(y_dev),
                    rows=2,
                    cols=3,  # caller pretending it's 2D; wrapper may validate more strictly
                    dtype=np.float32,
                    sync=True,
                )
        finally:
            self._free(x_dev)
            self._free(y_dev)
