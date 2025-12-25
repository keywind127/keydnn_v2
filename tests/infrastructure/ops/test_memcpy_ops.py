from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func


class TestMemcpyCudaOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        # Import ops module under test
        from src.keydnn.infrastructure.ops import memcpy_cuda as ops_memcpy
        cls.ops_memcpy = ops_memcpy

        # Resolve function names (adjust candidates if your module uses different names)
        cls.memcpy_htod = resolve_func(
            ops_memcpy,
            candidates=["memcpy_htod", "cuda_memcpy_htod", "copy_htod", "htod"],
        )
        cls.memcpy_dtoh = resolve_func(
            ops_memcpy,
            candidates=["memcpy_dtoh", "cuda_memcpy_dtoh", "copy_dtoh", "dtoh"],
        )
        cls.memcpy_dtod = resolve_func(
            ops_memcpy,
            candidates=["memcpy_dtod", "cuda_memcpy_dtod", "copy_dtod", "dtod"],
        )

    def _malloc(self, nbytes: int) -> int:
        return int(self.env.cuda_malloc(self.env.lib, int(nbytes)))

    def _free(self, dev_ptr: int) -> None:
        self.env.cuda_free(self.env.lib, int(dev_ptr))

    def test_htod_and_dtoh_roundtrip_float32(self) -> None:
        x = (np.random.randn(128).astype(np.float32))
        nbytes = int(x.nbytes)
        dev = self._malloc(nbytes)
        try:
            self.memcpy_htod(self.env.lib, dst_dev=int(dev), src_host=x, nbytes=nbytes, sync=True)

            out = np.empty_like(x)
            self.memcpy_dtoh(self.env.lib, dst_host=out, src_dev=int(dev), nbytes=nbytes, sync=True)

            np.testing.assert_array_equal(out, x)
        finally:
            self._free(dev)

    def test_dtod_copies_bytes(self) -> None:
        x = (np.random.randn(64).astype(np.float64))
        nbytes = int(x.nbytes)
        a = self._malloc(nbytes)
        b = self._malloc(nbytes)
        try:
            # host -> a
            self.memcpy_htod(self.env.lib, dst_dev=int(a), src_host=x, nbytes=nbytes, sync=True)
            # a -> b
            self.memcpy_dtod(self.env.lib, dst_dev=int(b), src_dev=int(a), nbytes=nbytes, sync=True)
            # b -> host
            out = np.empty_like(x)
            self.memcpy_dtoh(self.env.lib, dst_host=out, src_dev=int(b), nbytes=nbytes, sync=True)

            np.testing.assert_array_equal(out, x)
        finally:
            self._free(a)
            self._free(b)

    def test_invalid_nbytes_raises(self) -> None:
        x = np.random.randn(8).astype(np.float32)
        nbytes = int(x.nbytes)
        dev = self._malloc(nbytes)
        try:
            with self.assertRaises((ValueError, TypeError)):
                # negative bytes should be rejected at python boundary
                self.memcpy_htod(self.env.lib, dst_dev=int(dev), src_host=x, nbytes=-1, sync=True)
        finally:
            self._free(dev)
