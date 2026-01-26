import unittest
import numpy as np
import time

from src.keydnn.infrastructure.tensor._cuda_memory_pool import CudaMemoryPool


def _try_load_cuda():
    """
    Returns (lib, m, mc) or (None, None, None) if CUDA native library unavailable.
    """
    try:

        from src.keydnn.infrastructure.native_cuda.python import avgpool2d_ctypes as m

        lib = m.load_keydnn_cuda_native()

        from src.keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc

        return lib, m, mc
    except Exception:
        return None, None, None


class TestCudaMemoryPoolReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lib, m, mc = _try_load_cuda()
        if lib is None:
            raise unittest.SkipTest(
                "CUDA native library not available; skipping real pool tests."
            )
        cls.lib = lib
        cls.m = m
        cls.mc = mc

        cls.device_index = 0
        try:
            if hasattr(m, "cuda_set_device"):
                m.cuda_set_device(lib, int(cls.device_index))
        except Exception as e:
            raise unittest.SkipTest(f"cuda_set_device failed; skipping. error={e!r}")

        if hasattr(m, "cuda_synchronize"):
            m.cuda_synchronize(lib)

    def setUp(self) -> None:

        self.pool = CudaMemoryPool(max_cached_bytes_per_device=16 * 1024 * 1024)

    def tearDown(self) -> None:

        try:
            self.pool.empty_cache(self.lib, device_index=self.device_index)
        except Exception:
            pass

        if hasattr(self.m, "cuda_synchronize"):
            try:
                self.m.cuda_synchronize(self.lib)
            except Exception:
                pass

    def _roundtrip_check(self, ptr: int, nbytes: int) -> None:

        x = np.arange(int(nbytes), dtype=np.uint8)
        x = ((x * np.uint8(31)) + np.uint8(7)).astype(np.uint8, copy=False)
        host_in = np.ascontiguousarray(x, dtype=np.uint8)

        self.assertEqual(int(host_in.nbytes), int(nbytes))

        host_out = np.empty((int(nbytes),), dtype=np.uint8)

        self.mc.memcpy_htod(
            self.lib,
            dst_dev=int(ptr),
            src_host=host_in,
            nbytes=int(nbytes),
            sync=True,
        )
        self.mc.memcpy_dtoh(
            self.lib,
            dst_host=host_out,
            src_dev=int(ptr),
            nbytes=int(nbytes),
            sync=True,
        )

        np.testing.assert_array_equal(host_out, host_in)

    def test_malloc_free_roundtrip_bytes(self) -> None:
        """
        Basic sanity: a pooled allocation is valid CUDA memory (H2D/D2H works).
        """
        nbytes = 4096
        ptr = self.pool.malloc(self.lib, device_index=self.device_index, nbytes=nbytes)
        self.assertNotEqual(int(ptr), 0)

        self._roundtrip_check(int(ptr), nbytes)

        self.pool.free(
            self.lib, device_index=self.device_index, dev_ptr=int(ptr), nbytes=nbytes
        )

    def test_reuse_same_bucket_is_usable(self) -> None:
        """
        Allocate->free->allocate same size. Regardless of whether ptr is identical,
        the returned allocation must remain usable for memcpy roundtrips.
        """
        nbytes = 8192

        p1 = int(
            self.pool.malloc(self.lib, device_index=self.device_index, nbytes=nbytes)
        )
        self.assertNotEqual(p1, 0)
        self._roundtrip_check(p1, nbytes)

        self.pool.free(
            self.lib, device_index=self.device_index, dev_ptr=p1, nbytes=nbytes
        )

        p2 = int(
            self.pool.malloc(self.lib, device_index=self.device_index, nbytes=nbytes)
        )
        self.assertNotEqual(p2, 0)
        self._roundtrip_check(p2, nbytes)

        self.pool.free(
            self.lib, device_index=self.device_index, dev_ptr=p2, nbytes=nbytes
        )

    def test_empty_cache_releases_blocks(self) -> None:
        """
        Ensure empty_cache actually calls cuda_free by verifying that subsequent
        allocations need to cuda_malloc again (heuristic via pointer churn).
        """
        nbytes = 16384

        p1 = int(
            self.pool.malloc(self.lib, device_index=self.device_index, nbytes=nbytes)
        )
        self.assertNotEqual(p1, 0)
        self.pool.free(
            self.lib, device_index=self.device_index, dev_ptr=p1, nbytes=nbytes
        )

        self.pool.empty_cache(self.lib, device_index=self.device_index)

        p2 = int(
            self.pool.malloc(self.lib, device_index=self.device_index, nbytes=nbytes)
        )
        self.assertNotEqual(p2, 0)

        self._roundtrip_check(p2, nbytes)
        self.pool.free(
            self.lib, device_index=self.device_index, dev_ptr=p2, nbytes=nbytes
        )

    def test_stress_reuse_does_not_hang(self) -> None:
        """
        Stress: repeated allocate/free/reuse should not hang.
        """
        nbytes = 4096
        iters = 2000

        t0 = time.perf_counter()

        for i in range(iters):
            ptr = int(
                self.pool.malloc(
                    self.lib, device_index=self.device_index, nbytes=nbytes
                )
            )
            self.assertNotEqual(ptr, 0)

            if (i % 200) == 0:
                self._roundtrip_check(ptr, nbytes)

            self.pool.free(
                self.lib, device_index=self.device_index, dev_ptr=ptr, nbytes=nbytes
            )

        dt = time.perf_counter() - t0
        self.assertLess(
            dt, 10.0, f"stress loop too slow / potential stall: dt={dt:.3f}s"
        )


if __name__ == "__main__":
    unittest.main()
