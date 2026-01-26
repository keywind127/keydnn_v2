import unittest
from unittest.mock import patch


class FakeCudaBackend:
    """
    Fake CUDA backend for unit testing without a real CUDA DLL.
    """

    def __init__(self) -> None:
        self.active_device = 0
        self.set_device_calls = []
        self.malloc_calls = []
        self.free_calls = []
        self._next_ptr = 0x1000

    def cuda_set_device(self, lib, device_index: int) -> None:
        self.active_device = int(device_index)
        self.set_device_calls.append(int(device_index))

    def cuda_malloc(self, lib, nbytes: int) -> int:
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise AssertionError("cuda_malloc called with non-positive nbytes")

        ptr = self._next_ptr
        self._next_ptr += 0x1000
        self.malloc_calls.append((self.active_device, nbytes))
        return int(ptr)

    def cuda_free(self, lib, ptr: int) -> None:
        self.free_calls.append((self.active_device, int(ptr)))


class TestCudaMemoryPool(unittest.TestCase):
    def setUp(self) -> None:

        self.fake = FakeCudaBackend()

        self.p_set = patch(
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes.cuda_set_device",
            new=self.fake.cuda_set_device,
            create=True,
        )
        self.p_malloc = patch(
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes.cuda_malloc",
            new=self.fake.cuda_malloc,
            create=True,
        )
        self.p_free = patch(
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes.cuda_free",
            new=self.fake.cuda_free,
            create=True,
        )

        self.p_set.start()
        self.p_malloc.start()
        self.p_free.start()

        from keydnn.infrastructure.tensor._cuda_memory_pool import CudaMemoryPool

        self.CudaMemoryPool = CudaMemoryPool

        self.lib = object()

    def tearDown(self) -> None:
        self.p_free.stop()
        self.p_malloc.stop()
        self.p_set.stop()

    def test_reuse_same_device(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1 << 30)

        p1 = pool.malloc(self.lib, device_index=0, nbytes=1024)
        self.assertNotEqual(p1, 0)
        self.assertEqual(len(self.fake.malloc_calls), 1)

        pool.free(self.lib, device_index=0, dev_ptr=p1, nbytes=1024)
        self.assertEqual(len(self.fake.free_calls), 0)

        p2 = pool.malloc(self.lib, device_index=0, nbytes=1024)
        self.assertEqual(p2, p1)
        self.assertEqual(len(self.fake.malloc_calls), 1)

    def test_per_device_isolated(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1 << 30)

        p0 = pool.malloc(self.lib, device_index=0, nbytes=2048)
        pool.free(self.lib, device_index=0, dev_ptr=p0, nbytes=2048)

        p1 = pool.malloc(self.lib, device_index=1, nbytes=2048)
        self.assertNotEqual(p1, p0)
        self.assertEqual(len(self.fake.malloc_calls), 2)

    def test_cache_limit_forces_cuda_free(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1024)

        p = pool.malloc(self.lib, device_index=0, nbytes=4096)
        self.assertEqual(len(self.fake.malloc_calls), 1)

        pool.free(self.lib, device_index=0, dev_ptr=p, nbytes=4096)

        self.assertEqual(len(self.fake.free_calls), 1)

    def test_empty_cache_calls_cuda_free(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1 << 30)

        p1 = pool.malloc(self.lib, device_index=0, nbytes=512)
        p2 = pool.malloc(self.lib, device_index=0, nbytes=512)

        pool.free(self.lib, device_index=0, dev_ptr=p1, nbytes=512)
        pool.free(self.lib, device_index=0, dev_ptr=p2, nbytes=512)

        self.assertEqual(len(self.fake.free_calls), 0)

        pool.empty_cache(self.lib, device_index=0)
        self.assertEqual(len(self.fake.free_calls), 2)

    def test_free_unknown_size_still_works_for_pool_owned_ptr(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1 << 30)

        p = pool.malloc(self.lib, device_index=0, nbytes=1024)

        pool.free(self.lib, device_index=0, dev_ptr=p, nbytes=0)

        self.assertEqual(len(self.fake.free_calls), 0)
        p2 = pool.malloc(self.lib, device_index=0, nbytes=1024)
        self.assertEqual(p2, p)

    def test_stats_smoke(self) -> None:
        pool = self.CudaMemoryPool(max_cached_bytes_per_device=1 << 30)

        p = pool.malloc(self.lib, device_index=0, nbytes=1000)
        pool.free(self.lib, device_index=0, dev_ptr=p, nbytes=1000)

        st = pool.stats()
        self.assertIn(0, st)
        self.assertGreater(st[0]["cached_bytes"], 0)
        self.assertEqual(st[0]["num_cached_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
