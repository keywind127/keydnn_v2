from __future__ import annotations

import unittest
import numpy as np


def _cuda_available() -> bool:
    try:
        # same pattern used in your other cuda tests
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _get_ctx(t):
    # supports either _ctx or ctx attribute
    return getattr(t, "_ctx", None) or getattr(t, "ctx", None)


class TestTensorCloneCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev_cpu = Device("cpu")

    def test_clone_cpu_preserves_shape_dtype_and_values(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=True, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.clone()

        self.assertTrue(y.device.is_cpu())
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(
            np.dtype(getattr(y, "dtype", x_np.dtype)), np.dtype(np.float32)
        )

        np.testing.assert_allclose(y.to_numpy(), x_np, rtol=0, atol=0)

    def test_clone_cpu_is_deep_copy(self) -> None:
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.clone()

        # mutate clone's numpy buffer
        y_arr = y.to_numpy()
        y_arr[...] = -999.0

        # original must remain unchanged
        np.testing.assert_allclose(x.to_numpy(), x_np, rtol=0, atol=0)

    def test_clone_cpu_no_ctx_and_no_requires_grad(self) -> None:
        x_np = np.ones((2, 2), dtype=np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=True, ctx="dummy"
        )
        x.copy_from_numpy(x_np)

        y = x.clone()

        self.assertFalse(getattr(y, "requires_grad", True))
        self.assertIsNone(_get_ctx(y))


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorCloneCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev_cuda = Device("cuda:0")

        lib = Tensor._get_cuda_lib()
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )

        cuda_set_device(lib, 0)

        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        cls.lib = lib
        cls.mc = mc

    # -------------------------
    # helpers: choose correct memcpy symbols
    # -------------------------
    def _memcpy_htod(
        self, *, dst_dev: int, src_host: np.ndarray, nbytes: int, sync: bool
    ) -> None:
        mc = self.mc
        if hasattr(mc, "memcpy_htod"):
            mc.memcpy_htod(
                self.lib, dst_dev=dst_dev, src_host=src_host, nbytes=nbytes, sync=sync
            )
            return
        if hasattr(mc, "cuda_memcpy_h2d"):
            mc.cuda_memcpy_h2d(
                self.lib, dst_dev=dst_dev, src_host=src_host, nbytes=nbytes, sync=sync
            )
            return
        raise RuntimeError("memcpy_ctypes missing memcpy_htod/cuda_memcpy_h2d")

    def _memcpy_dtoh(
        self, *, dst_host: np.ndarray, src_dev: int, nbytes: int, sync: bool
    ) -> None:
        mc = self.mc
        if hasattr(mc, "memcpy_dtoh"):
            mc.memcpy_dtoh(
                self.lib, dst_host=dst_host, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        if hasattr(mc, "cuda_memcpy_d2h"):
            mc.cuda_memcpy_d2h(
                self.lib, dst_host=dst_host, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        raise RuntimeError("memcpy_ctypes missing memcpy_dtoh/cuda_memcpy_d2h")

    def _memcpy_dtod(
        self, *, dst_dev: int, src_dev: int, nbytes: int, sync: bool
    ) -> None:
        mc = self.mc
        if hasattr(mc, "memcpy_dtod"):
            mc.memcpy_dtod(
                self.lib, dst_dev=dst_dev, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        if hasattr(mc, "cuda_memcpy_d2d"):
            mc.cuda_memcpy_d2d(
                self.lib, dst_dev=dst_dev, src_dev=src_dev, nbytes=nbytes, sync=sync
            )
            return
        raise RuntimeError("memcpy_ctypes missing memcpy_dtod/cuda_memcpy_d2d")

    # -------------------------
    # helpers: cuda <-> numpy
    # -------------------------
    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool):
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        t = self.Tensor(
            shape=arr.shape, device=self.dev_cuda, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
        self.assertNotEqual(int(t.data), 0)

        self._memcpy_htod(
            dst_dev=int(t.data),
            src_host=arr,
            nbytes=int(arr.nbytes),
            sync=True,
        )
        return t

    def _cuda_to_numpy(self, t) -> np.ndarray:
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        self._memcpy_dtoh(
            dst_host=out,
            src_dev=int(t.data),
            nbytes=int(out.nbytes),
            sync=True,
        )
        return out

    # -------------------------
    # tests
    # -------------------------
    def test_clone_cuda_preserves_shape_dtype_and_values(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((4, 5)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)
        y = x.clone()

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(
            np.dtype(getattr(y, "dtype", x_np.dtype)), np.dtype(np.float32)
        )
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        np.testing.assert_allclose(got, x_np, rtol=0, atol=0)

    def test_clone_cuda_is_deep_copy(self) -> None:
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.clone()

        # overwrite clone's device memory
        y_new = (np.ones_like(x_np) * -777.0).astype(np.float32)
        self._memcpy_htod(
            dst_dev=int(y.data),
            src_host=y_new,
            nbytes=int(y_new.nbytes),
            sync=True,
        )

        # original unchanged
        x_back = self._cuda_to_numpy(x)
        np.testing.assert_allclose(x_back, x_np, rtol=0, atol=0)

        # clone changed
        y_back = self._cuda_to_numpy(y)
        np.testing.assert_allclose(y_back, y_new, rtol=0, atol=0)

    def test_clone_cuda_no_ctx_and_no_requires_grad(self) -> None:
        x_np = np.ones((2, 3), dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.clone()

        self.assertFalse(getattr(y, "requires_grad", True))
        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNone(ctx)

    def test_cuda_memcpy_roundtrip_sanity(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        x_back = self._cuda_to_numpy(x)

        np.testing.assert_allclose(x_back, x_np, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
