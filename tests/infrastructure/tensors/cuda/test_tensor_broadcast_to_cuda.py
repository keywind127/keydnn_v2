from __future__ import annotations

import unittest
import numpy as np


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _get_ctx(t):
    return getattr(t, "_ctx", None) or getattr(t, "ctx", None)


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorBroadcastToCUDA(unittest.TestCase):
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

    # -------------------------
    # helpers: cuda <-> numpy
    # -------------------------
    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool):
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        t = self.Tensor(
            shape=arr.shape, device=self.dev_cuda, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
        self.assertNotEqual(int(t.data), 0)

        # Copy only if there is payload
        nbytes = int(arr.nbytes)
        if nbytes > 0:
            self._memcpy_htod(
                dst_dev=int(t.data),
                src_host=arr,
                nbytes=nbytes,
                sync=True,
            )
        return t

    def _cuda_to_numpy(self, t) -> np.ndarray:
        out = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        if int(out.nbytes) > 0:
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
    def test_broadcast_to_cuda_preserves_dtype_and_device(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((3, 1)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.broadcast_to((3, 5))

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (3, 5))
        self.assertEqual(
            np.dtype(getattr(y, "dtype", x_np.dtype)), np.dtype(np.float32)
        )
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = np.broadcast_to(x_np, (3, 5)).astype(np.float32, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_broadcast_to_cuda_rank_increase(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((4,)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.broadcast_to((2, 4))

        self.assertEqual(tuple(y.shape), (2, 4))
        got = self._cuda_to_numpy(y)
        ref = np.broadcast_to(x_np, (2, 4)).astype(np.float32, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_broadcast_to_cuda_multi_axis(self) -> None:
        # (1,3,1) -> (2,3,4)
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((1, 3, 1)).astype(np.float64)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.broadcast_to((2, 3, 4))

        self.assertEqual(np.dtype(y.dtype), np.dtype(np.float64))
        self.assertEqual(tuple(y.shape), (2, 3, 4))
        got = self._cuda_to_numpy(y)
        ref = np.broadcast_to(x_np, (2, 3, 4)).astype(np.float64, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_broadcast_to_cuda_is_materialized_deep_copy(self) -> None:
        # verify y is not a view of x: mutating y should not change x
        x_np = np.arange(3, dtype=np.float32).reshape(3, 1)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.broadcast_to((3, 4))
        self.assertNotEqual(int(y.data), int(x.data))

        # overwrite y device memory with a new constant tensor payload
        y_new = (np.ones((3, 4), dtype=np.float32) * -999.0).astype(
            np.float32, copy=False
        )
        self._memcpy_htod(
            dst_dev=int(y.data),
            src_host=y_new,
            nbytes=int(y_new.nbytes),
            sync=True,
        )

        # x should remain unchanged
        x_back = self._cuda_to_numpy(x)
        np.testing.assert_allclose(x_back, x_np, rtol=0, atol=0)

        # y should now equal y_new
        y_back = self._cuda_to_numpy(y)
        np.testing.assert_allclose(y_back, y_new, rtol=0, atol=0)

    def test_broadcast_to_cuda_attaches_ctx_when_requires_grad(self) -> None:
        x_np = np.ones((2, 1), dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.broadcast_to((2, 3))

        ctx = _get_ctx(y)
        self.assertIsNotNone(ctx)

        # Optional: verify metadata
        saved_meta = getattr(ctx, "saved_meta", {})
        self.assertEqual(saved_meta.get("broadcast_from"), (2, 1))
        self.assertEqual(saved_meta.get("broadcast_to"), (2, 3))

    def test_broadcast_to_cuda_incompatible_shape_raises(self) -> None:
        x_np = np.ones((2, 3), dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = x.broadcast_to((2, 4))

    def test_broadcast_to_cuda_scalar_to_tensor_if_supported(self) -> None:
        """
        Scalar (0-d) input broadcast. Skip if your Tensor constructor disallows 0-d shapes.
        """
        try:
            # 0-d ndarray
            x_np = np.array(3.25, dtype=np.float64)
            x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        except Exception as e:
            self.skipTest(f"0-d CUDA Tensor not supported in this repo: {e!r}")
            return

        y = x.broadcast_to((2, 3, 4))
        self.assertEqual(tuple(y.shape), (2, 3, 4))
        got = self._cuda_to_numpy(y)
        ref = np.broadcast_to(x_np, (2, 3, 4)).astype(np.float64, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_broadcast_to_cuda_zero_numel_is_ok(self) -> None:
        x_np = np.ones((1,), dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.broadcast_to((0, 5))
        self.assertEqual(tuple(y.shape), (0, 5))
        got = self._cuda_to_numpy(y)
        self.assertEqual(got.size, 0)


if __name__ == "__main__":
    unittest.main()
