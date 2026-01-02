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


def _sum_to_shape_cpu_ref(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    NumPy reference for sum-to-shape (inverse of broadcast).

    Preconditions:
    - target_shape rank <= x.ndim
    - target_shape is broadcastable to x.shape
    """
    src_shape = tuple(int(d) for d in x.shape)
    tgt_shape = tuple(int(d) for d in target_shape)

    if len(tgt_shape) > len(src_shape):
        raise ValueError("target_shape rank > src rank")

    pad = len(src_shape) - len(tgt_shape)
    padded_tgt = (1,) * pad + tgt_shape

    for sd, td in zip(src_shape, padded_tgt):
        if td not in (1, sd):
            raise ValueError("target_shape not broadcastable to src shape")

    reduce_axes = tuple(
        i
        for i, (sd, td) in enumerate(zip(src_shape, padded_tgt))
        if td == 1 and sd != 1
    )

    y = x
    if reduce_axes:
        y = y.sum(axis=reduce_axes, keepdims=True)

    if pad:
        for _ in range(pad):
            y = np.squeeze(y, axis=0)

    if y.shape != tgt_shape:
        raise RuntimeError(
            f"ref produced wrong shape: got {y.shape}, expected {tgt_shape}"
        )

    return y


class TestTensorSumToShapeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev_cpu = Device("cpu")

    def test_sum_to_shape_cpu_reduces_values(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.sum_to_shape((3, 1))  # keep middle dim, reduce others
        self.assertTrue(y.device.is_cpu())
        self.assertEqual(tuple(y.shape), (3, 1))

        ref = _sum_to_shape_cpu_ref(x_np, (3, 1)).astype(np.float32, copy=False)
        np.testing.assert_allclose(y.to_numpy(), ref, rtol=0, atol=0)

    def test_sum_to_shape_cpu_keeps_dtype_and_shape(self) -> None:
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.sum_to_shape((1, 3, 1))
        self.assertEqual(tuple(y.shape), (1, 3, 1))
        self.assertEqual(
            np.dtype(getattr(y, "dtype", np.float32)), np.dtype(np.float32)
        )

        ref = _sum_to_shape_cpu_ref(x_np, (1, 3, 1))
        np.testing.assert_allclose(y.to_numpy(), ref, rtol=0, atol=0)

    def test_sum_to_shape_cpu_invalid_target_raises(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        # target dim 2 is not 1 or 3 -> not broadcastable to source
        with self.assertRaises(ValueError):
            _ = x.sum_to_shape((2, 2, 1))

        # target rank > src rank
        with self.assertRaises(ValueError):
            _ = x.sum_to_shape((1, 2, 3, 4))

    def test_sum_to_shape_cpu_autograd_attaches_ctx(self) -> None:
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=True, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.sum_to_shape((1, 3, 1))
        self.assertTrue(getattr(y, "requires_grad", False))
        self.assertIsNotNone(_get_ctx(y))


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorSumToShapeCUDA(unittest.TestCase):
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
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        t = self.Tensor(
            shape=arr.shape, device=self.dev_cuda, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
        self.assertNotEqual(int(t.data), 0)

        self._memcpy_htod(
            dst_dev=int(t.data), src_host=arr, nbytes=int(arr.nbytes), sync=True
        )
        return t

    def _cuda_to_numpy(self, t) -> np.ndarray:
        shape = tuple(int(d) for d in t.shape)
        if shape == ():
            out = np.empty((1,), dtype=np.dtype(t.dtype))
        else:
            out = np.empty(shape, dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        self._memcpy_dtoh(
            dst_host=out, src_dev=int(t.data), nbytes=int(out.nbytes), sync=True
        )
        return out[0:1] if shape == () else out

    # -------------------------
    # tests
    # -------------------------
    def test_sum_to_shape_cuda_reduces_values(self) -> None:
        rng = np.random.default_rng(10)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum_to_shape((3, 1))

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (3, 1))
        self.assertEqual(
            np.dtype(getattr(y, "dtype", np.float32)), np.dtype(np.float32)
        )
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = _sum_to_shape_cpu_ref(x_np, (3, 1)).astype(np.float32, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_sum_to_shape_cuda_rank_drop(self) -> None:
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((5, 7)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum_to_shape((7,))  # drop leading dim via sum

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (7,))

        got = self._cuda_to_numpy(y)
        ref = _sum_to_shape_cpu_ref(x_np, (7,)).astype(np.float32, copy=False)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_sum_to_shape_cuda_invalid_target_raises(self) -> None:
        rng = np.random.default_rng(12)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = x.sum_to_shape((2, 2, 1))

        with self.assertRaises(ValueError):
            _ = x.sum_to_shape((1, 2, 3, 4))

    def test_sum_to_shape_cuda_autograd_attaches_ctx(self) -> None:
        rng = np.random.default_rng(13)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)
        y = x.sum_to_shape((1, 3, 1))

        self.assertTrue(getattr(y, "requires_grad", False))
        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)


if __name__ == "__main__":
    unittest.main()
