# tests/infrastructure/tensors/cuda/test_tensor_concat_cuda.py

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


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorConcatCudaForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev = Device("cuda:0")

        lib = Tensor._get_cuda_lib()
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )

        cuda_set_device(lib, 0)

        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        cls.mc = mc
        cls.lib = lib

    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool):
        Tensor = self.Tensor
        t = Tensor(
            shape=arr.shape, device=self.dev, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))

        dev_ptr = int(t.data)
        self.assertNotEqual(dev_ptr, 0)

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        self.mc.cuda_memcpy_h2d(self.lib, dev_ptr, arr, int(arr.nbytes))
        return t

    def _cuda_to_numpy(self, t):
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        self.mc.cuda_memcpy_d2h(self.lib, out, int(t.data), int(out.nbytes))
        return out

    def test_concat_cuda_allocates_device_memory(self) -> None:
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.ones((4, 3), dtype=np.float32) * 2

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        out = self.Tensor.concat([a, b], axis=0)
        self.assertTrue(out.device.is_cuda())
        self.assertNotEqual(int(out.data), 0)
        self.assertEqual(out.shape, (6, 3))

    def test_concat_cuda_forward_axis0_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a_np = rng.standard_normal((2, 5)).astype(np.float32)
        b_np = rng.standard_normal((3, 5)).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        out = self.Tensor.concat([a, b], axis=0)
        got = self._cuda_to_numpy(out)

        exp = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_concat_cuda_axis1_fallback_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a_np = rng.standard_normal((2, 3)).astype(np.float32)
        b_np = rng.standard_normal((2, 4)).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        out = self.Tensor.concat([a, b], axis=1)
        got = self._cuda_to_numpy(out)

        exp = np.concatenate([a_np, b_np], axis=1)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_concat_cuda_rejects_mismatched_shapes(self) -> None:
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.ones((4, 4), dtype=np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = self.Tensor.concat([a, b], axis=0)

    def test_concat_cuda_mixed_device_raises(self) -> None:
        a_np = np.ones((2, 3), dtype=np.float32)
        a = self._cuda_tensor_from_numpy(a_np, requires_grad=False)

        b_np = np.ones((2, 3), dtype=np.float32)
        b = self.Tensor(
            shape=b_np.shape, device=self.Device("cpu"), requires_grad=False, ctx=None
        )
        b.copy_from_numpy(b_np)

        with self.assertRaises(ValueError):
            _ = self.Tensor.concat([a, b], axis=0)


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorConcatCudaBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev = Device("cuda:0")

        lib = Tensor._get_cuda_lib()
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )

        cuda_set_device(lib, 0)

        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        cls.mc = mc
        cls.lib = lib

    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool):
        t = self.Tensor(
            shape=arr.shape, device=self.dev, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        self.mc.cuda_memcpy_h2d(self.lib, int(t.data), arr, int(arr.nbytes))
        return t

    def _cuda_to_numpy(self, t):
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        self.mc.cuda_memcpy_d2h(self.lib, out, int(t.data), int(out.nbytes))
        return out

    def test_concat_cuda_backward_splits_grad_axis0(self) -> None:
        rng = np.random.default_rng(1)
        a_np = rng.standard_normal((2, 4)).astype(np.float32)
        b_np = rng.standard_normal((3, 4)).astype(np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=True)

        out = self.Tensor.concat([a, b], axis=0)
        self.assertTrue(out.requires_grad)

        ctx = getattr(out, "_ctx", None) or getattr(out, "ctx", None)
        self.assertIsNotNone(ctx)

        grad_out = self.Tensor.full(
            out.shape, 1.0, device=self.dev, requires_grad=False
        )

        backward_fn = getattr(ctx, "backward_fn")
        grads = backward_fn(grad_out)

        self.assertEqual(len(grads), 2)
        ga, gb = grads

        self.assertIsNotNone(ga)
        self.assertIsNotNone(gb)

        self.assertEqual(ga.shape, a.shape)
        self.assertEqual(gb.shape, b.shape)

        ga_np = self._cuda_to_numpy(ga)
        gb_np = self._cuda_to_numpy(gb)

        np.testing.assert_allclose(ga_np, np.ones_like(a_np), rtol=0, atol=0)
        np.testing.assert_allclose(gb_np, np.ones_like(b_np), rtol=0, atol=0)

    def test_concat_cuda_backward_grad_out_wrong_device_raises(self) -> None:
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.ones((3, 3), dtype=np.float32)

        a = self._cuda_tensor_from_numpy(a_np, requires_grad=True)
        b = self._cuda_tensor_from_numpy(b_np, requires_grad=True)

        out = self.Tensor.concat([a, b], axis=0)
        ctx = getattr(out, "_ctx", None) or getattr(out, "ctx", None)
        self.assertIsNotNone(ctx)

        grad_out_cpu = self.Tensor.full(
            out.shape, 1.0, device=self.Device("cpu"), requires_grad=False
        )

        backward_fn = getattr(ctx, "backward_fn")
        with self.assertRaises(RuntimeError):
            _ = backward_fn(grad_out_cpu)
