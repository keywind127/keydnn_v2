# tests/infrastructure/tensors/cuda/test_tensor_exp_cuda.py
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
class _CudaTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev = Device("cuda:0")

        # Load CUDA lib and set device
        lib = Tensor._get_cuda_lib()
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )

        cuda_set_device(lib, 0)

        # Use the dedicated memcpy ops module (GetProcAddress-based)
        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        cls.lib = lib
        cls.mc = mc

    # -------------------------
    # helpers: cuda <-> numpy
    # -------------------------

    def _cuda_tensor_from_numpy(self, arr: np.ndarray, *, requires_grad: bool):
        Tensor = self.Tensor
        t = Tensor(
            shape=arr.shape, device=self.dev, requires_grad=requires_grad, ctx=None
        )

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        # Ensure device allocation and upload
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
        self.assertNotEqual(int(t.data), 0)

        self.mc.cuda_memcpy_h2d(
            self.lib,
            dst_dev=int(t.data),
            src_host=arr,
            nbytes=int(arr.nbytes),
        )
        return t

    def _cuda_to_numpy(self, t) -> np.ndarray:
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        self.mc.cuda_memcpy_d2h(
            self.lib,
            dst_host=out,
            src_dev=int(t.data),
            nbytes=int(out.nbytes),
        )
        return out


class TestTensorExpCudaForward(_CudaTestBase):
    def test_exp_cuda_forward_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        # keep values moderate to avoid inf
        x_np = rng.uniform(low=-3.0, high=3.0, size=(4, 7)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.exp()

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(y.shape, x.shape)
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        exp = np.exp(x_np).astype(np.float32, copy=False)

        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)

    def test_exp_cuda_dtype_and_shape_preserved(self) -> None:
        x_np = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.exp()
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(np.dtype(y.dtype), np.dtype(np.float32))


class TestTensorExpCudaBackward(_CudaTestBase):
    def test_exp_cuda_backward_matches_exp(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.uniform(low=-4.0, high=4.0, size=(3, 5)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)
        y = x.exp()
        self.assertTrue(y.requires_grad)

        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)

        # grad_out = ones on CUDA
        grad_out = self.Tensor.full(y.shape, 1.0, device=self.dev, requires_grad=False)

        backward_fn = getattr(ctx, "backward_fn")
        (gx,) = backward_fn(grad_out)

        self.assertIsNotNone(gx)
        self.assertTrue(gx.device.is_cuda())
        self.assertEqual(gx.shape, x.shape)

        gx_np = self._cuda_to_numpy(gx)
        exp = np.exp(x_np).astype(np.float32, copy=False)

        np.testing.assert_allclose(gx_np, exp, rtol=1e-6, atol=1e-6)

    def test_exp_cuda_backward_grad_out_wrong_device_raises(self) -> None:
        x_np = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 2.0]], dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.exp()
        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)

        grad_out_cpu = self.Tensor.full(
            y.shape, 1.0, device=self.Device("cpu"), requires_grad=False
        )

        backward_fn = getattr(ctx, "backward_fn")
        with self.assertRaises(RuntimeError):
            _ = backward_fn(grad_out_cpu)
