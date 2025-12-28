# tests/infrastructure/tensors/cuda/test_tensor_sum_cuda.py
from __future__ import annotations

import unittest
from typing import Any, Optional, Tuple

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

        # Dedicated memcpy ops module (GetProcAddress-based)
        from src.keydnn.infrastructure.native_cuda.python.ops import (  # type: ignore
            memcpy_ctypes as mc,
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
        # scalar shape () is supported by numpy empty; nbytes matches dtype.itemsize
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

    def _ctx_of(self, t) -> Any:
        return getattr(t, "_ctx", None) or getattr(t, "ctx", None)


# -----------------------------------------------------------------------------
# Forward tests
# -----------------------------------------------------------------------------


class TestTensorSumCudaForward(_CudaTestBase):
    def test_sum_all_cuda_forward_scalar_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((257,)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum()

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), ())
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y).astype(np.float32, copy=False)
        ref = np.array(x_np.sum(), dtype=np.float32)

        # float32 reduction order may differ (atomicAdd)
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=5e-6)

    def test_sum_all_cuda_forward_keepdims_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((4, 7)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum(axis=None, keepdims=True)

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (1, 1))
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = np.sum(x_np, axis=None, keepdims=True).astype(np.float32, copy=False)

        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=5e-6)

    def test_sum_axis0_cuda_forward_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((19, 29)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum(axis=0, keepdims=False)

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (29,))
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = np.sum(x_np, axis=0, keepdims=False).astype(np.float32, copy=False)

        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=5e-6)

    def test_sum_axis1_cuda_forward_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((17, 31)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum(axis=1, keepdims=False)

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (17,))
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = np.sum(x_np, axis=1, keepdims=False).astype(np.float32, copy=False)

        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=5e-6)

    def test_sum_axis_negative_cuda_forward_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal((8, 9)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.sum(axis=-1, keepdims=False)

        self.assertTrue(y.device.is_cuda())
        self.assertEqual(tuple(y.shape), (8,))
        self.assertNotEqual(int(y.data), 0)

        got = self._cuda_to_numpy(y)
        ref = np.sum(x_np, axis=-1, keepdims=False).astype(np.float32, copy=False)

        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=5e-6)

    def test_sum_axis_keepdims_cuda_forward_matches_numpy(self) -> None:
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((6, 10)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y0 = x.sum(axis=0, keepdims=True)
        self.assertEqual(tuple(y0.shape), (1, 10))
        got0 = self._cuda_to_numpy(y0)
        ref0 = np.sum(x_np, axis=0, keepdims=True).astype(np.float32, copy=False)
        np.testing.assert_allclose(got0, ref0, rtol=2e-5, atol=5e-6)

        y1 = x.sum(axis=1, keepdims=True)
        self.assertEqual(tuple(y1.shape), (6, 1))
        got1 = self._cuda_to_numpy(y1)
        ref1 = np.sum(x_np, axis=1, keepdims=True).astype(np.float32, copy=False)
        np.testing.assert_allclose(got1, ref1, rtol=2e-5, atol=5e-6)

    def test_sum_axis_non_2d_cuda_raises(self) -> None:
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        with self.assertRaises(NotImplementedError):
            _ = x.sum(axis=1)

    def test_sum_axis_out_of_bounds_raises(self) -> None:
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((2, 3)).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = x.sum(axis=2)


# -----------------------------------------------------------------------------
# Backward tests
# -----------------------------------------------------------------------------


class TestTensorSumCudaBackward(_CudaTestBase):
    def _run_backward_case(
        self,
        *,
        x_np: np.ndarray,
        axis: Optional[int],
        keepdims: bool,
        grad_out_value: float,
    ) -> None:
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)
        y = x.sum(axis=axis, keepdims=keepdims)

        # Some implementations track autograd via ctx even if requires_grad flag
        # on the output is not propagated. Accept either behavior.
        ctx = self._ctx_of(y)
        self.assertTrue(
            bool(getattr(y, "requires_grad", False)) or (ctx is not None),
            "Expected CUDA Tensor.sum output to participate in autograd "
            "(requires_grad=True or ctx attached) when input requires_grad=True.",
        )
        self.assertIsNotNone(ctx, "Expected Tensor.sum to attach a backward Context")

        # grad_out on CUDA (shape matches output)
        grad_out = self.Tensor.full(
            y.shape, float(grad_out_value), device=self.dev, requires_grad=False
        )

        backward_fn = getattr(ctx, "backward_fn")
        (gx,) = backward_fn(grad_out)

        self.assertIsNotNone(gx)
        self.assertTrue(gx.device.is_cuda())
        self.assertEqual(tuple(gx.shape), tuple(x.shape))

        gx_np = self._cuda_to_numpy(gx)

        # reference
        if axis is None:
            ref = np.ones_like(x_np, dtype=np.float32) * np.float32(grad_out_value)
        else:
            ndim = x_np.ndim
            axis_ = axis if axis >= 0 else ndim + axis
            go = np.full(
                np.sum(x_np, axis=axis_, keepdims=keepdims).shape,
                grad_out_value,
                dtype=np.float32,
            )
            if not keepdims:
                go = np.expand_dims(go, axis=axis_)
            ref = np.ones_like(x_np, dtype=np.float32) * go

        np.testing.assert_allclose(gx_np, ref, rtol=2e-5, atol=5e-6)

    def test_sum_all_cuda_backward_broadcasts_scalar(self) -> None:
        rng = np.random.default_rng(10)
        x_np = rng.standard_normal((3, 5)).astype(np.float32)
        self._run_backward_case(
            x_np=x_np, axis=None, keepdims=False, grad_out_value=1.0
        )

    def test_sum_all_cuda_backward_keepdims_broadcasts(self) -> None:
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((4, 7)).astype(np.float32)
        self._run_backward_case(
            x_np=x_np, axis=None, keepdims=True, grad_out_value=-2.25
        )

    def test_sum_axis0_cuda_backward_broadcasts(self) -> None:
        rng = np.random.default_rng(12)
        x_np = rng.standard_normal((19, 29)).astype(np.float32)
        self._run_backward_case(x_np=x_np, axis=0, keepdims=False, grad_out_value=0.75)

    def test_sum_axis1_cuda_backward_broadcasts(self) -> None:
        rng = np.random.default_rng(13)
        x_np = rng.standard_normal((17, 31)).astype(np.float32)
        self._run_backward_case(x_np=x_np, axis=1, keepdims=False, grad_out_value=1.0)

    def test_sum_axis_keepdims_cuda_backward_broadcasts(self) -> None:
        rng = np.random.default_rng(14)
        x_np = rng.standard_normal((6, 10)).astype(np.float32)
        self._run_backward_case(x_np=x_np, axis=0, keepdims=True, grad_out_value=1.5)
        self._run_backward_case(x_np=x_np, axis=1, keepdims=True, grad_out_value=-0.5)

    def test_sum_cuda_backward_grad_out_wrong_device_raises(self) -> None:
        x_np = np.full((2, 3), 1.0, dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)
        y = x.sum(axis=None, keepdims=False)

        ctx = self._ctx_of(y)
        self.assertIsNotNone(ctx)

        grad_out_cpu = self.Tensor.full(
            y.shape, 1.0, device=self.Device("cpu"), requires_grad=False
        )

        backward_fn = getattr(ctx, "backward_fn")
        with self.assertRaises(RuntimeError):
            _ = backward_fn(grad_out_cpu)


if __name__ == "__main__":
    unittest.main()
