# tests/infrastructure/tensors/cuda/test_tensor_to_cuda_cpu.py
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
class TestTensorToCpuCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.keydnn.infrastructure.tensor._tensor import Tensor
        from src.keydnn.domain.device._device import Device
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )
        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        cls.Tensor = Tensor
        cls.Device = Device
        cls.dev_cpu = Device("cpu")
        cls.dev_cuda = Device("cuda:0")

        cls.lib = Tensor._get_cuda_lib()
        cuda_set_device(cls.lib, 0)

        cls.mc = mc

    # -------------------------
    # helpers
    # -------------------------

    def _cuda_tensor_from_numpy(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> "object":
        Tensor = self.Tensor
        t = Tensor(
            shape=arr.shape, device=self.dev_cuda, requires_grad=requires_grad, ctx=None
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        self.mc.cuda_memcpy_h2d(
            self.lib, dst_dev=int(t.data), src_host=arr, nbytes=int(arr.nbytes)
        )
        return t

    def _cuda_to_numpy(self, t: "object") -> np.ndarray:
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        self.mc.cuda_memcpy_d2h(
            self.lib, dst_host=out, src_dev=int(t.data), nbytes=int(out.nbytes)
        )
        return out

    # -------------------------
    # tests: CPU -> CUDA -> CPU roundtrip
    # -------------------------

    def test_to_cpu_to_cuda_roundtrip_matches_numpy_float32(self) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((3, 4)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=True, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.to(self.dev_cuda)
        self.assertTrue(y.device.is_cuda())
        self.assertEqual(y.shape, x.shape)

        z = y.to(self.dev_cpu)
        self.assertTrue(z.device.is_cpu())
        np.testing.assert_allclose(z.to_numpy(), x_np, rtol=0, atol=0)

    def test_to_cpu_to_cuda_preserves_shape_and_values_float32(self) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 1, 5)).astype(np.float32)

        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.to(self.dev_cuda)
        self.assertEqual(y.shape, x_np.shape)
        self.assertTrue(y.device.is_cuda())

        got = self._cuda_to_numpy(y)
        np.testing.assert_allclose(got, x_np, rtol=0, atol=0)

    def test_to_cpu_to_cuda_requires_grad_is_reset_for_safety(self) -> None:
        # If your `to()` intentionally detaches grads across device moves:
        x_np = np.ones((2, 2), dtype=np.float32)
        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=True, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.to(self.dev_cuda)
        self.assertFalse(y.requires_grad)

    # -------------------------
    # tests: CUDA -> CPU
    # -------------------------

    def test_to_cuda_to_cpu_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((4,)).astype(np.float32)

        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)
        y = x.to(self.dev_cpu)

        self.assertTrue(y.device.is_cpu())
        np.testing.assert_allclose(y.to_numpy(), x_np, rtol=0, atol=0)

    def test_to_cuda_to_cpu_requires_grad_is_reset_for_safety(self) -> None:
        x_np = np.ones((3, 3), dtype=np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        y = x.to(self.dev_cpu)
        self.assertTrue(y.device.is_cpu())
        self.assertFalse(y.requires_grad)

    # -------------------------
    # tests: same-device behavior (copy=False only, since clone() not implemented yet)
    # -------------------------

    def test_to_same_device_copy_false_returns_self_cpu(self) -> None:
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        x = self.Tensor(
            shape=x_np.shape, device=self.dev_cpu, requires_grad=False, ctx=None
        )
        x.copy_from_numpy(x_np)

        y = x.to(self.dev_cpu, copy=False)
        self.assertIs(y, x)

    def test_to_same_device_copy_false_returns_self_cuda(self) -> None:
        x_np = np.arange(8, dtype=np.float32).reshape(2, 4)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=False)

        y = x.to(self.dev_cuda, copy=False)
        self.assertIs(y, x)

    # -------------------------
    # tests: invalid device creation (matches your Device validation)
    # -------------------------

    def test_device_rejects_invalid_string(self) -> None:
        with self.assertRaises(ValueError):
            _ = self.Device("tpu:0")


if __name__ == "__main__":
    unittest.main()
