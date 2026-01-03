from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.layers._dropout import Dropout
from src.keydnn.infrastructure.tensor._tensor import Tensor


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
class TestDropoutCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = Device("cuda:0")

        # Ensure device is set once
        lib = Tensor._get_cuda_lib()
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            cuda_set_device,  # type: ignore
        )

        cuda_set_device(lib, 0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _cuda_tensor_from_numpy(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))

        t = Tensor(
            shape=arr.shape,
            device=self.device,
            requires_grad=requires_grad,
            ctx=None,
        )
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
        self.assertNotEqual(int(t.data), 0)

        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        mc.memcpy_htod(
            Tensor._get_cuda_lib(),
            dst_dev=int(t.data),
            src_host=arr,
            nbytes=int(arr.nbytes),
            sync=True,
        )
        return t

    def _cuda_to_numpy(self, t: Tensor) -> np.ndarray:
        out = np.empty(t.shape, dtype=np.dtype(t.dtype))
        out = np.ascontiguousarray(out)

        from src.keydnn.infrastructure.native_cuda.python.ops import (
            memcpy_ctypes as mc,  # type: ignore
        )

        mc.memcpy_dtoh(
            Tensor._get_cuda_lib(),
            dst_host=out,
            src_dev=int(t.data),
            nbytes=int(out.nbytes),
            sync=True,
        )
        return out

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_eval_mode_is_identity_cuda(self) -> None:
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        d = Dropout(p=0.6)
        d.training = False

        y = d(x)

        # eval mode returns input tensor directly
        self.assertIs(y, x)

        y_np = self._cuda_to_numpy(y)
        np.testing.assert_allclose(y_np, x_np, rtol=0.0, atol=0.0)

    def test_p_zero_is_noop_in_train_cuda(self) -> None:
        x_np = np.random.randn(6, 6).astype(np.float32)
        x = self._cuda_tensor_from_numpy(x_np, requires_grad=True)

        d = Dropout(p=0.0)
        d.training = True

        y = d(x)

        self.assertIs(y, x)

        y_np = self._cuda_to_numpy(y)
        np.testing.assert_allclose(y_np, x_np, rtol=0.0, atol=0.0)

    def test_forward_mask_and_scaling_cuda(self) -> None:
        p = 0.5
        scale = 1.0 / (1.0 - p)

        x = self._cuda_tensor_from_numpy(
            np.ones((128, 128), dtype=np.float32),
            requires_grad=False,
        )

        d = Dropout(p=p)
        d.training = True

        np.random.seed(123)
        y = d(x)

        y_np = self._cuda_to_numpy(y)

        uniq = np.unique(y_np)
        for val in uniq:
            ok = np.isclose(val, 0.0) or np.isclose(val, scale)
            self.assertTrue(ok, msg=f"Unexpected value {val}")

        # statistically both should appear
        self.assertTrue(np.any(np.isclose(y_np, 0.0)))
        self.assertTrue(np.any(np.isclose(y_np, scale)))

    def test_backward_masked_and_scaled_cuda(self) -> None:
        p = 0.4
        scale = 1.0 / (1.0 - p)

        x = self._cuda_tensor_from_numpy(
            np.ones((64, 64), dtype=np.float32),
            requires_grad=True,
        )

        d = Dropout(p=p)
        d.training = True

        np.random.seed(99)
        y = d(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.device.is_cuda())

        g = self._cuda_to_numpy(x.grad)

        uniq = np.unique(g)
        for val in uniq:
            ok = np.isclose(val, 0.0) or np.isclose(val, scale)
            self.assertTrue(
                ok, msg=f"Unexpected grad value {val}; expected 0 or {scale}"
            )

        self.assertTrue(np.any(np.isclose(g, 0.0)))
        self.assertTrue(np.any(np.isclose(g, scale)))

    def test_expected_mean_close_to_one_cuda(self) -> None:
        p = 0.25
        x = self._cuda_tensor_from_numpy(
            np.ones((256, 256), dtype=np.float32),
            requires_grad=False,
        )

        d = Dropout(p=p)
        d.training = True

        np.random.seed(2024)
        y = d(x)

        y_np = self._cuda_to_numpy(y)
        self.assertTrue(abs(float(y_np.mean()) - 1.0) < 0.03)
