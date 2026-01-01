# tests/infrastructure/conv2d/test_conv2d_fn_cuda.py
"""
Unit tests for Conv2dFn on CUDA devices.

These tests validate the *autograd-level* Conv2dFn dispatches to the CUDA
boundary (conv2d_cuda_ext) when inputs are CUDA tensors, and that forward/backward
numerics match the CPU reference kernels (conv2d_forward_cpu/conv2d_backward_cpu).

Notes
-----
- Skips entirely if the KeyDNN native CUDA DLL / wrappers are not available.
- Uses explicit device allocations + memcpy to construct CUDA tensors with
  valid devptrs (Tensor._from_devptr).
- Backward test calls Conv2dFn.backward(...) directly with a CUDA grad_out tensor
  of ones (does not rely on CUDA sum/reduce availability).
"""

from __future__ import annotations

import unittest
from typing import Optional, Tuple, List

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.infrastructure.convolution._conv2d_function import Conv2dFn
from src.keydnn.infrastructure.ops.conv2d_cpu import (
    conv2d_forward_cpu,
    conv2d_backward_cpu,
)

from tests.infrastructure.ops._cuda_test_utils import try_get_cuda_env


def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to construct a CUDA Device for this repo.
    """
    if hasattr(Device, "cuda") and callable(getattr(Device, "cuda")):
        return Device.cuda(index)  # type: ignore[attr-defined]

    try:
        return Device("cuda", index)  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(f"cuda:{index}")  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(kind="cuda", index=index)  # type: ignore[call-arg]
    except Exception as e:
        raise RuntimeError(
            "Unable to construct a CUDA Device; update _get_cuda_device()."
        ) from e


class _CudaAllocs:
    """
    Simple devptr allocation tracker to prevent leaks in unit tests.
    """

    def __init__(self, env):
        self.env = env
        self.ptrs: List[int] = []

    def malloc(self, nbytes: int) -> int:
        p = int(self.env.cuda_malloc(self.env.lib, int(nbytes)))
        if p == 0 and int(nbytes) != 0:
            raise RuntimeError("cuda_malloc returned 0 for non-zero allocation")
        self.ptrs.append(p)
        return p

    def free_all(self) -> None:
        for p in reversed(self.ptrs):
            try:
                if int(p) != 0:
                    self.env.cuda_free(self.env.lib, int(p))
            except Exception:
                pass
        self.ptrs.clear()


class TestConv2dFnCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env
        cls.device_index = 0
        cls.cuda_device = _get_cuda_device(cls.device_index)

    def setUp(self) -> None:
        self.allocs = _CudaAllocs(self.env)

    def tearDown(self) -> None:
        self.allocs.free_all()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _make_cuda_tensor_from_host(
        self, arr: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        arr_c = np.ascontiguousarray(arr)
        dev_ptr = self.allocs.malloc(int(arr_c.nbytes))
        if int(arr_c.nbytes) > 0:
            self.env.cudaMemcpyHtoD(
                self.env.lib, int(dev_ptr), arr_c, int(arr_c.nbytes)
            )
            self.env.cuda_synchronize(self.env.lib)

        return Tensor._from_devptr(
            dev_ptr=int(dev_ptr),
            shape=tuple(int(d) for d in arr_c.shape),
            device=self.cuda_device,
            requires_grad=bool(requires_grad),
            ctx=None,
            dtype=np.dtype(arr_c.dtype),
        )

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        nbytes = int(host.nbytes)
        if nbytes > 0:
            src = int(t.data)
            if src == 0:
                raise RuntimeError("CUDA tensor has data==0 in test readback")
            self.env.cudaMemcpyDtoH(self.env.lib, host, int(src), nbytes)
            self.env.cuda_synchronize(self.env.lib)
        return host

    # -----------------------------
    # Tests
    # -----------------------------
    def test_forward_matches_cpu(self) -> None:
        rng = np.random.default_rng(0)

        x_np = rng.standard_normal((2, 3, 6, 5), dtype=np.float32)
        w_np = rng.standard_normal((4, 3, 3, 3), dtype=np.float32)
        b_np = rng.standard_normal((4,), dtype=np.float32)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        stride = (2, 1)
        padding = (1, 2)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dFn.forward(ctx, x, w, b, stride=stride, padding=padding)

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=stride, padding=padding)

        self.assertTrue(y.device.is_cuda(), "output should be CUDA")
        self.assertEqual(tuple(y.shape), tuple(y_ref.shape))
        np.testing.assert_allclose(y_host, y_ref, rtol=1e-4, atol=1e-4)

    def test_backward_matches_cpu_under_grad_out_ones(self) -> None:
        """
        Directly validate Conv2dFn.backward on CUDA by feeding grad_out=ones (on CUDA),
        and comparing returned (gx, gw, gb) to conv2d_backward_cpu reference.
        """
        rng = np.random.default_rng(1)

        x_np = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
        w_np = rng.standard_normal((3, 2, 3, 2), dtype=np.float32)
        b_np = rng.standard_normal((3,), dtype=np.float32)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=True)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=True)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=True)

        stride = (1, 2)
        padding = (1, 0)

        # Forward to get output shape
        ctx = Context(
            parents=(x, w, b),
            backward_fn=lambda grad_out: Conv2dFn.backward(ctx, grad_out),  # type: ignore[misc]
        )
        out = Conv2dFn.forward(ctx, x, w, b, stride=stride, padding=padding)

        # grad_out = ones on CUDA
        out_host = self._read_cuda_tensor_to_host(out)
        grad_out_np = np.ones_like(out_host, dtype=np.float32)
        grad_out = self._make_cuda_tensor_from_host(grad_out_np, requires_grad=False)

        gx_t, gw_t, gb_t = Conv2dFn.backward(ctx, grad_out)  # type: ignore[assignment]

        self.assertIsNotNone(gx_t)
        self.assertIsNotNone(gw_t)
        self.assertIsNotNone(gb_t)

        gx = self._read_cuda_tensor_to_host(gx_t)  # type: ignore[arg-type]
        gw = self._read_cuda_tensor_to_host(gw_t)  # type: ignore[arg-type]
        gb = self._read_cuda_tensor_to_host(gb_t)  # type: ignore[arg-type]

        gx_ref, gw_ref, gb_ref = conv2d_backward_cpu(
            x_np, w_np, b_np, grad_out_np, stride=stride, padding=padding
        )

        np.testing.assert_allclose(gx, gx_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gw, gw_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gb, gb_ref, rtol=1e-5, atol=1e-5)

    def test_forward_no_bias_matches_cpu(self) -> None:
        rng = np.random.default_rng(2)

        x_np = rng.standard_normal((1, 2, 7, 7), dtype=np.float32)
        w_np = rng.standard_normal((3, 2, 3, 3), dtype=np.float32)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)

        stride = (1, 1)
        padding = (1, 1)

        ctx = Context(parents=(x, w), backward_fn=lambda _: (None, None))
        y = Conv2dFn.forward(ctx, x, w, None, stride=stride, padding=padding)

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_forward_cpu(x_np, w_np, None, stride=stride, padding=padding)

        self.assertTrue(y.device.is_cuda(), "output should be CUDA")
        self.assertEqual(tuple(y.shape), tuple(y_ref.shape))
        np.testing.assert_allclose(y_host, y_ref, rtol=1e-4, atol=1e-4)

    def test_raises_on_device_mismatch(self) -> None:
        """
        x on cuda, w on cpu should raise (no implicit device copies in Conv2dFn).
        """
        rng = np.random.default_rng(3)

        x_np = rng.standard_normal((1, 1, 5, 5), dtype=np.float32)
        w_np = rng.standard_normal((2, 1, 3, 3), dtype=np.float32)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)

        w_cpu = Tensor(
            shape=w_np.shape,
            device=Device("cpu"),
            requires_grad=False,
            ctx=None,
            dtype=np.float32,
        )
        w_cpu.copy_from_numpy(w_np)

        ctx = Context(parents=(x, w_cpu), backward_fn=lambda _: (None, None))

        with self.assertRaises(RuntimeError):
            _ = Conv2dFn.forward(ctx, x, w_cpu, None, stride=(1, 1), padding=(0, 0))


if __name__ == "__main__":
    unittest.main()
