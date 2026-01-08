# tests/infrastructure/convolution/transpose/test_conv2d_transpose_fn_cuda.py
"""
CUDA unit tests for Conv2dTransposeFn (autograd-enabled transpose conv2d).

These tests mirror the existing CPU-only Conv2dTransposeFn tests, but execute the
function on CUDA tensors to validate:

- Forward correctness vs CPU reference kernel (NumPy) on the same inputs.
- Backward correctness via autograd (loss = out.sum()) vs CPU reference backward,
  under grad_out = ones.
- Bias / no-bias behavior.
- requires_grad propagation.
- Device mismatch rejection (CPU vs CUDA).

Notes
-----
- This suite requires the KeyDNN CUDA native DLL and memcpy symbols to be available.
- We compare CUDA results (copied back to host) against the CPU reference kernels:
  `conv2d_transpose_forward_cpu` / `conv2d_transpose_backward_cpu`.
- Backward on CUDA may use atomicAdd in kernels, so float32 tolerances are looser.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple, Optional

import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.convolution.transpose._conv2d_transpose_function import (
    Conv2dTransposeFn,
)

from src.keydnn.infrastructure.ops.conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)

from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)


# ---------------------------
# CUDA test utilities
# ---------------------------
def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.
    Mirrors helpers used in other CUDA ext tests.
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
            "Unable to construct a CUDA Device; update _get_cuda_device() for this repo."
        ) from e


def _bind_memcpy_symbols(lib: ctypes.CDLL) -> Tuple[ctypes._CFuncPtr, ctypes._CFuncPtr]:
    """
    Bind keydnn_cuda_memcpy_h2d and keydnn_cuda_memcpy_d2h from the CUDA DLL.
    """
    if not hasattr(lib, "keydnn_cuda_memcpy_h2d"):
        raise AttributeError("CUDA DLL missing symbol: keydnn_cuda_memcpy_h2d")
    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        raise AttributeError("CUDA DLL missing symbol: keydnn_cuda_memcpy_d2h")

    h2d = lib.keydnn_cuda_memcpy_h2d
    h2d.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    h2d.restype = ctypes.c_int

    d2h = lib.keydnn_cuda_memcpy_d2h
    d2h.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    d2h.restype = ctypes.c_int

    return h2d, d2h


def _h2d(lib: ctypes.CDLL, dst_dev: int, src: np.ndarray) -> None:
    src_c = np.ascontiguousarray(src)
    h2d, _ = _bind_memcpy_symbols(lib)
    st = int(
        h2d(
            ctypes.c_uint64(int(dst_dev)),
            ctypes.c_void_p(src_c.ctypes.data),
            ctypes.c_size_t(src_c.nbytes),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")


def _d2h(lib: ctypes.CDLL, src_dev: int, dst: np.ndarray) -> None:
    dst_c = np.ascontiguousarray(dst)
    _, d2h = _bind_memcpy_symbols(lib)
    st = int(
        d2h(
            ctypes.c_void_p(dst_c.ctypes.data),
            ctypes.c_uint64(int(src_dev)),
            ctypes.c_size_t(dst_c.nbytes),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _tol_cuda(dtype: np.dtype) -> tuple[float, float]:
    """
    Tolerances for comparing CUDA autograd path vs CPU reference kernels.

    Notes
    -----
    - float32 CUDA backward often uses atomics => accumulation order differs.
    - float64 should be close (but keep a small tol).
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return 3e-4, 3e-4
    if dtype == np.float64:
        return 1e-10, 1e-10
    return 3e-4, 3e-4


class TestConv2dTransposeFnCuda(unittest.TestCase):
    """
    CUDA tests for Conv2dTransposeFn.

    These tests exercise the device-dispatch path inside Conv2dTransposeFn, which
    calls conv2d_transpose_{forward,backward}_cuda_tensor via the CUDA ext boundary.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Load CUDA DLL; skip suite if unavailable.
        try:
            cls.lib = _load_cuda_lib()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL not available: {e}") from e

        cls.device_index = 0
        try:
            cuda_set_device(cls.lib, cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to set CUDA device: {e}") from e

        # Ensure memcpy symbols exist; if not, skip (cannot validate numerics).
        try:
            _bind_memcpy_symbols(cls.lib)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA memcpy symbols unavailable: {e}") from e

        # Construct CUDA device object
        try:
            cls.cuda_device = _get_cuda_device(cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to construct CUDA Device: {e}") from e

    def _make_cuda_tensor_from_host(
        self, x: np.ndarray, *, requires_grad: bool
    ) -> Tensor:
        x_c = np.ascontiguousarray(x)
        nbytes = int(x_c.nbytes)
        dev_ptr = int(cuda_malloc(self.lib, nbytes if nbytes > 0 else 1))
        try:
            if nbytes > 0:
                _h2d(self.lib, dev_ptr, x_c)
        except Exception:
            cuda_free(self.lib, dev_ptr)
            raise

        # Use the same construction style as other CUDA ext tests
        return Tensor._from_devptr(
            dev_ptr,
            shape=tuple(int(d) for d in x_c.shape),
            dtype=x_c.dtype,
            device=self.cuda_device,
            requires_grad=requires_grad,
        )

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        nbytes = int(host.nbytes)
        if nbytes > 0:
            _d2h(self.lib, int(t.data), host)
        return host

    def test_forward_matches_cpu_kernel_f32(self) -> None:
        rng = np.random.default_rng(0)
        dtype = np.float32

        x_np = rng.standard_normal((2, 3, 5, 4), dtype=dtype)
        w_np = rng.standard_normal((3, 4, 3, 2), dtype=dtype)  # (C_in, C_out, Kh, Kw)
        b_np = rng.standard_normal((4,), dtype=dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        stride = (2, 1)
        padding = (1, 0)
        output_padding = (0, 0)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol_cuda(dtype)
        self.assertEqual(y_host.shape, y_ref.shape)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def test_forward_no_bias_matches_cpu_kernel_f32(self) -> None:
        rng = np.random.default_rng(7)
        dtype = np.float32

        x_np = rng.standard_normal((1, 2, 4, 3), dtype=dtype)
        w_np = rng.standard_normal((2, 3, 2, 3), dtype=dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)

        stride = (1, 2)
        padding = (1, 1)
        output_padding = (0, 0)

        ctx = Context(parents=(x, w), backward_fn=lambda _: (None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol_cuda(dtype)
        self.assertEqual(y_host.shape, y_ref.shape)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def test_forward_matches_cpu_kernel_f64(self) -> None:
        rng = np.random.default_rng(2)
        dtype = np.float64

        x_np = rng.standard_normal((1, 2, 5, 4)).astype(dtype)
        w_np = rng.standard_normal((2, 3, 3, 2)).astype(dtype)  # (C_in, C_out, Kh, Kw)
        b_np = rng.standard_normal((3,)).astype(dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol_cuda(dtype)
        self.assertEqual(y_host.shape, y_ref.shape)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def test_backward_via_autograd_matches_cpu_kernel_f32(self) -> None:
        """
        out = conv2d_transpose(x,w,b); loss = out.sum(); loss.backward()
        Compare x.grad/w.grad/b.grad with conv2d_transpose_backward_cpu under grad_out=ones.
        """
        rng = np.random.default_rng(1)
        dtype = np.float32

        x_np = rng.standard_normal((1, 2, 4, 3), dtype=dtype)
        w_np = rng.standard_normal((2, 3, 2, 2), dtype=dtype)
        b_np = rng.standard_normal((3,), dtype=dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=True)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=True)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=True)

        stride = (2, 1)
        padding = (0, 1)
        output_padding = (0, 0)

        ctx = Context(
            parents=(x, w, b),
            backward_fn=lambda grad_out: Conv2dTransposeFn.backward(ctx, grad_out),
        )
        out = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        out.requires_grad = True
        out._set_ctx(ctx)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)

        # Reference grads with grad_out=ones (matching loss=sum)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        grad_out_np = np.ones_like(y_ref, dtype=dtype)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            b_np,
            grad_out_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        gx = self._read_cuda_tensor_to_host(x.grad)  # type: ignore[arg-type]
        gw = self._read_cuda_tensor_to_host(w.grad)  # type: ignore[arg-type]
        gb = self._read_cuda_tensor_to_host(b.grad)  # type: ignore[arg-type]

        rtol, atol = _tol_cuda(dtype)
        np.testing.assert_allclose(gx, gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gw, gw_ref, rtol=rtol, atol=atol)

        # Bias grad is a reduction; keep modestly tighter
        np.testing.assert_allclose(gb, gb_ref, rtol=1e-4, atol=1e-4)

    def test_backward_none_bias_returns_none(self) -> None:
        rng = np.random.default_rng(3)
        dtype = np.float32

        x_np = rng.standard_normal((2, 2, 3, 3), dtype=dtype)
        w_np = rng.standard_normal((2, 1, 3, 2), dtype=dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=True)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=True)

        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        ctx = Context(
            parents=(x, w),
            backward_fn=lambda grad_out: Conv2dTransposeFn.backward(ctx, grad_out),
        )
        out = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        out.requires_grad = True
        out._set_ctx(ctx)

        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        # no bias => Conv2dTransposeFn.backward returns (grad_x, grad_w) only

    def test_forward_respects_out_requires_grad_flag_cuda(self) -> None:
        rng = np.random.default_rng(11)
        dtype = np.float32

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=dtype)
        w_np = rng.standard_normal((1, 2, 2, 2), dtype=dtype)
        b_np = rng.standard_normal((2,), dtype=dtype)

        # Case A: parents do not require grad -> output should not require grad
        x0 = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w0 = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b0 = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        ctx0 = Context(parents=(x0, w0, b0), backward_fn=lambda _: (None, None, None))
        y0 = Conv2dTransposeFn.forward(
            ctx0, x0, w0, b0, stride=1, padding=0, output_padding=0
        )
        self.assertFalse(y0.requires_grad)

        # Case B: one parent requires grad -> output should require grad
        x1 = self._make_cuda_tensor_from_host(x_np, requires_grad=True)
        w1 = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b1 = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        ctx1 = Context(parents=(x1, w1, b1), backward_fn=lambda _: (None, None, None))
        y1 = Conv2dTransposeFn.forward(
            ctx1, x1, w1, b1, stride=1, padding=0, output_padding=0
        )
        self.assertTrue(y1.requires_grad)

    def test_forward_rejects_mismatched_devices_cpu_vs_cuda(self) -> None:
        rng = np.random.default_rng(0)
        dtype = np.float32

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=dtype)
        w_np = rng.standard_normal((1, 1, 2, 2), dtype=dtype)

        # x on CUDA, w on CPU -> should raise
        x_cuda = self._make_cuda_tensor_from_host(x_np, requires_grad=False)

        w_cpu = Tensor(
            shape=w_np.shape, device=Device("cpu"), requires_grad=False, ctx=None
        )
        w_cpu.copy_from_numpy(w_np)

        ctx = Context(parents=(x_cuda, w_cpu), backward_fn=lambda _: (None, None))
        with self.assertRaises(RuntimeError):
            _ = Conv2dTransposeFn.forward(ctx, x_cuda, w_cpu, None, stride=1, padding=0)

    def test_forward_output_padding_nonzero_cuda(self) -> None:
        """
        Ensures CUDA dispatch path supports output_padding != 0 (if kernels support it).
        Compares against CPU reference kernel.
        """
        rng = np.random.default_rng(4)
        dtype = np.float32

        x_np = rng.standard_normal((1, 1, 3, 3), dtype=dtype)
        w_np = rng.standard_normal((1, 2, 3, 3), dtype=dtype)
        b_np = rng.standard_normal((2,), dtype=dtype)

        x = self._make_cuda_tensor_from_host(x_np, requires_grad=False)
        w = self._make_cuda_tensor_from_host(w_np, requires_grad=False)
        b = self._make_cuda_tensor_from_host(b_np, requires_grad=False)

        stride = (2, 2)
        padding = (1, 1)
        output_padding = (1, 0)

        ctx = Context(parents=(x, w, b), backward_fn=lambda _: (None, None, None))
        y = Conv2dTransposeFn.forward(
            ctx,
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        y_host = self._read_cuda_tensor_to_host(y)
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        rtol, atol = _tol_cuda(dtype)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
