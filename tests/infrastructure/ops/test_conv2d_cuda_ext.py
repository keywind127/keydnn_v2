# tests/infrastructure/ops/test_conv2d_cuda_ext.py
"""
Unit tests for CUDA Tensor-boundary conv2d wrapper (conv2d_cuda_ext.py).

These tests validate that the Tensor-facing wrapper:
- Accepts CUDA Tensors (device-pointer backed)
- Allocates output device memory
- Calls the underlying CUDA ops wrapper / kernels
- Returns CUDA Tensors with correct shape/dtype
- Produces numerically correct results vs NumPy reference
- Raises appropriate errors for invalid inputs

The tests are written with unittest and skip on CPU-only environments where the
KeyDNN CUDA native DLL cannot be loaded.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple, Optional

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)

# wrapper under test
from src.keydnn.infrastructure.ops.conv2d_cuda_ext import (
    conv2d_forward_cuda_tensor,
    conv2d_backward_cuda_tensor,
)


def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.
    Mirrors the helper used in matmul_cuda_ext tests.
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


def _conv2d_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    *,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
) -> np.ndarray:
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for co in range(C_out):
            bias = b[co] if b is not None else x.dtype.type(0)
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    acc = x.dtype.type(0)
                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                acc += (
                                    x_pad[n, ci, h0 + kh, w0 + kw] * w[co, ci, kh, kw]
                                )
                    y[n, co, i, j] = acc + bias
    return y


def _conv2d_backward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    grad_out: np.ndarray,
    *,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    N2, C_out2, H_out, W_out = grad_out.shape
    if N2 != N or C_out2 != C_out:
        raise ValueError("grad_out shape mismatch")

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0.0,
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    grad_x_pad = np.zeros((N, C_in, H_pad, W_pad), dtype=x.dtype)
    grad_w = np.zeros_like(w)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    go = grad_out[n, co, i, j]
                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                grad_w[co, ci, kh, kw] += (
                                    go * x_pad[n, ci, h0 + kh, w0 + kw]
                                )
                                grad_x_pad[n, ci, h0 + kh, w0 + kw] += (
                                    go * w[co, ci, kh, kw]
                                )

    grad_x = grad_x_pad[:, :, p_h : p_h + H, p_w : p_w + W]

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    return grad_x, grad_w, grad_b


class TestConv2dCudaExt(unittest.TestCase):
    """
    Tests for Tensor-boundary CUDA conv2d wrapper.

    Wrappers under test:
        keydnn.infrastructure.ops.conv2d_cuda_ext.conv2d_forward_cuda_tensor
        keydnn.infrastructure.ops.conv2d_cuda_ext.conv2d_backward_cuda_tensor
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Try to load CUDA DLL; skip all tests if unavailable.
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

        # Construct a CUDA Device object for Tensor wrapping.
        try:
            cls.cuda_device = _get_cuda_device(cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to construct CUDA Device: {e}") from e

    def _make_cuda_tensor_from_host(self, x: np.ndarray) -> Tensor:
        x_c = np.ascontiguousarray(x)
        dev_ptr = int(cuda_malloc(self.lib, int(x_c.nbytes)))
        try:
            _h2d(self.lib, dev_ptr, x_c)
        except Exception:
            cuda_free(self.lib, dev_ptr)
            raise

        return Tensor._from_devptr(
            dev_ptr,
            shape=tuple(int(d) for d in x_c.shape),
            dtype=x_c.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        _d2h(self.lib, int(t.data), host)
        return host

    def test_conv2d_forward_f32_no_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        dtype = np.float32

        N, C_in, H, W = 2, 3, 6, 5
        C_out, K_h, K_w = 4, 3, 2
        stride = (1, 2)
        padding = (1, 0)

        x = rng.standard_normal((N, C_in, H, W), dtype=dtype)
        w = rng.standard_normal((C_out, C_in, K_h, K_w), dtype=dtype)
        b = None

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        y_t = conv2d_forward_cuda_tensor(
            x_t,
            w_t,
            b=None,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_forward_f32_with_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        dtype = np.float32

        N, C_in, H, W = 2, 2, 6, 6
        C_out, K_h, K_w = 3, 3, 3
        stride = (1, 1)
        padding = (1, 1)

        x = rng.standard_normal((N, C_in, H, W), dtype=dtype)
        w = rng.standard_normal((C_out, C_in, K_h, K_w), dtype=dtype)
        b = rng.standard_normal((C_out,), dtype=dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = conv2d_forward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_forward_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        dtype = np.float64

        N, C_in, H, W = 1, 2, 7, 6
        C_out, K_h, K_w = 2, 3, 2
        stride = (2, 1)
        padding = (1, 0)

        x = rng.standard_normal((N, C_in, H, W)).astype(dtype)
        w = rng.standard_normal((C_out, C_in, K_h, K_w)).astype(dtype)
        b = rng.standard_normal((C_out,)).astype(dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = conv2d_forward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            stride=stride,
            padding=padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        np.testing.assert_allclose(y, ref, rtol=1e-10, atol=1e-10)

    def test_conv2d_backward_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        dtype = np.float32

        N, C_in, H, W = 2, 2, 6, 5
        C_out, K_h, K_w = 3, 3, 3
        stride = (1, 1)
        padding = (1, 1)

        x = rng.standard_normal((N, C_in, H, W), dtype=dtype)
        w = rng.standard_normal((C_out, C_in, K_h, K_w), dtype=dtype)
        b = rng.standard_normal((C_out,), dtype=dtype)

        # grad_out shape from forward
        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        grad_out = rng.standard_normal(y_ref.shape, dtype=dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)
        go_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t, gw_t, gb_t = conv2d_backward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            grad_out=go_t,
            stride=stride,
            padding=padding,
            device_index=self.device_index,
            sync=True,
        )

        gx = self._read_cuda_tensor_to_host(gx_t)
        gw = self._read_cuda_tensor_to_host(gw_t)
        gb = self._read_cuda_tensor_to_host(gb_t) if gb_t is not None else None

        gx_ref, gw_ref, gb_ref = _conv2d_backward_ref(
            x, w, b, grad_out, stride=stride, padding=padding
        )

        # backward uses atomics -> slightly looser tolerance
        np.testing.assert_allclose(gx, gx_ref, rtol=3e-4, atol=3e-4)
        np.testing.assert_allclose(gw, gw_ref, rtol=3e-4, atol=3e-4)
        self.assertIsNotNone(gb)
        self.assertIsNotNone(gb_ref)
        np.testing.assert_allclose(gb, gb_ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_backward_f64_no_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        dtype = np.float64

        N, C_in, H, W = 1, 2, 7, 7
        C_out, K_h, K_w = 2, 3, 3
        stride = (2, 2)
        padding = (1, 1)

        x = rng.standard_normal((N, C_in, H, W)).astype(dtype)
        w = rng.standard_normal((C_out, C_in, K_h, K_w)).astype(dtype)
        b = None

        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        grad_out = rng.standard_normal(y_ref.shape).astype(dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        go_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t, gw_t, gb_t = conv2d_backward_cuda_tensor(
            x_t,
            w_t,
            b=None,
            grad_out=go_t,
            stride=stride,
            padding=padding,
            device_index=self.device_index,
            sync=True,
        )

        gx = self._read_cuda_tensor_to_host(gx_t)
        gw = self._read_cuda_tensor_to_host(gw_t)
        self.assertIsNone(gb_t)

        gx_ref, gw_ref, gb_ref = _conv2d_backward_ref(
            x, w, b, grad_out, stride=stride, padding=padding
        )
        np.testing.assert_allclose(gx, gx_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(gw, gw_ref, rtol=1e-10, atol=1e-10)
        self.assertIsNone(gb_ref)

    def test_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.standard_normal((1, 2, 5, 5), dtype=np.float32)
        w = rng.standard_normal((3, 2, 3, 3)).astype(np.float64)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        with self.assertRaises(TypeError):
            _ = conv2d_forward_cuda_tensor(
                x_t,
                w_t,
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )

    def test_raises_on_in_channels_mismatch(self) -> None:
        rng = np.random.default_rng(6)
        dtype = np.float32
        x = rng.standard_normal((1, 3, 5, 5), dtype=dtype)
        w = rng.standard_normal((4, 2, 3, 3), dtype=dtype)  # mismatch

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        with self.assertRaises(ValueError):
            _ = conv2d_forward_cuda_tensor(
                x_t,
                w_t,
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )

    def test_raises_on_cpu_tensor(self) -> None:
        x_np = np.ones((1, 1, 5, 5), dtype=np.float32)
        w_np = np.ones((1, 1, 3, 3), dtype=np.float32)

        # Construct CPU tensors in a robust way (mirror matmul test)
        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            x_cpu = Tensor.from_numpy(x_np, device=Device("cpu"))  # type: ignore[call-arg]
            w_cpu = Tensor.from_numpy(w_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                x_cpu = Tensor(x_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                x_cpu._data = x_np  # type: ignore[attr-defined]
                w_cpu = Tensor(w_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                w_cpu._data = w_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensors in this repo; update test_raises_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = conv2d_forward_cuda_tensor(
                x_cpu,  # type: ignore[arg-type]
                w_cpu,  # type: ignore[arg-type]
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )


if __name__ == "__main__":
    unittest.main()
