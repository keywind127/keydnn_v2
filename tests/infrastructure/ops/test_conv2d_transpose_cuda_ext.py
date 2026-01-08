# tests/infrastructure/ops/test_conv2d_transpose_cuda_ext.py
"""
Unit tests for CUDA Tensor-boundary conv2d-transpose wrapper (conv2d_transpose_cuda_ext.py).

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
from src.keydnn.infrastructure.ops.conv2d_transpose_cuda_ext import (
    conv2d_transpose_forward_cuda_tensor,
    conv2d_transpose_backward_cuda_tensor,
)


def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.
    Mirrors the helper used in conv2d_cuda_ext tests.
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


def _conv2d_transpose_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    *,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    output_padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    """
    Naive NCHW/IOHW transpose-conv forward reference (scatter accumulate).

    x: (N, C_in, H_in, W_in)
    w: (C_in, C_out, K_h, K_w)
    b: (C_out,) or None

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    if x.ndim != 4 or w.ndim != 4:
        raise ValueError("x and w must be 4D")

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    if b is not None:
        if b.ndim != 1 or b.shape[0] != C_out:
            raise ValueError("bias shape mismatch")

    if s_h <= 0 or s_w <= 0:
        raise ValueError("stride must be positive")
    if op_h < 0 or op_w < 0:
        raise ValueError("output_padding must be non-negative")
    if op_h >= s_h or op_w >= s_w:
        raise ValueError("output_padding must be < stride per dim")

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if H_out < 0 or W_out < 0:
        raise ValueError("invalid output size")

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - p_h
                for wi in range(W_in):
                    base_ow = wi * s_w - p_w
                    xv = x[n, ci, hi, wi]
                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                y[n, co, oh, ow] += xv * w[ci, co, kh, kw]

        if b is not None:
            for co in range(C_out):
                y[n, co, :, :] += b[co]

    return y


def _conv2d_transpose_backward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    grad_out: np.ndarray,
    *,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    output_padding: int | Tuple[int, int] = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Naive transpose-conv backward reference returning (grad_x, grad_w, grad_b).

    grad_x: same shape as x (N, C_in, H_in, W_in)
    grad_w: same shape as w (C_in, C_out, K_h, K_w)
    grad_b: (C_out,) if b is not None else None
    """
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

    H_out = (H_in - 1) * s_h - 2 * p_h + K_h + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_w + op_w
    if grad_out.shape != (N, C_out, H_out, W_out):
        raise ValueError("grad_out shape mismatch")

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

    # Scatter forward:
    # y[n,co,hi*s_h + kh - p_h, wi*s_w + kw - p_w] += x[n,ci,hi,wi] * w[ci,co,kh,kw]
    #
    # Backward:
    # grad_x += grad_out[...] * w
    # grad_w += x * grad_out[...]
    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - p_h
                for wi in range(W_in):
                    base_ow = wi * s_w - p_w
                    xv = x[n, ci, hi, wi]

                    acc_gx = x.dtype.type(0)
                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                go = grad_out[n, co, oh, ow]
                                acc_gx += go * w[ci, co, kh, kw]
                                grad_w[ci, co, kh, kw] += xv * go

                    grad_x[n, ci, hi, wi] += acc_gx

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    return grad_x, grad_w, grad_b


class TestConv2dTransposeCudaExt(unittest.TestCase):
    """
    Tests for Tensor-boundary CUDA conv2d-transpose wrapper.

    Wrappers under test:
        keydnn.infrastructure.ops.conv2d_transpose_cuda_ext.conv2d_transpose_forward_cuda_tensor
        keydnn.infrastructure.ops.conv2d_transpose_cuda_ext.conv2d_transpose_backward_cuda_tensor
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
        nbytes = int(x_c.nbytes)
        dev_ptr = int(cuda_malloc(self.lib, nbytes if nbytes > 0 else 1))
        try:
            if nbytes > 0:
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
        nbytes = int(host.nbytes)
        if nbytes > 0:
            _d2h(self.lib, int(t.data), host)
        return host

    # -------------------------------------------------------------------------
    # Forward tests
    # -------------------------------------------------------------------------
    def test_conv2d_transpose_forward_f32_no_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        dtype = np.float32

        N, C_in, H_in, W_in = 2, 3, 5, 4
        C_out, K_h, K_w = 4, 3, 2
        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=dtype)  # IOHW
        b = None

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        y_t = conv2d_transpose_forward_cuda_tensor(
            x_t,
            w_t,
            b=None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_transpose_forward_f32_with_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        dtype = np.float32

        N, C_in, H_in, W_in = 2, 2, 5, 5
        C_out, K_h, K_w = 3, 3, 3
        stride = (1, 1)
        padding = (1, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=dtype)  # IOHW
        b = rng.standard_normal((C_out,), dtype=dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = conv2d_transpose_forward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_transpose_forward_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        dtype = np.float64

        N, C_in, H_in, W_in = 1, 2, 4, 5
        C_out, K_h, K_w = 2, 3, 2
        stride = (2, 1)
        padding = (1, 0)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in)).astype(dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w)).astype(dtype)
        b = rng.standard_normal((C_out,)).astype(dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = conv2d_transpose_forward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            out_requires_grad=False,
            device_index=self.device_index,
            sync=True,
        )

        y = self._read_cuda_tensor_to_host(y_t)
        ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        np.testing.assert_allclose(y, ref, rtol=1e-10, atol=1e-10)

    # -------------------------------------------------------------------------
    # Backward tests
    # -------------------------------------------------------------------------
    def test_conv2d_transpose_backward_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        dtype = np.float32

        N, C_in, H_in, W_in = 2, 2, 5, 4
        C_out, K_h, K_w = 3, 3, 3
        stride = (1, 1)
        padding = (1, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=dtype)
        b = rng.standard_normal((C_out,), dtype=dtype)

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        grad_out = rng.standard_normal(y_ref.shape, dtype=dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        b_t = self._make_cuda_tensor_from_host(b)
        go_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t, gw_t, gb_t = conv2d_transpose_backward_cuda_tensor(
            x_t,
            w_t,
            b=b_t,
            grad_out=go_t,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            device_index=self.device_index,
            sync=True,
        )

        gx = self._read_cuda_tensor_to_host(gx_t)
        gw = self._read_cuda_tensor_to_host(gw_t)
        gb = self._read_cuda_tensor_to_host(gb_t) if gb_t is not None else None

        gx_ref, gw_ref, gb_ref = _conv2d_transpose_backward_ref(
            x,
            w,
            b,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        # Backward may use atomics -> slightly looser tolerance
        np.testing.assert_allclose(gx, gx_ref, rtol=3e-4, atol=3e-4)
        np.testing.assert_allclose(gw, gw_ref, rtol=3e-4, atol=3e-4)
        self.assertIsNotNone(gb)
        self.assertIsNotNone(gb_ref)
        np.testing.assert_allclose(gb, gb_ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_transpose_backward_f64_no_bias_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        dtype = np.float64

        N, C_in, H_in, W_in = 1, 2, 4, 4
        C_out, K_h, K_w = 2, 3, 3
        stride = (2, 2)
        padding = (1, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in)).astype(dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w)).astype(dtype)
        b = None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        grad_out = rng.standard_normal(y_ref.shape).astype(dtype)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        go_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t, gw_t, gb_t = conv2d_transpose_backward_cuda_tensor(
            x_t,
            w_t,
            b=None,
            grad_out=go_t,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            device_index=self.device_index,
            sync=True,
        )

        gx = self._read_cuda_tensor_to_host(gx_t)
        gw = self._read_cuda_tensor_to_host(gw_t)
        self.assertIsNone(gb_t)

        gx_ref, gw_ref, gb_ref = _conv2d_transpose_backward_ref(
            x,
            w,
            b,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        np.testing.assert_allclose(gx, gx_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(gw, gw_ref, rtol=1e-10, atol=1e-10)
        self.assertIsNone(gb_ref)

    # -------------------------------------------------------------------------
    # Error / validation tests
    # -------------------------------------------------------------------------
    def test_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.standard_normal((1, 2, 4, 4), dtype=np.float32)
        w = rng.standard_normal((2, 3, 3, 3)).astype(
            np.float64
        )  # IOHW but dtype mismatch

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        with self.assertRaises(TypeError):
            _ = conv2d_transpose_forward_cuda_tensor(
                x_t,
                w_t,
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                output_padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )

    def test_raises_on_in_channels_mismatch(self) -> None:
        rng = np.random.default_rng(6)
        dtype = np.float32
        x = rng.standard_normal((1, 3, 4, 4), dtype=dtype)
        w = rng.standard_normal(
            (2, 4, 3, 3), dtype=dtype
        )  # C_in mismatch (w C_in=2, x C_in=3)

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)

        with self.assertRaises(ValueError):
            _ = conv2d_transpose_forward_cuda_tensor(
                x_t,
                w_t,
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                output_padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )

    def test_raises_on_grad_out_shape_mismatch(self) -> None:
        rng = np.random.default_rng(7)
        dtype = np.float32

        N, C_in, H_in, W_in = 1, 2, 4, 4
        C_out, K_h, K_w = 3, 3, 3
        stride = (1, 1)
        padding = (1, 1)
        output_padding = (0, 0)

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=dtype)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=dtype)
        b = None

        # correct output shape
        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )

        grad_out_bad = rng.standard_normal(
            (
                y_ref.shape[0],
                y_ref.shape[1] - 1,
                y_ref.shape[2],
                y_ref.shape[3],
            ),  # wrong C_out
            dtype=dtype,
        )

        x_t = self._make_cuda_tensor_from_host(x)
        w_t = self._make_cuda_tensor_from_host(w)
        go_t = self._make_cuda_tensor_from_host(grad_out_bad)

        with self.assertRaises(ValueError):
            _ = conv2d_transpose_backward_cuda_tensor(
                x_t,
                w_t,
                b=None,
                grad_out=go_t,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                device_index=self.device_index,
                sync=True,
            )

    def test_raises_on_cpu_tensor(self) -> None:
        x_np = np.ones((1, 1, 4, 4), dtype=np.float32)
        w_np = np.ones((1, 1, 3, 3), dtype=np.float32)  # IOHW

        # Construct CPU tensors robustly (mirror conv2d_cuda_ext test)
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
            _ = conv2d_transpose_forward_cuda_tensor(
                x_cpu,  # type: ignore[arg-type]
                w_cpu,  # type: ignore[arg-type]
                b=None,
                stride=(1, 1),
                padding=(0, 0),
                output_padding=(0, 0),
                out_requires_grad=False,
                device_index=self.device_index,
                sync=True,
            )


if __name__ == "__main__":
    unittest.main()
