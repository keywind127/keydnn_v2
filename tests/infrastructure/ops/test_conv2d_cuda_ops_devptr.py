from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _conv2d_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
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
    b: np.ndarray | None,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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


class TestConv2dCudaDevPtrOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import conv2d_cuda as ops_conv

        cls.ops_conv = ops_conv

        cls.fwd_devptr = resolve_func(
            ops_conv,
            candidates=["conv2d_forward_cuda_devptr", "conv2d_forward_devptr"],
        )
        cls.bwd_devptr = resolve_func(
            ops_conv,
            candidates=["conv2d_backward_cuda_devptr", "conv2d_backward_devptr"],
        )

        try:
            cls.cuda_malloc = resolve_func(
                ops_conv, candidates=["cuda_malloc", "malloc_cuda"]
            )
            cls.cuda_free = resolve_func(
                ops_conv, candidates=["cuda_free", "free_cuda"]
            )
        except AttributeError:

            from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                cuda_malloc,
                cuda_free,
            )

            cls.cuda_malloc = cuda_malloc
            cls.cuda_free = cuda_free

        try:
            cls.memcpy_h2d = resolve_func(
                ops_conv,
                candidates=["cuda_memcpy_h2d", "memcpy_h2d"],
            )
        except AttributeError:
            from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                cuda_memcpy_h2d,
            )

            cls.memcpy_h2d = cuda_memcpy_h2d

        try:
            cls.memcpy_d2h = resolve_func(
                ops_conv,
                candidates=["cuda_memcpy_d2h", "memcpy_d2h"],
            )
        except AttributeError:

            try:
                from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                    cuda_memcpy_d2h,
                )

                cls.memcpy_d2h = cuda_memcpy_d2h
            except Exception:
                from src.keydnn.infrastructure.native_cuda.python.ops.memcpy_ctypes import (
                    cuda_memcpy_d2h,
                )

                cls.memcpy_d2h = cuda_memcpy_d2h

        cls.cuda_malloc = staticmethod(cls.cuda_malloc)
        cls.cuda_free = staticmethod(cls.cuda_free)
        cls.memcpy_h2d = staticmethod(cls.memcpy_h2d)
        cls.memcpy_d2h = staticmethod(cls.memcpy_d2h)

        cls.fwd_devptr = staticmethod(cls.fwd_devptr)
        cls.bwd_devptr = staticmethod(cls.bwd_devptr)

    def _malloc_dev(self, nbytes: int) -> int:

        return int(self.cuda_malloc(self.env.lib, int(nbytes if nbytes > 0 else 1)))

    def _free_dev(self, dev: int) -> None:
        self.cuda_free(self.env.lib, int(dev))

    def _h2d(self, dev: int, host: np.ndarray) -> None:
        host = np.ascontiguousarray(host)
        self.memcpy_h2d(self.env.lib, int(dev), host)

    def _d2h(self, dev: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        out = np.empty(shape, dtype=np.dtype(dtype))
        self.memcpy_d2h(self.env.lib, out, int(dev))
        return out

    def _run_forward_devptr(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        N, C_in, C_out = 2, 3, 4
        H, W = 5, 6
        K_h, K_w = 3, 2
        stride = (1, 2)
        padding = (1, 0)

        x = np.random.randn(N, C_in, H, W).astype(dtype)
        w = np.random.randn(C_out, C_in, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)

        itemsize = int(dtype.itemsize)
        x_bytes = int(x.size * itemsize)
        w_bytes = int(w.size * itemsize)
        b_bytes = int((b.size * itemsize) if b is not None else 0)
        y_bytes = int(y_ref.size * itemsize)

        x_dev = self._malloc_dev(x_bytes)
        w_dev = self._malloc_dev(w_bytes)
        y_dev = self._malloc_dev(y_bytes)
        b_dev = self._malloc_dev(b_bytes) if b is not None else None

        try:
            self._h2d(x_dev, x)
            self._h2d(w_dev, w)
            if b is not None:
                self._h2d(int(b_dev), b)

            self.fwd_devptr(
                self.env.lib,
                x_dev=int(x_dev),
                w_dev=int(w_dev),
                b_dev=None if b_dev is None else int(b_dev),
                y_dev=int(y_dev),
                N=N,
                C_in=C_in,
                H=H,
                W=W,
                C_out=C_out,
                K_h=K_h,
                K_w=K_w,
                stride=stride,
                padding=padding,
                dtype=dtype,
                device_index=0,
                sync=True,
            )

            y = self._d2h(y_dev, y_ref.shape, dtype)
            self.assertEqual(y.shape, y_ref.shape)
            assert_allclose_by_dtype(y, y_ref, dtype, op="conv2d_forward_cuda_devptr")

        finally:
            self._free_dev(x_dev)
            self._free_dev(w_dev)
            self._free_dev(y_dev)
            if b_dev is not None:
                self._free_dev(int(b_dev))

    def test_forward_devptr_float32_no_bias(self) -> None:
        self._run_forward_devptr(np.float32, with_bias=False)

    def test_forward_devptr_float32_with_bias(self) -> None:
        self._run_forward_devptr(np.float32, with_bias=True)

    def test_forward_devptr_float64_no_bias(self) -> None:
        self._run_forward_devptr(np.float64, with_bias=False)

    def test_forward_devptr_float64_with_bias(self) -> None:
        self._run_forward_devptr(np.float64, with_bias=True)

    def _run_backward_devptr(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        N, C_in, C_out = 2, 2, 3
        H, W = 6, 5
        K_h, K_w = 3, 3
        stride = (1, 1)
        padding = (1, 1)

        x = np.random.randn(N, C_in, H, W).astype(dtype)
        w = np.random.randn(C_out, C_in, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_forward_ref(x, w, b, stride=stride, padding=padding)
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        grad_x_ref, grad_w_ref, grad_b_ref = _conv2d_backward_ref(
            x, w, b, grad_out, stride=stride, padding=padding
        )

        itemsize = int(dtype.itemsize)

        x_bytes = int(x.size * itemsize)
        w_bytes = int(w.size * itemsize)
        go_bytes = int(grad_out.size * itemsize)

        gx_bytes = int(grad_x_ref.size * itemsize)
        gw_bytes = int(grad_w_ref.size * itemsize)
        gb_bytes = int((C_out * itemsize) if with_bias else 0)

        x_dev = self._malloc_dev(x_bytes)
        w_dev = self._malloc_dev(w_bytes)
        go_dev = self._malloc_dev(go_bytes)

        gx_dev = self._malloc_dev(gx_bytes)
        gw_dev = self._malloc_dev(gw_bytes)
        gb_dev = self._malloc_dev(gb_bytes) if with_bias else None

        try:
            self._h2d(x_dev, x)
            self._h2d(w_dev, w)
            self._h2d(go_dev, grad_out)

            self.bwd_devptr(
                self.env.lib,
                x_dev=int(x_dev),
                w_dev=int(w_dev),
                grad_out_dev=int(go_dev),
                grad_x_dev=int(gx_dev),
                grad_w_dev=int(gw_dev),
                grad_b_dev=None if gb_dev is None else int(gb_dev),
                N=N,
                C_in=C_in,
                H=H,
                W=W,
                C_out=C_out,
                H_out=int(y_ref.shape[2]),
                W_out=int(y_ref.shape[3]),
                K_h=K_h,
                K_w=K_w,
                stride=stride,
                padding=padding,
                dtype=dtype,
                device_index=0,
                sync=True,
            )

            grad_x = self._d2h(gx_dev, grad_x_ref.shape, dtype)
            grad_w = self._d2h(gw_dev, grad_w_ref.shape, dtype)

            self.assertEqual(grad_x.shape, grad_x_ref.shape)
            self.assertEqual(grad_w.shape, grad_w_ref.shape)

            if dtype == np.float32:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=3e-4, atol=3e-4)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=3e-4, atol=3e-4)
            else:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=1e-10, atol=1e-10)

            if with_bias:
                self.assertIsNotNone(gb_dev)
                self.assertIsNotNone(grad_b_ref)
                grad_b = self._d2h(int(gb_dev), (C_out,), dtype)
                assert_allclose_by_dtype(
                    grad_b, grad_b_ref, dtype, op="conv2d_backward_cuda_devptr_grad_b"
                )
            else:

                pass

        finally:
            self._free_dev(x_dev)
            self._free_dev(w_dev)
            self._free_dev(go_dev)
            self._free_dev(gx_dev)
            self._free_dev(gw_dev)
            if gb_dev is not None:
                self._free_dev(int(gb_dev))

    def test_backward_devptr_float32_no_bias(self) -> None:
        self._run_backward_devptr(np.float32, with_bias=False)

    def test_backward_devptr_float32_with_bias(self) -> None:
        self._run_backward_devptr(np.float32, with_bias=True)

    def test_backward_devptr_float64_no_bias(self) -> None:
        self._run_backward_devptr(np.float64, with_bias=False)

    def test_backward_devptr_float64_with_bias(self) -> None:
        self._run_backward_devptr(np.float64, with_bias=True)


if __name__ == "__main__":
    unittest.main()
