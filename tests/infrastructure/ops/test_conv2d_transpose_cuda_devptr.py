from __future__ import annotations

import unittest
import numpy as np

from ._cuda_test_utils import try_get_cuda_env, resolve_func, assert_allclose_by_dtype


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return v if isinstance(v, tuple) else (int(v), int(v))


def _conv2d_transpose_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    op_h, op_w = _pair(output_padding)

    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError("in_channels mismatch")

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
    b: np.ndarray | None,
    grad_out: np.ndarray,
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    output_padding: int | tuple[int, int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - p_h
                for wi in range(W_in):
                    base_ow = wi * s_w - p_w
                    xval = x[n, ci, hi, wi]
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
                                grad_x[n, ci, hi, wi] += go * w[ci, co, kh, kw]
                                grad_w[ci, co, kh, kw] += xval * go

    grad_b = None
    if b is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(x.dtype, copy=False)

    return grad_x, grad_w, grad_b


class TestConv2dTransposeCudaDevPtrOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = try_get_cuda_env()
        if env is None:
            raise unittest.SkipTest("CUDA native library/wrappers not available.")
        cls.env = env

        from src.keydnn.infrastructure.ops import conv2d_transpose_cuda as ops_conv_t

        cls.ops_conv_t = ops_conv_t

        cls.fwd_devptr = resolve_func(
            ops_conv_t,
            candidates=[
                "conv2d_transpose_forward_cuda_devptr",
                "conv2d_transpose_forward_devptr",
            ],
        )
        cls.bwd_devptr = resolve_func(
            ops_conv_t,
            candidates=[
                "conv2d_transpose_backward_cuda_devptr",
                "conv2d_transpose_backward_devptr",
            ],
        )

        try:
            cuda_malloc = resolve_func(
                ops_conv_t, candidates=["cuda_malloc", "malloc_cuda"]
            )
            cuda_free = resolve_func(ops_conv_t, candidates=["cuda_free", "free_cuda"])
        except AttributeError:
            from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                cuda_malloc,
                cuda_free,
            )

        cls.cuda_malloc = staticmethod(cuda_malloc)
        cls.cuda_free = staticmethod(cuda_free)

        try:
            memcpy_h2d = resolve_func(
                ops_conv_t,
                candidates=["cuda_memcpy_h2d", "memcpy_h2d", "cudaMemcpyHtoD"],
            )
        except AttributeError:
            try:
                from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                    cudaMemcpyHtoD as memcpy_h2d,
                )
            except Exception:
                from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                    cuda_memcpy_h2d as memcpy_h2d,
                )

        try:
            memcpy_d2h = resolve_func(
                ops_conv_t,
                candidates=["cuda_memcpy_d2h", "memcpy_d2h", "cudaMemcpyDtoH"],
            )
        except AttributeError:
            try:
                from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
                    cudaMemcpyDtoH as memcpy_d2h,
                )
            except Exception:
                from src.keydnn.infrastructure.native_cuda.python.ops.memcpy_ctypes import (
                    cuda_memcpy_d2h as memcpy_d2h,
                )

        cls.memcpy_h2d = staticmethod(memcpy_h2d)
        cls.memcpy_d2h = staticmethod(memcpy_d2h)

    def _malloc_dev(self, nbytes: int) -> int:

        return int(self.cuda_malloc(self.env.lib, int(nbytes if nbytes > 0 else 1)))

    def _free_dev(self, dev: int) -> None:
        self.cuda_free(self.env.lib, int(dev))

    def _h2d(self, dev: int, host: np.ndarray) -> None:
        host_c = np.ascontiguousarray(host)

        try:
            self.memcpy_h2d(self.env.lib, int(dev), host_c)
        except TypeError:
            self.memcpy_h2d(self.env.lib, int(dev), host_c, int(host_c.nbytes))

    def _d2h(self, dev: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        out = np.empty(shape, dtype=np.dtype(dtype))

        try:
            self.memcpy_d2h(self.env.lib, out, int(dev))
        except TypeError:
            self.memcpy_d2h(self.env.lib, out, int(dev), int(out.nbytes))
        return out

    def _run_forward_devptr(self, dtype: np.dtype, *, with_bias: bool) -> None:
        dtype = np.dtype(dtype)

        N, C_in, C_out = 2, 3, 4
        H_in, W_in = 4, 5
        K_h, K_w = 3, 2
        stride = (1, 2)
        padding = (1, 0)
        output_padding = (0, 0)

        x = np.random.randn(N, C_in, H_in, W_in).astype(dtype)
        w = np.random.randn(C_in, C_out, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )

        itemsize = int(dtype.itemsize)
        x_dev = self._malloc_dev(int(x.size * itemsize))
        w_dev = self._malloc_dev(int(w.size * itemsize))
        y_dev = self._malloc_dev(int(y_ref.size * itemsize))
        b_dev = self._malloc_dev(int(b.size * itemsize)) if b is not None else None

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
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                K_h=K_h,
                K_w=K_w,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dtype=dtype,
                device_index=0,
                sync=True,
            )

            y = self._d2h(y_dev, y_ref.shape, dtype)
            self.assertEqual(y.shape, y_ref.shape)
            assert_allclose_by_dtype(
                y, y_ref, dtype, op="conv2d_transpose_forward_cuda_devptr"
            )

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
        H_in, W_in = 4, 4
        K_h, K_w = 3, 3
        stride = (1, 1)
        padding = (1, 1)
        output_padding = (0, 0)

        x = np.random.randn(N, C_in, H_in, W_in).astype(dtype)
        w = np.random.randn(C_in, C_out, K_h, K_w).astype(dtype)
        b = np.random.randn(C_out).astype(dtype) if with_bias else None

        y_ref = _conv2d_transpose_forward_ref(
            x, w, b, stride=stride, padding=padding, output_padding=output_padding
        )
        grad_out = np.random.randn(*y_ref.shape).astype(dtype)

        grad_x_ref, grad_w_ref, grad_b_ref = _conv2d_transpose_backward_ref(
            x,
            w,
            b,
            grad_out,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        itemsize = int(dtype.itemsize)

        x_dev = self._malloc_dev(int(x.size * itemsize))
        w_dev = self._malloc_dev(int(w.size * itemsize))
        go_dev = self._malloc_dev(int(grad_out.size * itemsize))

        gx_dev = self._malloc_dev(int(grad_x_ref.size * itemsize))
        gw_dev = self._malloc_dev(int(grad_w_ref.size * itemsize))
        gb_dev = self._malloc_dev(int(C_out * itemsize)) if with_bias else None

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
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                K_h=K_h,
                K_w=K_w,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dtype=dtype,
                device_index=0,
                sync=True,
            )

            grad_x = self._d2h(gx_dev, grad_x_ref.shape, dtype)
            grad_w = self._d2h(gw_dev, grad_w_ref.shape, dtype)

            self.assertEqual(grad_x.shape, grad_x_ref.shape)
            self.assertEqual(grad_w.shape, grad_w_ref.shape)

            if dtype == np.float32:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=4e-4, atol=4e-4)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=4e-4, atol=4e-4)
            else:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=2e-10, atol=2e-10)

            if with_bias:
                self.assertIsNotNone(gb_dev)
                self.assertIsNotNone(grad_b_ref)
                grad_b = self._d2h(int(gb_dev), (C_out,), dtype)
                assert_allclose_by_dtype(
                    grad_b,
                    grad_b_ref,
                    dtype,
                    op="conv2d_transpose_backward_cuda_devptr_grad_b",
                )

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

    def test_rejects_output_padding_ge_stride_devptr(self) -> None:
        dtype = np.float32
        with self.assertRaises(ValueError):
            self.fwd_devptr(
                self.env.lib,
                x_dev=1,
                w_dev=1,
                b_dev=None,
                y_dev=1,
                N=1,
                C_in=1,
                H_in=3,
                W_in=3,
                C_out=1,
                K_h=3,
                K_w=3,
                stride=(2, 2),
                padding=(0, 0),
                output_padding=(2, 0),
                dtype=dtype,
                device_index=0,
                sync=False,
            )


if __name__ == "__main__":
    unittest.main()
