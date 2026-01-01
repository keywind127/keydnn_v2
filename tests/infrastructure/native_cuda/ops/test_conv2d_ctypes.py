from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_synchronize,
    cudaMemcpyHtoD,
    cudaMemcpyDtoH,
)

from src.keydnn.infrastructure.native_cuda.python.ops.conv2d_ctypes import (
    conv2d_forward_cuda,
    conv2d_backward_cuda,
)


def _conv2d_forward_ref(
    x_pad: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    s_h: int,
    s_w: int,
) -> np.ndarray:
    """
    Naive NCHW/OIHW forward reference matching your C++/CUDA semantics.
    x_pad: (N, C_in, H_pad, W_pad)
    w:     (C_out, C_in, K_h, K_w)
    b:     (C_out,) or None
    """
    assert x_pad.ndim == 4 and w.ndim == 4
    N, C_in, H_pad, W_pad = x_pad.shape
    C_out, C_in2, K_h, K_w = w.shape
    assert C_in2 == C_in
    H_out = (H_pad - K_h) // s_h + 1
    W_out = (W_pad - K_w) // s_w + 1
    assert H_out >= 0 and W_out >= 0

    y = np.zeros((N, C_out, H_out, W_out), dtype=x_pad.dtype)
    for n in range(N):
        for co in range(C_out):
            bias = b[co] if b is not None else x_pad.dtype.type(0)
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    acc = x_pad.dtype.type(0)
                    for ci in range(C_in):
                        for kh in range(K_h):
                            h = h0 + kh
                            for kw in range(K_w):
                                ww = w0 + kw
                                acc += x_pad[n, ci, h, ww] * w[co, ci, kh, kw]
                    y[n, co, i, j] = acc + bias
    return y


def _conv2d_backward_ref(
    x_pad: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    s_h: int,
    s_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive backward reference matching your C++/CUDA semantics (accumulating).
    Returns (grad_x_pad, grad_w). Does NOT compute grad_b.
    """
    N, C_in, H_pad, W_pad = x_pad.shape
    C_out, C_in2, K_h, K_w = w.shape
    assert C_in2 == C_in
    _, C_out2, H_out, W_out = grad_out.shape
    assert C_out2 == C_out

    grad_x_pad = np.zeros_like(x_pad)
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
                            h = h0 + kh
                            for kw in range(K_w):
                                ww = w0 + kw
                                grad_w[co, ci, kh, kw] += go * x_pad[n, ci, h, ww]
                                grad_x_pad[n, ci, h, ww] += go * w[co, ci, kh, kw]

    return grad_x_pad, grad_w


class _CudaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = load_keydnn_cuda_native()
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            cls.lib = None
            cls._skip_reason = f"CUDA native library not available: {e!r}"

    def setUp(self) -> None:
        if getattr(self, "lib", None) is None:
            self.skipTest(getattr(self, "_skip_reason", "CUDA not available"))


class TestConv2dCtypes(_CudaTestCase):
    def _alloc_and_copy_in(self, arr: np.ndarray) -> int:
        lib = self.lib
        nbytes = int(arr.nbytes)
        dev = int(cuda_malloc(lib, nbytes if nbytes > 0 else 1))
        if nbytes > 0:
            cudaMemcpyHtoD(lib, dev, arr, nbytes)
        return dev

    def _copy_out_and_free(self, dev: int, out: np.ndarray) -> None:
        lib = self.lib
        nbytes = int(out.nbytes)
        if nbytes > 0:
            cudaMemcpyDtoH(lib, out, dev, nbytes)
        cuda_free(lib, dev)

    def _run_forward(self, dtype: np.dtype, *, with_bias: bool) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        # Small deterministic-ish case
        N, C_in, C_out = 2, 3, 4
        H_pad, W_pad = 6, 5
        K_h, K_w = 3, 2
        s_h, s_w = 1, 2
        H_out = (H_pad - K_h) // s_h + 1
        W_out = (W_pad - K_w) // s_w + 1

        x_pad = (np.random.randn(N, C_in, H_pad, W_pad)).astype(dtype, copy=False)
        w = (np.random.randn(C_out, C_in, K_h, K_w)).astype(dtype, copy=False)
        b = (np.random.randn(C_out)).astype(dtype, copy=False) if with_bias else None

        y_ref = _conv2d_forward_ref(x_pad, w, b, s_h=s_h, s_w=s_w)
        self.assertEqual(y_ref.shape, (N, C_out, H_out, W_out))

        x_dev = self._alloc_and_copy_in(x_pad)
        w_dev = self._alloc_and_copy_in(w)
        b_dev = self._alloc_and_copy_in(b) if b is not None else 0

        y = np.empty_like(y_ref)
        y_dev = int(cuda_malloc(lib, int(y.nbytes) if int(y.nbytes) > 0 else 1))

        try:
            conv2d_forward_cuda(
                lib,
                x_pad_dev=x_dev,
                w_dev=w_dev,
                b_dev=(b_dev if with_bias else None),
                y_dev=y_dev,
                N=N,
                C_in=C_in,
                H_pad=H_pad,
                W_pad=W_pad,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            cudaMemcpyDtoH(lib, y, y_dev, int(y.nbytes))

            if dtype == np.float32:
                np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)
            else:
                np.testing.assert_allclose(y, y_ref, rtol=1e-12, atol=1e-12)

        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, w_dev)
            if with_bias and b is not None:
                cuda_free(lib, b_dev)
            cuda_free(lib, y_dev)

    def test_forward_float32_no_bias(self) -> None:
        self._run_forward(np.float32, with_bias=False)

    def test_forward_float32_with_bias(self) -> None:
        self._run_forward(np.float32, with_bias=True)

    def test_forward_float64_no_bias(self) -> None:
        self._run_forward(np.float64, with_bias=False)

    def test_forward_float64_with_bias(self) -> None:
        self._run_forward(np.float64, with_bias=True)

    def _run_backward(self, dtype: np.dtype) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        # Keep sizes small; backward uses atomics and can be slow
        N, C_in, C_out = 2, 2, 3
        H_pad, W_pad = 6, 6
        K_h, K_w = 3, 3
        s_h, s_w = 1, 1
        H_out = (H_pad - K_h) // s_h + 1
        W_out = (W_pad - K_w) // s_w + 1

        x_pad = (np.random.randn(N, C_in, H_pad, W_pad)).astype(dtype, copy=False)
        w = (np.random.randn(C_out, C_in, K_h, K_w)).astype(dtype, copy=False)
        grad_out = (np.random.randn(N, C_out, H_out, W_out)).astype(dtype, copy=False)

        grad_x_ref, grad_w_ref = _conv2d_backward_ref(
            x_pad, w, grad_out, s_h=s_h, s_w=s_w
        )

        # Device buffers
        x_dev = self._alloc_and_copy_in(x_pad)
        w_dev = self._alloc_and_copy_in(w)
        go_dev = self._alloc_and_copy_in(grad_out)

        grad_x = np.zeros_like(x_pad)
        grad_w = np.zeros_like(w)

        grad_x_dev = self._alloc_and_copy_in(grad_x)  # zero-init required
        grad_w_dev = self._alloc_and_copy_in(grad_w)  # zero-init required

        try:
            conv2d_backward_cuda(
                lib,
                x_pad_dev=x_dev,
                w_dev=w_dev,
                grad_out_dev=go_dev,
                grad_x_pad_dev=grad_x_dev,
                grad_w_dev=grad_w_dev,
                N=N,
                C_in=C_in,
                H_pad=H_pad,
                W_pad=W_pad,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            cudaMemcpyDtoH(lib, grad_x, grad_x_dev, int(grad_x.nbytes))
            cudaMemcpyDtoH(lib, grad_w, grad_w_dev, int(grad_w.nbytes))

            # Atomics change summation order -> allow slightly looser tolerance
            if dtype == np.float32:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=2e-4, atol=2e-4)
            else:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=5e-11, atol=5e-11)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=5e-11, atol=5e-11)

        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, w_dev)
            cuda_free(lib, go_dev)
            cuda_free(lib, grad_x_dev)
            cuda_free(lib, grad_w_dev)

    def test_backward_float32(self) -> None:
        self._run_backward(np.float32)

    def test_backward_float64(self) -> None:
        self._run_backward(np.float64)

    def test_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            conv2d_forward_cuda(
                lib,
                x_pad_dev=1,
                w_dev=2,
                b_dev=None,
                y_dev=3,
                N=1,
                C_in=1,
                H_pad=3,
                W_pad=3,
                C_out=1,
                H_out=1,
                W_out=1,
                K_h=3,
                K_w=3,
                s_h=1,
                s_w=1,
                dtype=np.int32,
            )

        with self.assertRaises(TypeError):
            conv2d_backward_cuda(
                lib,
                x_pad_dev=1,
                w_dev=2,
                grad_out_dev=3,
                grad_x_pad_dev=4,
                grad_w_dev=5,
                N=1,
                C_in=1,
                H_pad=3,
                W_pad=3,
                C_out=1,
                H_out=1,
                W_out=1,
                K_h=3,
                K_w=3,
                s_h=1,
                s_w=1,
                dtype=np.int32,
            )

    def test_forward_zero_output_is_ok(self) -> None:
        lib = self.lib
        dtype = np.float32

        # Force H_out = 0 by making H_pad < K_h
        N, C_in, C_out = 1, 1, 1
        H_pad, W_pad = 2, 2
        K_h, K_w = 3, 3
        s_h, s_w = 1, 1
        H_out, W_out = 0, 0

        # Allocate minimal buffers (we won't really read/write meaningful data)
        x_dev = int(cuda_malloc(lib, 1))
        w_dev = int(cuda_malloc(lib, 1))
        y_dev = int(cuda_malloc(lib, 1))
        try:
            conv2d_forward_cuda(
                lib,
                x_pad_dev=x_dev,
                w_dev=w_dev,
                b_dev=None,
                y_dev=y_dev,
                N=N,
                C_in=C_in,
                H_pad=H_pad,
                W_pad=W_pad,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, w_dev)
            cuda_free(lib, y_dev)


if __name__ == "__main__":
    unittest.main()
