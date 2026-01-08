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

from src.keydnn.infrastructure.native_cuda.python.ops.conv2d_transpose_ctypes import (
    conv2d_transpose_forward_cuda,
    conv2d_transpose_backward_cuda,
)


def _conv2d_transpose_forward_ref(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    H_out: int,
    W_out: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    """
    Naive NCHW/IOHW ConvTranspose2D forward reference matching the provided CUDA gather semantics.

    x: (N, C_in, H_in, W_in)
    w: (C_in, C_out, K_h, K_w)
    b: (C_out,) or None
    y: (N, C_out, H_out, W_out)

    Semantics (matches gather implementation):
      y[n,co,oh,ow] = sum_{ci,kh,kw} x[n,ci,hi,wi] * w[ci,co,kh,kw] + b[co]
    where hi = (oh + pad_h - kh)/s_h and wi = (ow + pad_w - kw)/s_w must be integers in-range.
    """
    assert x.ndim == 4 and w.ndim == 4
    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    assert C_in2 == C_in

    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for co in range(C_out):
            bias = b[co] if b is not None else x.dtype.type(0)
            for oh in range(H_out):
                for ow in range(W_out):
                    acc = x.dtype.type(0)
                    for ci in range(C_in):
                        for kh in range(K_h):
                            th = oh + pad_h - kh
                            if th < 0 or (th % s_h) != 0:
                                continue
                            hi = th // s_h
                            if hi < 0 or hi >= H_in:
                                continue

                            for kw in range(K_w):
                                tw = ow + pad_w - kw
                                if tw < 0 or (tw % s_w) != 0:
                                    continue
                                wi = tw // s_w
                                if wi < 0 or wi >= W_in:
                                    continue

                                acc += x[n, ci, hi, wi] * w[ci, co, kh, kw]
                    y[n, co, oh, ow] = acc + bias
    return y


def _conv2d_transpose_backward_ref(
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive ConvTranspose2D backward reference matching the provided CUDA gather semantics.

    x:        (N, C_in, H_in, W_in)
    w:        (C_in, C_out, K_h, K_w)
    grad_out: (N, C_out, H_out, W_out)

    Returns (grad_x, grad_w). Does NOT compute grad_b.

    Matches kernel math:
      grad_x[n,ci,hi,wi] = sum_{co,kh,kw} grad_out[n,co,oh,ow] * w[ci,co,kh,kw]
        where oh = hi*s_h - pad_h + kh, ow = wi*s_w - pad_w + kw (must be in bounds)

      grad_w[ci,co,kh,kw] = sum_{n,hi,wi} x[n,ci,hi,wi] * grad_out[n,co,oh,ow]
        with same oh/ow mapping.
    """
    N, C_in, H_in, W_in = x.shape
    C_in2, C_out, K_h, K_w = w.shape
    assert C_in2 == C_in
    N2, C_out2, H_out, W_out = grad_out.shape
    assert N2 == N and C_out2 == C_out

    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)

    # grad_x
    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                base_oh = hi * s_h - pad_h
                for wi in range(W_in):
                    base_ow = wi * s_w - pad_w
                    acc = x.dtype.type(0)
                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                acc += grad_out[n, co, oh, ow] * w[ci, co, kh, kw]
                    grad_x[n, ci, hi, wi] = acc

    # grad_w
    for ci in range(C_in):
        for co in range(C_out):
            for kh in range(K_h):
                for kw in range(K_w):
                    acc = x.dtype.type(0)
                    for n in range(N):
                        for hi in range(H_in):
                            oh = hi * s_h - pad_h + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for wi in range(W_in):
                                ow = wi * s_w - pad_w + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                acc += x[n, ci, hi, wi] * grad_out[n, co, oh, ow]
                    grad_w[ci, co, kh, kw] = acc

    return grad_x, grad_w


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


class TestConv2dTransposeCtypesCuda(_CudaTestCase):
    def _alloc_and_copy_in(self, arr: np.ndarray) -> int:
        lib = self.lib
        nbytes = int(arr.nbytes)
        dev = int(cuda_malloc(lib, nbytes if nbytes > 0 else 1))
        if nbytes > 0:
            cudaMemcpyHtoD(lib, dev, arr, nbytes)
        return dev

    def _copy_out(self, dev: int, out: np.ndarray) -> None:
        lib = self.lib
        nbytes = int(out.nbytes)
        if nbytes > 0:
            cudaMemcpyDtoH(lib, out, dev, nbytes)

    def _run_forward(self, dtype: np.dtype, *, with_bias: bool) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        # Small deterministic-ish case
        N, C_in, C_out = 2, 3, 4
        H_in, W_in = 4, 5
        K_h, K_w = 3, 2
        s_h, s_w = 2, 1
        pad_h, pad_w = 1, 0

        # Output size chosen by caller (matches your kernel contract)
        H_out = (H_in - 1) * s_h - 2 * pad_h + K_h
        W_out = (W_in - 1) * s_w - 2 * pad_w + K_w

        x = (np.random.randn(N, C_in, H_in, W_in)).astype(dtype, copy=False)
        w = (np.random.randn(C_in, C_out, K_h, K_w)).astype(dtype, copy=False)
        b = (np.random.randn(C_out)).astype(dtype, copy=False) if with_bias else None

        y_ref = _conv2d_transpose_forward_ref(
            x,
            w,
            b,
            H_out=H_out,
            W_out=W_out,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )
        self.assertEqual(y_ref.shape, (N, C_out, H_out, W_out))

        x_dev = self._alloc_and_copy_in(x)
        w_dev = self._alloc_and_copy_in(w)
        b_dev = self._alloc_and_copy_in(b) if b is not None else 0

        y = np.empty_like(y_ref)
        y_dev = int(cuda_malloc(lib, int(y.nbytes) if int(y.nbytes) > 0 else 1))

        try:
            conv2d_transpose_forward_cuda(
                lib,
                x_dev=x_dev,
                w_dev=w_dev,
                b_dev=(b_dev if with_bias else None),
                y_dev=y_dev,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            self._copy_out(y_dev, y)

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

        # Keep sizes small; backward includes a grad_w kernel that sums over N*H_in*W_in
        N, C_in, C_out = 2, 2, 3
        H_in, W_in = 4, 4
        K_h, K_w = 3, 3
        s_h, s_w = 2, 2
        pad_h, pad_w = 1, 1

        H_out = (H_in - 1) * s_h - 2 * pad_h + K_h
        W_out = (W_in - 1) * s_w - 2 * pad_w + K_w

        x = (np.random.randn(N, C_in, H_in, W_in)).astype(dtype, copy=False)
        w = (np.random.randn(C_in, C_out, K_h, K_w)).astype(dtype, copy=False)
        grad_out = (np.random.randn(N, C_out, H_out, W_out)).astype(dtype, copy=False)

        grad_x_ref, grad_w_ref = _conv2d_transpose_backward_ref(
            x, w, grad_out, s_h=s_h, s_w=s_w, pad_h=pad_h, pad_w=pad_w
        )

        x_dev = self._alloc_and_copy_in(x)
        w_dev = self._alloc_and_copy_in(w)
        go_dev = self._alloc_and_copy_in(grad_out)

        grad_x = np.empty_like(x)
        grad_w = np.empty_like(w)

        grad_x_dev = int(
            cuda_malloc(lib, int(grad_x.nbytes) if int(grad_x.nbytes) > 0 else 1)
        )
        grad_w_dev = int(
            cuda_malloc(lib, int(grad_w.nbytes) if int(grad_w.nbytes) > 0 else 1)
        )

        try:
            conv2d_transpose_backward_cuda(
                lib,
                x_dev=x_dev,
                w_dev=w_dev,
                grad_out_dev=go_dev,
                grad_x_dev=grad_x_dev,
                grad_w_dev=grad_w_dev,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            self._copy_out(grad_x_dev, grad_x)
            self._copy_out(grad_w_dev, grad_w)

            # Gather-style kernels are deterministic; tolerances can be tight
            if dtype == np.float32:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=1e-5, atol=1e-5)
            else:
                np.testing.assert_allclose(grad_x, grad_x_ref, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(grad_w, grad_w_ref, rtol=1e-12, atol=1e-12)

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
            conv2d_transpose_forward_cuda(
                lib,
                x_dev=1,
                w_dev=2,
                b_dev=None,
                y_dev=3,
                N=1,
                C_in=1,
                H_in=3,
                W_in=3,
                C_out=1,
                H_out=1,
                W_out=1,
                K_h=3,
                K_w=3,
                s_h=1,
                s_w=1,
                pad_h=0,
                pad_w=0,
                dtype=np.int32,
            )

        with self.assertRaises(TypeError):
            conv2d_transpose_backward_cuda(
                lib,
                x_dev=1,
                w_dev=2,
                grad_out_dev=3,
                grad_x_dev=4,
                grad_w_dev=5,
                N=1,
                C_in=1,
                H_in=3,
                W_in=3,
                C_out=1,
                H_out=1,
                W_out=1,
                K_h=3,
                K_w=3,
                s_h=1,
                s_w=1,
                pad_h=0,
                pad_w=0,
                dtype=np.int32,
            )

    def test_forward_zero_output_is_ok(self) -> None:
        lib = self.lib
        dtype = np.float32

        # Force H_out = 0 by choosing H_out/W_out directly (kernel contract)
        N, C_in, C_out = 1, 1, 1
        H_in, W_in = 2, 2
        K_h, K_w = 3, 3
        s_h, s_w = 1, 1
        pad_h, pad_w = 0, 0
        H_out, W_out = 0, 0

        # Allocate minimal buffers (kernel should early-exit by guard)
        x_dev = int(cuda_malloc(lib, 1))
        w_dev = int(cuda_malloc(lib, 1))
        y_dev = int(cuda_malloc(lib, 1))
        try:
            conv2d_transpose_forward_cuda(
                lib,
                x_dev=x_dev,
                w_dev=w_dev,
                b_dev=None,
                y_dev=y_dev,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
                dtype=dtype,
            )
            cuda_synchronize(lib)
        finally:
            cuda_free(lib, x_dev)
            cuda_free(lib, w_dev)
            cuda_free(lib, y_dev)


if __name__ == "__main__":
    unittest.main()
