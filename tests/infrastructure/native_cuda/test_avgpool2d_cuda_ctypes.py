import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_h2d,
    cuda_memcpy_d2h,
    cuda_memset,
    avgpool2d_forward_cuda,
    avgpool2d_backward_cuda,
)

# NOTE:
# This test module intentionally mirrors the coding style used for the
# maxpool2d CUDA ctypes tests:
# - setUpClass loads DLL once
# - skips cleanly if CUDA is unavailable
# - allocates device buffers explicitly
# - compares against NumPy reference


def _avgpool2d_forward_ref(
    x_pad: np.ndarray,
    *,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> np.ndarray:
    """
    NumPy reference implementation of AvgPool2D forward.

    x_pad: (N,C,H_pad,W_pad), zero-padded
    y:     (N,C,H_out,W_out)
    """
    y = np.empty((N, C, H_out, W_out), dtype=x_pad.dtype)
    denom = float(k_h * k_w)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    window = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    y[n, c, i, j] = window.sum(dtype=np.float64) / denom
    return y


def _avgpool2d_backward_ref(
    grad_out: np.ndarray,
    *,
    N: int,
    C: int,
    H_out: int,
    W_out: int,
    H_pad: int,
    W_pad: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> np.ndarray:
    """
    NumPy reference implementation of AvgPool2D backward.

    grad_out:   (N,C,H_out,W_out)
    grad_x_pad: (N,C,H_pad,W_pad), zero-initialized
    """
    grad_x_pad = np.zeros((N, C, H_pad, W_pad), dtype=grad_out.dtype)
    denom = float(k_h * k_w)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w
                    go = grad_out[n, c, i, j] / denom
                    grad_x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w] += go
    return grad_x_pad


class TestAvgPool2DCudaCtypes(unittest.TestCase):
    """
    Unit tests for CUDA AvgPool2D ctypes wrapper.

    These tests validate:
    - Forward output matches NumPy reference for float32/float64
    - Backward gradient matches NumPy reference for float32/float64
    - Wrapper enforces dtype constraints
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Try loading the DLL and selecting device 0.
        # Skip the entire suite if unavailable.
        try:
            cls.lib = load_keydnn_cuda_native()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        try:
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA device unavailable / cannot set device: {e}")

    def _run_forward_case(self, dtype: np.dtype) -> None:
        rng = np.random.default_rng(0)

        # Small, deterministic shapes for correctness tests
        N, C = 2, 3
        H_pad, W_pad = 7, 8
        k_h, k_w = 3, 2
        s_h, s_w = 2, 2

        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        # Zero-padded input (avgpool assumes zeros are meaningful)
        x_pad = rng.normal(size=(N, C, H_pad, W_pad)).astype(dtype, copy=False)

        # Reference
        y_ref = _avgpool2d_forward_ref(
            x_pad,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

        # Device allocations
        x_dev = cuda_malloc(self.lib, x_pad.nbytes)
        y = np.empty((N, C, H_out, W_out), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y.nbytes)

        try:
            cuda_memcpy_h2d(self.lib, x_dev, x_pad)

            avgpool2d_forward_cuda(
                self.lib,
                x_pad_dev=x_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
                dtype=dtype,
                sync=True,
            )

            cuda_memcpy_d2h(self.lib, y, y_dev)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        # Compare
        atol = 1e-5 if dtype == np.float32 else 1e-12
        rtol = 1e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(y, y_ref, rtol=rtol, atol=atol)

    def _run_backward_case(self, dtype: np.dtype) -> None:
        rng = np.random.default_rng(1)

        N, C = 2, 2
        H_pad, W_pad = 6, 7
        k_h, k_w = 2, 3
        s_h, s_w = 2, 2

        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        grad_out = rng.normal(size=(N, C, H_out, W_out)).astype(dtype, copy=False)

        grad_x_ref = _avgpool2d_backward_ref(
            grad_out,
            N=N,
            C=C,
            H_out=H_out,
            W_out=W_out,
            H_pad=H_pad,
            W_pad=W_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

        grad_out_dev = cuda_malloc(self.lib, grad_out.nbytes)
        grad_x = np.empty((N, C, H_pad, W_pad), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x.nbytes)

        try:
            cuda_memcpy_h2d(self.lib, grad_out_dev, grad_out)

            # Important: grad_x_pad must be zero-initialized on device
            cuda_memset(self.lib, grad_x_dev, 0, grad_x.nbytes)

            avgpool2d_backward_cuda(
                self.lib,
                grad_out_dev=grad_out_dev,
                grad_x_pad_dev=grad_x_dev,
                N=N,
                C=C,
                H_out=H_out,
                W_out=W_out,
                H_pad=H_pad,
                W_pad=W_pad,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
                dtype=dtype,
                sync=True,
            )

            cuda_memcpy_d2h(self.lib, grad_x, grad_x_dev)

        finally:
            cuda_free(self.lib, grad_out_dev)
            cuda_free(self.lib, grad_x_dev)

        atol = 1e-5 if dtype == np.float32 else 1e-12
        rtol = 1e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(grad_x, grad_x_ref, rtol=rtol, atol=atol)

    # ----------------------------
    # Tests
    # ----------------------------

    def test_avgpool2d_forward_f32_matches_numpy(self) -> None:
        self._run_forward_case(np.float32)

    def test_avgpool2d_forward_f64_matches_numpy(self) -> None:
        self._run_forward_case(np.float64)

    def test_avgpool2d_backward_f32_matches_numpy(self) -> None:
        self._run_backward_case(np.float32)

    def test_avgpool2d_backward_f64_matches_numpy(self) -> None:
        self._run_backward_case(np.float64)

    def test_forward_rejects_unsupported_dtype(self) -> None:
        # Allocate dummy device pointers for call path validation
        N, C = 1, 1
        H_pad, W_pad = 4, 4
        k_h, k_w = 2, 2
        s_h, s_w = 2, 2
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        x = np.zeros((N, C, H_pad, W_pad), dtype=np.float32)
        y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        x_dev = cuda_malloc(self.lib, x.nbytes)
        y_dev = cuda_malloc(self.lib, y.nbytes)

        try:
            cuda_memcpy_h2d(self.lib, x_dev, x)

            with self.assertRaises(TypeError):
                avgpool2d_forward_cuda(
                    self.lib,
                    x_pad_dev=x_dev,
                    y_dev=y_dev,
                    N=N,
                    C=C,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    H_out=H_out,
                    W_out=W_out,
                    k_h=k_h,
                    k_w=k_w,
                    s_h=s_h,
                    s_w=s_w,
                    dtype=np.int32,  # unsupported
                    sync=True,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

    def test_backward_rejects_unsupported_dtype(self) -> None:
        N, C = 1, 1
        H_pad, W_pad = 4, 4
        k_h, k_w = 2, 2
        s_h, s_w = 2, 2
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        grad_out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        grad_x = np.zeros((N, C, H_pad, W_pad), dtype=np.float32)

        go_dev = cuda_malloc(self.lib, grad_out.nbytes)
        gx_dev = cuda_malloc(self.lib, grad_x.nbytes)

        try:
            cuda_memcpy_h2d(self.lib, go_dev, grad_out)
            cuda_memset(self.lib, gx_dev, 0, grad_x.nbytes)

            with self.assertRaises(TypeError):
                avgpool2d_backward_cuda(
                    self.lib,
                    grad_out_dev=go_dev,
                    grad_x_pad_dev=gx_dev,
                    N=N,
                    C=C,
                    H_out=H_out,
                    W_out=W_out,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    k_h=k_h,
                    k_w=k_w,
                    s_h=s_h,
                    s_w=s_w,
                    dtype=np.int32,  # unsupported
                    sync=True,
                )
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, gx_dev)


if __name__ == "__main__":
    unittest.main()
