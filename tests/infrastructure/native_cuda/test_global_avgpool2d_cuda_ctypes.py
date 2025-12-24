"""
Unit tests for KeyDNN v2 CUDA GlobalAvgPool2D ctypes bindings.

These tests validate correctness of the CUDA GlobalAvgPool2D forward/backward
kernels (float32/float64) by comparing against NumPy reference implementations.

Assumptions
-----------
- The CUDA native DLL has been built and is loadable.
- The Python ctypes wrapper for GlobalAvgPool2D is available at:
    keydnn/infrastructure/native_cuda/python/global_avgpool2d_ctypes.py
- Tests are written in `unittest` style to match the existing native_cuda tests.

Notes
-----
- Tests are skipped cleanly if the DLL cannot be loaded.
- We use relatively strict tolerances for float64 and practical tolerances for float32.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_from_host,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_d2h,
    cuda_synchronize,
    global_avgpool2d_forward_cuda,
    global_avgpool2d_backward_cuda,
)

# ---------------------------------------------------------------------
# NumPy reference implementations (match your CPU semantics)
# ---------------------------------------------------------------------


def global_avgpool2d_forward_cpu(x: np.ndarray) -> np.ndarray:
    """
    Global average pooling forward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (N, C, H, W).

    Returns
    -------
    np.ndarray
        Output tensor of shape (N, C, 1, 1).
    """
    return x.mean(axis=(2, 3), keepdims=True).astype(x.dtype, copy=False)


def global_avgpool2d_backward_cpu(
    grad_out: np.ndarray, *, x_shape: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Global average pooling backward pass (CPU, NumPy), NCHW.

    Parameters
    ----------
    grad_out : np.ndarray
        Gradient with respect to output, shape (N, C, 1, 1).
    x_shape : tuple[int, int, int, int]
        Original input shape.

    Returns
    -------
    np.ndarray
        Gradient with respect to input, shape (N, C, H, W).
    """
    N, C, H, W = x_shape
    scale = 1.0 / float(H * W)
    return (np.ones((N, C, H, W), dtype=grad_out.dtype) * grad_out * scale).astype(
        grad_out.dtype, copy=False
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestGlobalAvgPool2DCudaCtypes(unittest.TestCase):
    """
    Correctness tests for CUDA GlobalAvgPool2D via ctypes.

    The tests allocate device buffers using the exported CUDA malloc/free,
    run the forward/backward kernels, and copy results back to host for
    comparison against NumPy references.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Try to load the CUDA DLL; if unavailable, skip.
        try:
            cls.lib = load_keydnn_cuda_native()
        except (OSError, FileNotFoundError) as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        # Try to set device 0; if fails, skip.
        try:
            cuda_set_device(cls.lib, 0)
        except RuntimeError as e:
            raise unittest.SkipTest(f"CUDA device unavailable: {e}")

    def _rng(self) -> np.random.Generator:
        # Deterministic tests
        return np.random.default_rng(12345)

    def _tols(self, dtype: np.dtype) -> tuple[float, float]:
        if dtype == np.float32:
            return (1e-5, 1e-6)
        return (1e-12, 1e-12)

    def _run_forward_case(
        self, dtype: np.dtype, *, N: int, C: int, H: int, W: int
    ) -> None:
        rng = self._rng()
        x = rng.standard_normal((N, C, H, W)).astype(dtype, copy=False)

        # Reference
        y_ref = global_avgpool2d_forward_cpu(x)

        # Device allocations:
        # - x_dev holds full input
        # - y_dev holds N*C contiguous values (flattened y)
        x_dev = cuda_from_host(self.lib, x)
        y_flat = np.empty((N * C,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_flat.nbytes)

        try:
            global_avgpool2d_forward_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                dtype=dtype,
            )

            # copy back
            cuda_memcpy_d2h(self.lib, y_flat, y_dev)
            cuda_synchronize(self.lib)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        y = y_flat.reshape((N, C, 1, 1))

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(y, y_ref, rtol=rtol, atol=atol)

    def _run_backward_case(
        self, dtype: np.dtype, *, N: int, C: int, H: int, W: int
    ) -> None:
        rng = self._rng()
        grad_out = rng.standard_normal((N, C, 1, 1)).astype(dtype, copy=False)

        # Reference
        grad_x_ref = global_avgpool2d_backward_cpu(grad_out, x_shape=(N, C, H, W))

        # Device allocations:
        # - grad_out_dev holds N*C contiguous values (flattened grad_out)
        # - grad_x_dev holds full (N*C*H*W)
        go_flat = grad_out.reshape(N * C)
        go_dev = cuda_from_host(self.lib, go_flat)

        grad_x = np.empty((N, C, H, W), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x.nbytes)

        try:
            global_avgpool2d_backward_cuda(
                self.lib,
                grad_out_dev=go_dev,
                grad_x_dev=grad_x_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                dtype=dtype,
            )

            cuda_memcpy_d2h(self.lib, grad_x, grad_x_dev)
            cuda_synchronize(self.lib)

        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(grad_x, grad_x_ref, rtol=rtol, atol=atol)

    # -------------------------
    # Forward tests
    # -------------------------

    def test_forward_f32_matches_numpy_small(self) -> None:
        self._run_forward_case(np.float32, N=2, C=3, H=4, W=5)

    def test_forward_f64_matches_numpy_small(self) -> None:
        self._run_forward_case(np.float64, N=2, C=3, H=4, W=5)

    def test_forward_f32_matches_numpy_edge_hw1(self) -> None:
        # H=W=1 => output equals input per (n,c)
        self._run_forward_case(np.float32, N=3, C=4, H=1, W=1)

    def test_forward_f64_matches_numpy_edge_hw1(self) -> None:
        self._run_forward_case(np.float64, N=3, C=4, H=1, W=1)

    def test_forward_f32_larger_spatial(self) -> None:
        self._run_forward_case(np.float32, N=1, C=8, H=32, W=33)

    def test_forward_f64_larger_spatial(self) -> None:
        self._run_forward_case(np.float64, N=1, C=8, H=32, W=33)

    # -------------------------
    # Backward tests
    # -------------------------

    def test_backward_f32_matches_numpy_small(self) -> None:
        self._run_backward_case(np.float32, N=2, C=3, H=4, W=5)

    def test_backward_f64_matches_numpy_small(self) -> None:
        self._run_backward_case(np.float64, N=2, C=3, H=4, W=5)

    def test_backward_f32_edge_hw1(self) -> None:
        # H=W=1 => grad_x == grad_out broadcast
        self._run_backward_case(np.float32, N=2, C=7, H=1, W=1)

    def test_backward_f64_edge_hw1(self) -> None:
        self._run_backward_case(np.float64, N=2, C=7, H=1, W=1)

    def test_backward_f32_larger_spatial(self) -> None:
        self._run_backward_case(np.float32, N=1, C=8, H=31, W=29)

    def test_backward_f64_larger_spatial(self) -> None:
        self._run_backward_case(np.float64, N=1, C=8, H=31, W=29)

    # -------------------------
    # Basic shape + sanity tests
    # -------------------------

    def test_forward_output_shape_is_n_c_1_1(self) -> None:
        dtype = np.float32
        N, C, H, W = 4, 5, 6, 7
        rng = self._rng()
        x = rng.standard_normal((N, C, H, W)).astype(dtype, copy=False)

        x_dev = cuda_from_host(self.lib, x)
        y_flat = np.empty((N * C,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_flat.nbytes)

        try:
            global_avgpool2d_forward_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, y_flat, y_dev)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        y = y_flat.reshape((N, C, 1, 1))
        self.assertEqual(y.shape, (N, C, 1, 1))

    def test_backward_distributes_grad_evenly(self) -> None:
        dtype = np.float64
        N, C, H, W = 2, 3, 4, 5

        grad_out = np.zeros((N, C, 1, 1), dtype=dtype)
        grad_out[0, 0, 0, 0] = 10.0

        grad_x_ref = global_avgpool2d_backward_cpu(grad_out, x_shape=(N, C, H, W))

        go_flat = grad_out.reshape(N * C)
        go_dev = cuda_from_host(self.lib, go_flat)
        grad_x = np.empty((N, C, H, W), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x.nbytes)

        try:
            global_avgpool2d_backward_cuda(
                self.lib,
                grad_out_dev=go_dev,
                grad_x_dev=grad_x_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, grad_x, grad_x_dev)
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(grad_x, grad_x_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
