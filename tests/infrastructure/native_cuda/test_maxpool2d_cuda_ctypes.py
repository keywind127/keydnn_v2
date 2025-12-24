"""
Unit tests for KeyDNN v2 CUDA MaxPool2D ctypes wrapper.

These tests use Python's built-in `unittest` library and validate:
- DLL loading (or skip cleanly)
- CUDA device availability (or skip cleanly)
- CUDA utils: malloc/free, memcpy roundtrip, memset
- MaxPool2D forward correctness vs NumPy reference (float32/float64)
- MaxPool2D backward correctness vs NumPy reference (float32/float64)

To override the DLL path at runtime, set:
    KEYDNN_CUDA_DLL=/abs/path/to/KeyDNNV2CudaNative.dll

If the DLL or CUDA device is not available, tests are skipped.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np


# Update import path to match your repo:
# keydnn/infrastructure/native_cuda/python/maxpool2d_cuda_ctypes.py
from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_h2d,
    cuda_memcpy_d2h,
    cuda_memset,
    cuda_synchronize,
    cuda_from_host,
    maxpool2d_forward_cuda,
    maxpool2d_backward_cuda,
)


def _ref_maxpool2d_forward(
    x_pad: np.ndarray,
    *,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    H_out: int,
    W_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy reference MaxPool2D forward (KeyDNN v2 semantics).

    - x_pad: already padded, NCHW contiguous
    - argmax_idx stores flattened index into padded plane: h * W_pad + w
    - tie-break: first occurrence row-major => strict ">"
    """
    N, C, H_pad, W_pad = x_pad.shape
    y = np.empty((N, C, H_out, W_out), dtype=x_pad.dtype)
    argmax_idx = np.empty((N, C, H_out, W_out), dtype=np.int64)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w

                    best = -np.inf
                    best_h = h0
                    best_w = w0

                    for ph in range(k_h):
                        h = h0 + ph
                        for pw in range(k_w):
                            w = w0 + pw
                            v = x_pad[n, c, h, w]
                            if v > best:
                                best = v
                                best_h = h
                                best_w = w

                    y[n, c, i, j] = best
                    argmax_idx[n, c, i, j] = best_h * W_pad + best_w

    return y, argmax_idx


def _ref_maxpool2d_backward(
    grad_out: np.ndarray,
    argmax_idx: np.ndarray,
    *,
    H_pad: int,
    W_pad: int,
) -> np.ndarray:
    """
    NumPy reference MaxPool2D backward (KeyDNN v2 semantics).

    Accumulates grad_out into grad_x_pad at argmax locations.
    """
    N, C, H_out, W_out = grad_out.shape
    grad_x_pad = np.zeros((N, C, H_pad, W_pad), dtype=grad_out.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    idx = int(argmax_idx[n, c, i, j])
                    h = idx // W_pad
                    w = idx % W_pad
                    grad_x_pad[n, c, h, w] += grad_out[n, c, i, j]

    return grad_x_pad


def _dll_path_from_env() -> Path | None:
    p = os.environ.get("KEYDNN_CUDA_DLL", "").strip()
    if not p:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


class TestMaxPool2DCudaCtypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load DLL (skip all tests if missing / unloadable)
        dll_path = _dll_path_from_env()
        try:
            cls.lib = (
                load_keydnn_cuda_native(dll_path)
                if dll_path
                else load_keydnn_cuda_native()
            )
        except FileNotFoundError as e:
            raise unittest.SkipTest(
                f"CUDA DLL not found. Set KEYDNN_CUDA_DLL to override. Details: {e}"
            )
        except OSError as e:
            raise unittest.SkipTest(
                f"Failed to load CUDA DLL (deps/arch/CUDA runtime). Details: {e}"
            )

        # Ensure CUDA device is available (skip all tests if not)
        try:
            cuda_set_device(cls.lib, 0)
        except RuntimeError as e:
            raise unittest.SkipTest(
                f"CUDA device not available / set_device failed. Details: {e}"
            )

    def test_cuda_malloc_free_smoke(self) -> None:
        dev = cuda_malloc(self.lib, 256)
        try:
            self.assertIsInstance(dev, int)
            self.assertNotEqual(dev, 0)
        finally:
            cuda_free(self.lib, dev)

    def test_cuda_memcpy_roundtrip_f32(self) -> None:
        x = (np.arange(128, dtype=np.float32) * 0.25).reshape(32, 4)
        dev = cuda_from_host(self.lib, x)
        y = np.empty_like(x)
        try:
            cuda_memcpy_d2h(self.lib, y, dev)
            np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)
        finally:
            cuda_free(self.lib, dev)

    def test_cuda_memset_zero(self) -> None:
        x = np.random.randn(64).astype(np.float32)
        dev = cuda_from_host(self.lib, x)
        y = np.empty_like(x)
        try:
            cuda_memset(self.lib, dev, value=0, nbytes=x.nbytes)
            cuda_synchronize(self.lib)
            cuda_memcpy_d2h(self.lib, y, dev)
            self.assertTrue(np.all(y == 0.0))
        finally:
            cuda_free(self.lib, dev)

    def _run_forward_case(self, dtype: np.dtype) -> None:
        N, C = 2, 3
        H_pad, W_pad = 7, 8
        k_h, k_w = 3, 2
        s_h, s_w = 2, 2
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        rng = np.random.default_rng(0)
        x_pad = rng.standard_normal((N, C, H_pad, W_pad)).astype(dtype, copy=False)
        if not x_pad.flags["C_CONTIGUOUS"]:
            x_pad = np.ascontiguousarray(x_pad)

        y_ref, idx_ref = _ref_maxpool2d_forward(
            x_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
            H_out=H_out,
            W_out=W_out,
        )

        x_dev = cuda_from_host(self.lib, x_pad)

        y_host = np.empty((N, C, H_out, W_out), dtype=dtype)
        idx_host = np.empty((N, C, H_out, W_out), dtype=np.int64)

        y_dev = cuda_malloc(self.lib, y_host.nbytes)
        idx_dev = cuda_malloc(self.lib, idx_host.nbytes)

        try:
            maxpool2d_forward_cuda(
                self.lib,
                x_pad_dev=x_dev,
                y_dev=y_dev,
                argmax_idx_dev=idx_dev,
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

            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_memcpy_d2h(self.lib, idx_host, idx_dev)

            np.testing.assert_allclose(y_host, y_ref, rtol=0.0, atol=0.0)
            np.testing.assert_array_equal(idx_host, idx_ref)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

    def _run_backward_case(self, dtype: np.dtype) -> None:
        N, C = 2, 2
        H_pad, W_pad = 6, 7
        k_h, k_w = 2, 3
        s_h, s_w = 2, 1
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        rng = np.random.default_rng(123)
        x_pad = rng.standard_normal((N, C, H_pad, W_pad)).astype(dtype, copy=False)
        if not x_pad.flags["C_CONTIGUOUS"]:
            x_pad = np.ascontiguousarray(x_pad)

        _, idx_ref = _ref_maxpool2d_forward(
            x_pad,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
            H_out=H_out,
            W_out=W_out,
        )

        grad_out = rng.standard_normal((N, C, H_out, W_out)).astype(dtype, copy=False)
        if not grad_out.flags["C_CONTIGUOUS"]:
            grad_out = np.ascontiguousarray(grad_out)

        grad_x_ref = _ref_maxpool2d_backward(
            grad_out, idx_ref, H_pad=H_pad, W_pad=W_pad
        )

        grad_out_dev = cuda_from_host(self.lib, grad_out)
        idx_dev = cuda_from_host(self.lib, idx_ref.astype(np.int64, copy=False))

        grad_x_host = np.empty((N, C, H_pad, W_pad), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            # must be zero-initialized (kernel uses atomic adds)
            cuda_memset(self.lib, grad_x_dev, value=0, nbytes=grad_x_host.nbytes)

            maxpool2d_backward_cuda(
                self.lib,
                grad_out_dev=grad_out_dev,
                argmax_idx_dev=idx_dev,
                grad_x_pad_dev=grad_x_dev,
                N=N,
                C=C,
                H_out=H_out,
                W_out=W_out,
                H_pad=H_pad,
                W_pad=W_pad,
                dtype=dtype,
                sync=True,
            )

            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=0.0, atol=0.0)

        finally:
            cuda_free(self.lib, grad_out_dev)
            cuda_free(self.lib, idx_dev)
            cuda_free(self.lib, grad_x_dev)

    def test_maxpool2d_forward_f32_matches_numpy(self) -> None:
        self._run_forward_case(np.float32)

    def test_maxpool2d_forward_f64_matches_numpy(self) -> None:
        self._run_forward_case(np.float64)

    def test_maxpool2d_backward_f32_matches_numpy(self) -> None:
        self._run_backward_case(np.float32)

    def test_maxpool2d_backward_f64_matches_numpy(self) -> None:
        self._run_backward_case(np.float64)

    def test_forward_argmax_in_bounds(self) -> None:
        dtype = np.float32
        N, C = 1, 1
        H_pad, W_pad = 5, 5
        k_h, k_w = 3, 3
        s_h, s_w = 1, 1
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        x_pad = np.zeros((N, C, H_pad, W_pad), dtype=dtype)
        x_dev = cuda_from_host(self.lib, x_pad)

        y_host = np.empty((N, C, H_out, W_out), dtype=dtype)
        idx_host = np.empty((N, C, H_out, W_out), dtype=np.int64)

        y_dev = cuda_malloc(self.lib, y_host.nbytes)
        idx_dev = cuda_malloc(self.lib, idx_host.nbytes)

        try:
            maxpool2d_forward_cuda(
                self.lib,
                x_pad_dev=x_dev,
                y_dev=y_dev,
                argmax_idx_dev=idx_dev,
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

            cuda_memcpy_d2h(self.lib, idx_host, idx_dev)
            self.assertGreaterEqual(int(idx_host.min()), 0)
            self.assertLess(int(idx_host.max()), H_pad * W_pad)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)


if __name__ == "__main__":
    unittest.main()
