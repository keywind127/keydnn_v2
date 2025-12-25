"""
Unit tests for KeyDNN v2 CUDA Pad2D/Crop2D ctypes bindings.

These tests validate that:
- pad2d_cuda correctly constructs padded tensors on the GPU for float32/float64
- crop2d_cuda correctly extracts the unpadded region on the GPU
- pad then crop recovers the original input exactly (within dtype tolerance)
- padding values are correctly filled (-inf and 0 cases)

Notes
-----
- Requires the CUDA native DLL to be present and loadable.
- Uses only ctypes-level CUDA malloc/memcpy/memset to avoid higher-level Tensor code.
"""

from __future__ import annotations

import unittest
from typing import Tuple

import numpy as np

from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_memcpy_h2d,
    cuda_memcpy_d2h,
    cuda_memset,
)
from src.keydnn.infrastructure.native_cuda.python.pad2d_cuda_ctypes import (
    pad2d_cuda,
    crop2d_cuda,
)


def _pad2d_ref(x: np.ndarray, p_h: int, p_w: int, pad_value: float) -> np.ndarray:
    """NumPy reference for NCHW pad2d."""
    return np.pad(
        x,
        pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=pad_value,
    ).astype(x.dtype, copy=False)


def _crop2d_ref(x_pad: np.ndarray, p_h: int, p_w: int, H: int, W: int) -> np.ndarray:
    """NumPy reference for NCHW crop2d."""
    return x_pad[:, :, p_h : p_h + H, p_w : p_w + W].astype(x_pad.dtype, copy=False)


class TestPad2DCudaCtypes(unittest.TestCase):
    """
    Tests for pad2d/crop2d CUDA kernels exposed via ctypes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = load_keydnn_cuda_native()
        cuda_set_device(cls.lib, 0)

    def _alloc_and_copy_in(self, x: np.ndarray) -> int:
        """Allocate device buffer and copy x into it. Returns DevPtr (int)."""
        dev = cuda_malloc(self.lib, int(x.nbytes))
        cuda_memcpy_h2d(self.lib, dev, x)
        return int(dev)

    def _alloc_out(self, nbytes: int, *, zero: bool = False) -> int:
        """Allocate device buffer; optionally zero it."""
        dev = cuda_malloc(self.lib, int(nbytes))
        if zero:
            cuda_memset(self.lib, dev, 0, int(nbytes))
        return int(dev)

    def _copy_out(
        self, dev_ptr: int, shape: Tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        """Copy device buffer into a host NumPy array."""
        out = np.empty(shape, dtype=dtype)
        cuda_memcpy_d2h(self.lib, out, dev_ptr)
        return out

    def _run_pad_case(
        self,
        dtype: np.dtype,
        shape: Tuple[int, int, int, int],
        p_h: int,
        p_w: int,
        pad_value: float,
    ) -> None:
        N, C, H, W = shape
        x = np.random.randn(N, C, H, W).astype(dtype, copy=False)

        H_pad = H + 2 * p_h
        W_pad = W + 2 * p_w
        y_ref = _pad2d_ref(x, p_h, p_w, pad_value)

        x_dev = self._alloc_and_copy_in(x)
        y_pad_dev = self._alloc_out(int(y_ref.nbytes), zero=True)

        try:
            pad2d_cuda(
                self.lib,
                x_dev=x_dev,
                y_pad_dev=y_pad_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                p_h=p_h,
                p_w=p_w,
                pad_value=pad_value,
                dtype=dtype,
                device=0,
                sync=True,
            )

            y_pad = self._copy_out(y_pad_dev, (N, C, H_pad, W_pad), dtype)

            # tolerances
            if dtype == np.float32:
                rtol, atol = 1e-6, 1e-6
            else:
                rtol, atol = 1e-12, 1e-12

            # Special handling if pad_value is -inf: comparisons still work with assert_allclose
            np.testing.assert_allclose(y_pad, y_ref, rtol=rtol, atol=atol)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_pad_dev)

    def _run_crop_case(
        self,
        dtype: np.dtype,
        shape: Tuple[int, int, int, int],
        p_h: int,
        p_w: int,
        pad_value: float,
    ) -> None:
        """
        Crop test: run pad2d then crop2d, verify recovered x.
        """
        N, C, H, W = shape
        x = np.random.randn(N, C, H, W).astype(dtype, copy=False)

        H_pad = H + 2 * p_h
        W_pad = W + 2 * p_w

        x_pad_ref = _pad2d_ref(x, p_h, p_w, pad_value)
        x_ref = _crop2d_ref(x_pad_ref, p_h, p_w, H, W)

        x_dev = self._alloc_and_copy_in(x)

        x_pad_dev = self._alloc_out(int(x_pad_ref.nbytes), zero=True)
        y_dev = self._alloc_out(int(x.nbytes), zero=True)

        try:
            # pad on GPU
            pad2d_cuda(
                self.lib,
                x_dev=x_dev,
                y_pad_dev=x_pad_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                p_h=p_h,
                p_w=p_w,
                pad_value=pad_value,
                dtype=dtype,
                device=0,
                sync=True,
            )

            # crop on GPU
            crop2d_cuda(
                self.lib,
                x_pad_dev=x_pad_dev,
                y_dev=y_dev,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                p_h=p_h,
                p_w=p_w,
                H=H,
                W=W,
                dtype=dtype,
                device=0,
                sync=True,
            )

            y = self._copy_out(y_dev, (N, C, H, W), dtype)

            if dtype == np.float32:
                rtol, atol = 1e-6, 1e-6
            else:
                rtol, atol = 1e-12, 1e-12

            np.testing.assert_allclose(y, x_ref, rtol=rtol, atol=atol)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, x_pad_dev)
            cuda_free(self.lib, y_dev)

    # -------------------------
    # Actual tests
    # -------------------------

    def test_pad2d_f32_zero_padding_matches_numpy(self) -> None:
        self._run_pad_case(
            np.float32,
            shape=(2, 3, 5, 7),
            p_h=1,
            p_w=2,
            pad_value=0.0,
        )

    def test_pad2d_f64_zero_padding_matches_numpy(self) -> None:
        self._run_pad_case(
            np.float64,
            shape=(1, 2, 4, 6),
            p_h=2,
            p_w=1,
            pad_value=0.0,
        )

    def test_pad2d_f32_neg_inf_padding_matches_numpy(self) -> None:
        self._run_pad_case(
            np.float32,
            shape=(1, 1, 3, 4),
            p_h=1,
            p_w=1,
            pad_value=float("-inf"),
        )

    def test_pad2d_f64_neg_inf_padding_matches_numpy(self) -> None:
        self._run_pad_case(
            np.float64,
            shape=(2, 1, 2, 3),
            p_h=3,
            p_w=2,
            pad_value=float("-inf"),
        )

    def test_crop2d_roundtrip_pad_then_crop_f32(self) -> None:
        self._run_crop_case(
            np.float32,
            shape=(2, 2, 8, 5),
            p_h=1,
            p_w=3,
            pad_value=0.0,
        )

    def test_crop2d_roundtrip_pad_then_crop_f64(self) -> None:
        self._run_crop_case(
            np.float64,
            shape=(1, 4, 7, 7),
            p_h=2,
            p_w=2,
            pad_value=float("-inf"),
        )

    def test_pad2d_no_padding_identity(self) -> None:
        """
        With p_h=p_w=0, pad2d should be an identity copy.
        """
        dtype = np.float32
        N, C, H, W = (2, 3, 4, 5)
        x = np.random.randn(N, C, H, W).astype(dtype, copy=False)

        x_dev = self._alloc_and_copy_in(x)
        y_dev = self._alloc_out(int(x.nbytes), zero=True)

        try:
            pad2d_cuda(
                self.lib,
                x_dev=x_dev,
                y_pad_dev=y_dev,
                N=N,
                C=C,
                H=H,
                W=W,
                p_h=0,
                p_w=0,
                pad_value=123.0,  # should not matter when no padding
                dtype=dtype,
                device=0,
                sync=True,
            )

            y = self._copy_out(y_dev, (N, C, H, W), dtype)
            np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-6)

        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

    def test_invalid_dtype_raises(self) -> None:
        """
        pad2d_cuda/crop2d_cuda should reject non-f32/f64.
        """
        # We won't actually run CUDA here; just verify the python wrapper rejects dtype.
        with self.assertRaises(TypeError):
            pad2d_cuda(
                self.lib,
                x_dev=1,
                y_pad_dev=2,
                N=1,
                C=1,
                H=1,
                W=1,
                p_h=1,
                p_w=1,
                pad_value=0.0,
                dtype=np.float16,  # unsupported
                device=0,
                sync=False,
            )

        with self.assertRaises(TypeError):
            crop2d_cuda(
                self.lib,
                x_pad_dev=1,
                y_dev=2,
                N=1,
                C=1,
                H_pad=3,
                W_pad=3,
                p_h=1,
                p_w=1,
                H=1,
                W=1,
                dtype=np.int32,  # unsupported
                device=0,
                sync=False,
            )
