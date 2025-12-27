"""
Unit tests for CUDA Tensor comparison + negation operators.

Covers (CUDA path):
- __neg__
- __gt__
- __ge__   (implemented via gt + arithmetic)
- __lt__   (implemented via gt)
- __le__   (implemented via gt + arithmetic)

These tests validate:
- Works on CUDA tensors (device-pointer backed)
- Output shapes/dtypes are correct
- Numerically matches NumPy reference
- Comparison outputs are float32 masks with values in {0.0, 1.0}

Skips on CPU-only environments where the KeyDNN CUDA native DLL cannot be loaded.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)


def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.
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

    Expected native signatures:
        int keydnn_cuda_memcpy_h2d(uint64_t dst_dev, void* src_host, size_t nbytes)
        int keydnn_cuda_memcpy_d2h(void* dst_host, uint64_t src_dev, size_t nbytes)
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
    """Copy a contiguous host array to device memory."""
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
    """Copy device memory into a contiguous host array."""
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


class TestTensorComparisonsNegCuda(unittest.TestCase):
    """
    CUDA tests for Tensor negation + comparisons.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = _load_cuda_lib()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL not available: {e}") from e

        cls.device_index = 0
        try:
            cuda_set_device(cls.lib, cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to set CUDA device: {e}") from e

        try:
            _bind_memcpy_symbols(cls.lib)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA memcpy symbols unavailable: {e}") from e

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

    # ----------------------------
    # __neg__
    # ----------------------------
    def test_neg_cuda_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 7), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = -x_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, -x, rtol=1e-5, atol=1e-5)

    def test_neg_cuda_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 5)).astype(np.float64)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = -x_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, -x, rtol=1e-12, atol=1e-12)

    # ----------------------------
    # __gt__ / __ge__ / __lt__ / __le__
    # ----------------------------
    def _assert_mask(self, y: np.ndarray) -> None:
        self.assertEqual(y.dtype, np.float32)
        # mask should be exactly 0.0 or 1.0
        u = np.unique(y)
        self.assertTrue(
            set(u.tolist()).issubset({0.0, 1.0}), f"mask values not in {{0,1}}: {u}"
        )

    def test_gt_tensor_tensor_cuda_f32(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((5, 6), dtype=np.float32)
        b = rng.standard_normal((5, 6), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t > b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)

        ref = (a > b).astype(np.float32)
        np.testing.assert_array_equal(y, ref)

    def test_gt_tensor_scalar_cuda_f32(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((4, 4), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)

        # _as_tensor_like() rejects numpy scalar types; use a Python float.
        s = 0.25
        y_t = a_t > s

        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)
        ref = (a > np.float32(s)).astype(
            np.float32
        )  # match float32 comparison semantics
        np.testing.assert_array_equal(y, ref)

    def test_ge_tensor_tensor_cuda_f32(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((6, 3), dtype=np.float32)
        b = rng.standard_normal((6, 3), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t >= b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)
        ref = (a >= b).astype(np.float32)
        np.testing.assert_array_equal(y, ref)

    def test_lt_tensor_tensor_cuda_f32(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((2, 9), dtype=np.float32)
        b = rng.standard_normal((2, 9), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t < b_t
        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)
        ref = (a < b).astype(np.float32)
        np.testing.assert_array_equal(y, ref)

    def test_le_tensor_scalar_cuda_f32(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.standard_normal((7,), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)

        # _as_tensor_like() rejects numpy scalar types; use a Python float.
        s = -0.1
        y_t = a_t <= s

        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)
        ref = (a <= np.float32(s)).astype(np.float32)
        np.testing.assert_array_equal(y, ref)

    def test_gt_tensor_tensor_cuda_f64(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((3, 3)).astype(np.float64)
        b = rng.standard_normal((3, 3)).astype(np.float64)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t > b_t
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        self._assert_mask(y)
        ref = (a > b).astype(np.float32)
        np.testing.assert_array_equal(y, ref)

    def test_ge_lt_le_cuda_f64(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((4, 2)).astype(np.float64)
        b = rng.standard_normal((4, 2)).astype(np.float64)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        ge = self._read_cuda_tensor_to_host(a_t >= b_t)
        lt = self._read_cuda_tensor_to_host(a_t < b_t)
        le = self._read_cuda_tensor_to_host(a_t <= b_t)

        self._assert_mask(ge)
        self._assert_mask(lt)
        self._assert_mask(le)

        np.testing.assert_array_equal(ge, (a >= b).astype(np.float32))
        np.testing.assert_array_equal(lt, (a < b).astype(np.float32))
        np.testing.assert_array_equal(le, (a <= b).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
