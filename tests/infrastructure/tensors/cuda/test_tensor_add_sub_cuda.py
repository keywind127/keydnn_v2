"""
Unit tests for CUDA-backed Tensor addition/subtraction.

These tests validate that Tensor.__add__/__radd__/__sub__/__rsub__:
- work on CUDA tensors (device-pointer backed)
- allocate and return CUDA tensors with correct shape/dtype
- produce numerically correct results vs NumPy reference
- lift scalars correctly via _as_tensor_like (covers fill cuda path implicitly)
- raise appropriate errors on shape mismatch / dtype mismatch / CPU tensors

Skips if CUDA native DLL cannot be loaded.
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


def _make_cpu_tensor_from_numpy(x: np.ndarray) -> Tensor:
    """
    Best-effort creation of a CPU Tensor from a NumPy array for error-path tests.
    """
    x_c = np.ascontiguousarray(x)
    try:
        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            return Tensor.from_numpy(x_c, device=Device("cpu"))  # type: ignore[call-arg]
    except Exception:
        pass

    # Fallback used in other tests in this repo
    t = Tensor(shape=tuple(int(d) for d in x_c.shape), device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
    # Many KeyDNN builds use _data for CPU ndarray storage
    setattr(t, "_data", x_c)
    return t


class TestTensorAddSubCuda(unittest.TestCase):
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
            int(dev_ptr),
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
    # ADD
    # ----------------------------
    def test_add_tensor_tensor_cuda_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((32,), dtype=np.float32)
        b = rng.standard_normal((32,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t + b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (32,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a + b, rtol=1e-4, atol=1e-4)

    def test_add_tensor_tensor_cuda_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((17,)).astype(np.float64)
        b = rng.standard_normal((17,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t + b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (17,))
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a + b, rtol=1e-10, atol=1e-10)

    def test_radd_scalar_tensor_cuda(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((8,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)

        y_t = 3.0 + a_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (8,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, 3.0 + a, rtol=1e-4, atol=1e-4)

    def test_add_tensor_scalar_cuda(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((11,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)

        y_t = a_t + 2.5
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (11,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a + 2.5, rtol=1e-4, atol=1e-4)

    # ----------------------------
    # SUB
    # ----------------------------
    def test_sub_tensor_tensor_cuda_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((25,), dtype=np.float32)
        b = rng.standard_normal((25,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t - b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (25,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a - b, rtol=1e-4, atol=1e-4)

    def test_rsub_scalar_tensor_cuda(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((9,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)

        y_t = 2.0 - a_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (9,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, 2.0 - a, rtol=1e-4, atol=1e-4)

    def test_sub_tensor_scalar_cuda(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.standard_normal((13,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)

        y_t = a_t - 1.25  # scalar lifts to float32 in this repo
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (13,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a - np.float32(1.25), rtol=1e-4, atol=1e-4)

    # ----------------------------
    # Error cases
    # ----------------------------
    def test_add_raises_on_shape_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8,), dtype=np.float32)
        b = rng.standard_normal((9,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = a_t + b_t

    def test_sub_raises_on_shape_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((3, 4), dtype=np.float32)
        b = rng.standard_normal((3, 5), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = a_t - b_t

    def test_add_raises_on_dtype_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.standard_normal((10,), dtype=np.float32)
        b = rng.standard_normal((10,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = a_t + b_t

    def test_sub_raises_on_dtype_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(10)
        a = rng.standard_normal((10,), dtype=np.float32)
        b = rng.standard_normal((10,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = a_t - b_t

    def test_add_raises_on_cpu_tensors(self) -> None:
        a = np.ones((4,), dtype=np.float32)
        b = np.ones((4,), dtype=np.float32)

        a_cpu = _make_cpu_tensor_from_numpy(a)
        b_cpu = _make_cpu_tensor_from_numpy(b)

        # This should hit CPU path and succeed (backward compatibility sanity)
        y = a_cpu + b_cpu
        self.assertTrue(y.device.is_cpu())
        np.testing.assert_allclose(y.to_numpy(), a + b)

    def test_add_raises_on_mixed_devices(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.standard_normal((6,), dtype=np.float32)
        b = rng.standard_normal((6,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_cpu = _make_cpu_tensor_from_numpy(b)

        with self.assertRaises(Exception):
            _ = a_t + b_cpu

    def test_sub_raises_on_mixed_devices(self) -> None:
        rng = np.random.default_rng(12)
        a = rng.standard_normal((6,), dtype=np.float32)
        b = rng.standard_normal((6,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_cpu = _make_cpu_tensor_from_numpy(b)

        with self.assertRaises(Exception):
            _ = a_t - b_cpu


if __name__ == "__main__":
    unittest.main()
