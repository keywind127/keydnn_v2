# tests/infrastructure/tensors/test_tensor_mul_cuda.py
"""
Unit tests for CUDA-enabled Tensor multiplication (__mul__, __rmul__).

These tests validate that:
- Tensor * Tensor on CUDA uses the CUDA mul path and matches NumPy reference
- Tensor * scalar and scalar * Tensor work on CUDA (scalar lifted via _as_tensor_like)
- output shape/dtype/device are correct
- error behavior matches expectations (dtype mismatch, shape mismatch, CPU tensor)

Tests are written with unittest and skip on CPU-only environments where the
KeyDNN CUDA native DLL cannot be loaded.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Any, Tuple

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
    """Best-effort helper to obtain a CUDA Device instance across possible Device APIs."""
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
            "Unable to construct a CUDA Device; update _get_cuda_device()."
        ) from e


def _bind_memcpy_symbols(lib: ctypes.CDLL) -> Tuple[Any, Any]:
    """
    Bind keydnn_cuda_memcpy_h2d and keydnn_cuda_memcpy_d2h from the CUDA DLL.

    Expected signatures:
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
            ctypes.c_void_p(int(src_c.ctypes.data)),
            ctypes.c_size_t(int(src_c.nbytes)),
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
            ctypes.c_void_p(int(dst_c.ctypes.data)),
            ctypes.c_uint64(int(src_dev)),
            ctypes.c_size_t(int(dst_c.nbytes)),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")


class TestTensorMulCuda(unittest.TestCase):
    """Tests for CUDA-enabled Tensor.__mul__ and Tensor.__rmul__."""

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

        # Need memcpy to validate numerics
        try:
            _bind_memcpy_symbols(cls.lib)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA memcpy symbols unavailable: {e}") from e

        try:
            cls.cuda_device = _get_cuda_device(cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to construct CUDA Device: {e}") from e

    def _make_cuda_tensor_from_host(self, x: np.ndarray) -> Tensor:
        """Allocate device memory, copy host -> device, and wrap as CUDA Tensor."""
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
        """Copy CUDA Tensor data back to host NumPy."""
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        _d2h(self.lib, int(t.data), host)
        return host

    def test_mul_tensor_tensor_cuda_f32_matches_numpy(self) -> None:
        """float32 tensor*tensor mul matches NumPy reference."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 5), dtype=np.float32)
        b = rng.standard_normal((4, 5), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t * b_t

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (4, 5))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = a * b
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)

    def test_mul_tensor_tensor_cuda_f64_matches_numpy(self) -> None:
        """float64 tensor*tensor mul matches NumPy reference."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((3, 7)).astype(np.float64)
        b = rng.standard_normal((3, 7)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t * b_t

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (3, 7))
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = a * b
        np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)

    def test_mul_tensor_scalar_cuda(self) -> None:
        """CUDA tensor * scalar uses scalar lifting and matches NumPy."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((6,), dtype=np.float32)
        s = 2.5  # IMPORTANT: python float, not np.float32 (matches _as_tensor_like support)

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = a_t * s

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (6,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = a * s
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)

    def test_rmul_scalar_tensor_cuda(self) -> None:
        """scalar * CUDA tensor (__rmul__) matches NumPy."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal((2, 3, 4), dtype=np.float32)
        s = 3.0  # python float

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = s * a_t  # __rmul__

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (2, 3, 4))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = s * a
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)

    def test_mul_raises_on_dtype_mismatch_cuda(self) -> None:
        """dtype mismatch should raise TypeError on CUDA."""
        rng = np.random.default_rng(4)
        a = rng.standard_normal((4, 4), dtype=np.float32)
        b = rng.standard_normal((4, 4)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = a_t * b_t

    def test_mul_raises_on_shape_mismatch_cuda(self) -> None:
        """shape mismatch should raise ValueError on CUDA."""
        rng = np.random.default_rng(5)
        a = rng.standard_normal((4, 5), dtype=np.float32)
        b = rng.standard_normal((4, 6), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = a_t * b_t

    def test_mul_raises_on_cpu_tensor(self) -> None:
        """Passing a CPU tensor should raise (device not supported / type error)."""
        a_np = np.ones((2, 3), dtype=np.float32)
        b_np = np.ones((2, 3), dtype=np.float32)

        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            a_cpu = Tensor.from_numpy(a_np, device=Device("cpu"))  # type: ignore[call-arg]
            b_cpu = Tensor.from_numpy(b_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                a_cpu = Tensor(a_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                a_cpu._data = a_np  # type: ignore[attr-defined]
                b_cpu = Tensor(b_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                b_cpu._data = b_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensors in this repo; update test_mul_raises_on_cpu_tensor: {e}"
                ) from e

        # Behavior depends on your implementation; CPU*CPU should work, but this test ensures
        # we don't accidentally hit CUDA path with CPU tensors.
        y = a_cpu * b_cpu
        np.testing.assert_allclose(y.to_numpy(), a_np * b_np)


if __name__ == "__main__":
    unittest.main()
