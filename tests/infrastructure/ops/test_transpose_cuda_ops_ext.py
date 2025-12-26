# tests/infrastructure/ops/test_transpose_cuda_ext.py
"""
Unit tests for CUDA Tensor-boundary transpose wrapper (transpose_cuda_ext.py).

These tests validate that the Tensor-facing wrapper:
- Accepts CUDA Tensors (device-pointer backed)
- Allocates output device memory
- Calls the underlying ops-layer transpose2d_cuda
- Returns a CUDA Tensor with correct shape/dtype
- Produces numerically correct results vs NumPy reference
- Raises appropriate errors for invalid inputs

The tests are written with unittest and are designed to skip on CPU-only
environments where the KeyDNN CUDA native DLL cannot be loaded.
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
from src.keydnn.infrastructure.ops.transpose_cuda_ext import transpose2d_forward


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


class TestTransposeCudaExt(unittest.TestCase):
    """
    Tests for Tensor-boundary CUDA transpose wrapper.

    Wrapper under test:
        keydnn.infrastructure.ops.transpose_cuda_ext.transpose2d_forward
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

    def test_transpose2d_forward_f32_matches_numpy(self) -> None:
        """float32 transpose matches NumPy reference."""
        rng = np.random.default_rng(0)
        rows, cols = 9, 13
        x = rng.standard_normal((rows, cols), dtype=np.float32)

        x_t = self._make_cuda_tensor_from_host(x)
        y_t = transpose2d_forward(x_t, device=self.device_index, sync=True)

        self.assertTrue(y_t.device.is_cuda(), "output should be a CUDA tensor")
        self.assertEqual(tuple(y_t.shape), (cols, rows))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = x.T
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_transpose2d_forward_f64_matches_numpy(self) -> None:
        """float64 transpose matches NumPy reference."""
        rng = np.random.default_rng(1)
        rows, cols = 7, 5
        x = rng.standard_normal((rows, cols)).astype(np.float64)

        x_t = self._make_cuda_tensor_from_host(x)
        y_t = transpose2d_forward(x_t, device=self.device_index, sync=True)

        self.assertTrue(y_t.device.is_cuda(), "output should be a CUDA tensor")
        self.assertEqual(tuple(y_t.shape), (cols, rows))
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = x.T
        np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)

    def test_raises_on_non_2d(self) -> None:
        """Non-2D input should raise ValueError."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((2, 3, 4), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(ValueError):
            _ = transpose2d_forward(x_t, device=self.device_index, sync=True)

    def test_raises_on_unsupported_dtype(self) -> None:
        """Unsupported dtype should raise TypeError."""
        # Make an int32 host buffer and wrap it; wrapper should reject dtype.
        x = np.arange(12, dtype=np.int32).reshape(3, 4)
        x_t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(TypeError):
            _ = transpose2d_forward(x_t, device=self.device_index, sync=True)

    def test_raises_on_cpu_tensor(self) -> None:
        """Passing a CPU tensor should raise TypeError."""
        x_np = np.ones((2, 3), dtype=np.float32)

        # CPU tensor creation varies across repos; attempt common patterns.
        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            x_cpu = Tensor.from_numpy(x_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                x_cpu = Tensor(x_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                x_cpu._data = x_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensor in this repo; update test_raises_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = transpose2d_forward(x_cpu, device=self.device_index, sync=True)

    def test_output_is_contiguous_in_memory_shape(self) -> None:
        """
        Basic sanity check: output buffer size equals rows*cols*itemsize
        and can be copied back without errors.
        """
        rng = np.random.default_rng(3)
        rows, cols = 16, 8
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = transpose2d_forward(x_t, device=self.device_index, sync=True)
        y_host = self._read_cuda_tensor_to_host(y_t)

        self.assertEqual(y_host.shape, (cols, rows))
        # Ensure values look reasonable (no NaN storms / uninitialized copies)
        self.assertFalse(np.isnan(y_host).any())


if __name__ == "__main__":
    unittest.main()
