# tests/infrastructure/ops/test_fill_cuda_ext.py
"""
Unit tests for CUDA Tensor-boundary fill wrapper (fill_cuda_ext.py).

These tests validate that the Tensor-facing fill utilities:
- Accept CUDA Tensors (device-pointer backed)
- Perform in-place initialization correctly (fill_/zeros_/ones_)
- Allocate + initialize correctly for *_like helpers (full_like/zeros_like/ones_like)
- Produce numerically correct results vs NumPy reference
- Raise appropriate errors for invalid inputs (CPU tensor, non-float dtype)

The tests are written with unittest and are designed to skip on CPU-only
environments where the KeyDNN CUDA native DLL cannot be loaded.
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

from src.keydnn.infrastructure.ops.fill_cuda_ext import (
    fill_,
    zeros_,
    ones_,
    full_like,
    zeros_like,
    ones_like,
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


class TestFillCudaExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Try to load CUDA DLL; skip all tests if unavailable.
        try:
            cls.lib = _load_cuda_lib()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL not available: {e}") from e

        cls.device_index = 0
        try:
            cuda_set_device(cls.lib, cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to set CUDA device: {e}") from e

        # Ensure memcpy symbols exist; if not, skip (cannot validate numerics).
        try:
            _bind_memcpy_symbols(cls.lib)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA memcpy symbols unavailable: {e}") from e

        # Construct a CUDA Device object for Tensor wrapping.
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
            dev_ptr,
            shape=tuple(int(d) for d in x_c.shape),
            dtype=x_c.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )

    def _make_empty_cuda_tensor(
        self, shape: Tuple[int, ...], dtype: np.dtype
    ) -> Tensor:
        """
        Allocate an uninitialized CUDA tensor buffer of given shape/dtype (no H2D copy).
        """
        shp = tuple(int(d) for d in shape)
        dt = np.dtype(dtype)
        n = int(np.prod(shp)) if len(shp) > 0 else 1
        nbytes = int(n) * int(dt.itemsize)
        dev_ptr = int(cuda_malloc(self.lib, int(nbytes))) if nbytes != 0 else 0
        return Tensor._from_devptr(
            dev_ptr,
            shape=shp,
            dtype=dt,
            device=self.cuda_device,
            requires_grad=False,
        )

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        """Copy CUDA Tensor data back to host NumPy."""
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        if host.nbytes == 0:
            return host
        _d2h(self.lib, int(t.data), host)
        return host

    # ----------------------------
    # In-place ops
    # ----------------------------

    def test_fill_inplace_f32(self) -> None:
        x = np.empty((32,), dtype=np.float32)
        t = self._make_cuda_tensor_from_host(x)

        fill_(t, 3.25, device=self.device_index, sync=True)

        out = self._read_cuda_tensor_to_host(t)
        ref = np.full_like(x, 3.25, dtype=np.float32)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=0.0)

    def test_fill_inplace_f64(self) -> None:
        x = np.empty((7, 5), dtype=np.float64)
        t = self._make_cuda_tensor_from_host(x)

        fill_(t, -2.0, device=self.device_index, sync=True)

        out = self._read_cuda_tensor_to_host(t)
        ref = np.full_like(x, -2.0, dtype=np.float64)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=0.0)

    def test_zeros_inplace_f32(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16,), dtype=np.float32)
        t = self._make_cuda_tensor_from_host(x)

        zeros_(t, device=self.device_index, sync=True)

        out = self._read_cuda_tensor_to_host(t)
        np.testing.assert_allclose(out, np.zeros_like(x), rtol=0.0, atol=0.0)

    def test_ones_inplace_f64(self) -> None:
        # This also implicitly covers the ones_cuda fallback path if fill kernel is unstable.
        x = np.empty((9,), dtype=np.float64)
        t = self._make_cuda_tensor_from_host(x)

        ones_(t, device=self.device_index, sync=True)

        out = self._read_cuda_tensor_to_host(t)
        np.testing.assert_allclose(out, np.ones_like(x), rtol=0.0, atol=0.0)

    # ----------------------------
    # Allocate + initialize helpers
    # ----------------------------

    def test_zeros_like_allocates_and_zeroes(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 6), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = zeros_like(x_t, device=self.device_index, sync=True)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, np.zeros_like(x), rtol=0.0, atol=0.0)

    def test_ones_like_allocates_and_ones(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((3, 3, 2), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = ones_like(x_t, device=self.device_index, sync=True)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, np.ones_like(x), rtol=0.0, atol=0.0)

    def test_full_like_allocates_and_fills(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.standard_normal((5,), dtype=np.float64)
        x_t = self._make_cuda_tensor_from_host(x)

        val = 0.125
        y_t = full_like(x_t, val, device=self.device_index, sync=True)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = np.full_like(x, val, dtype=np.float64)
        np.testing.assert_allclose(y, ref, rtol=0.0, atol=0.0)

    # ----------------------------
    # Edge cases
    # ----------------------------

    def test_fill_empty_tensor_noop(self) -> None:
        # numel == 0 should be a no-op; main check is "does not crash".
        # We don't memcpy back because size is 0.
        t = self._make_empty_cuda_tensor((0,), np.float32)
        fill_(t, 7.0, device=self.device_index, sync=True)
        self.assertEqual(tuple(t.shape), (0,))
        self.assertEqual(np.dtype(t.dtype), np.float32)

    # ----------------------------
    # Error handling
    # ----------------------------

    def test_fill_raises_on_non_float_dtype(self) -> None:
        # Construct int32 CUDA tensor and assert fill_ rejects it.
        x = np.empty((8,), dtype=np.int32)
        t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(TypeError):
            _ = fill_(t, 1.0, device=self.device_index, sync=True)

    def test_fill_raises_on_cpu_tensor(self) -> None:
        x_np = np.ones((3,), dtype=np.float32)

        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            x_cpu = Tensor.from_numpy(x_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                x_cpu = Tensor(x_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                x_cpu._data = x_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensors in this repo; update test_fill_raises_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = fill_(x_cpu, 2.0, device=self.device_index, sync=True)


if __name__ == "__main__":
    unittest.main()
