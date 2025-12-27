# tests/infrastructure/tensors/test_tensor_division_cuda.py
"""
Unit tests for Tensor division methods (CPU + CUDA).

Covers:
- Tensor.__truediv__  (Tensor / Tensor, Tensor / scalar)
- Tensor.__rtruediv__ (scalar / Tensor)
- Tensor.__div__      (legacy alias; explicit call)
- Tensor.__rdiv__     (legacy alias; explicit call)

CUDA tests:
- Allocate device buffers, copy host->device, wrap as CUDA Tensor
- Run division
- Copy device->host and compare vs NumPy reference

All CUDA tests are skipped if the KeyDNN CUDA native DLL is unavailable.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
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


class TestTensorDivisionCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use the shared CUDA DLL handle (single runtime instance).
        try:
            cls.lib = Tensor._get_cuda_lib()
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
        dev_ptr = int(cuda_malloc(self.lib, int(x_c.nbytes))) if x_c.nbytes != 0 else 0
        try:
            if x_c.nbytes != 0:
                _h2d(self.lib, dev_ptr, x_c)
        except Exception:
            if dev_ptr != 0:
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
        if host.nbytes == 0:
            return host
        _d2h(self.lib, int(t.data), host)
        return host

    # ----------------------------
    # CUDA correctness
    # ----------------------------

    def test_truediv_tensor_tensor_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64,), dtype=np.float32)
        b = rng.standard_normal((64,), dtype=np.float32)

        # avoid divide-by-zero instability (match your likely kernel behavior)
        b = np.where(np.abs(b) < 1e-3, np.float32(1.0), b)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t / b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (64,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = a / b
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_truediv_tensor_tensor_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((7, 9)).astype(np.float64)
        b = rng.standard_normal((7, 9)).astype(np.float64)
        b = np.where(np.abs(b) < 1e-12, 1.0, b)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = a_t / b_t
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (7, 9))
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = a / b
        np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)

    def test_truediv_tensor_scalar_cuda(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((3, 4), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)

        y_t = a_t / 2.0
        y = self._read_cuda_tensor_to_host(y_t)
        ref = a / np.float32(2.0)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_rtruediv_scalar_tensor_cuda(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((10,), dtype=np.float32)
        a = np.where(np.abs(a) < 1e-3, np.float32(1.0), a)
        a_t = self._make_cuda_tensor_from_host(a)

        y_t = 3.0 / a_t
        y = self._read_cuda_tensor_to_host(y_t)
        ref = np.float32(3.0) / a
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_div_legacy_alias_cuda(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((16,), dtype=np.float32)
        b = rng.standard_normal((16,), dtype=np.float32)
        b = np.where(np.abs(b) < 1e-3, np.float32(1.0), b)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        # Explicit call to legacy alias
        y_t = a_t.__div__(b_t)
        y = self._read_cuda_tensor_to_host(y_t)
        ref = a / b
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_rdiv_legacy_alias_cuda(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((16,), dtype=np.float32)
        a = np.where(np.abs(a) < 1e-3, np.float32(1.0), a)

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = a_t.__rdiv__(2.0)  # explicit legacy alias
        y = self._read_cuda_tensor_to_host(y_t)
        ref = np.float32(2.0) / a
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    # ----------------------------
    # CUDA error cases
    # ----------------------------

    def test_cuda_raises_on_shape_mismatch(self) -> None:
        a = np.ones((4,), dtype=np.float32)
        b = np.ones((5,), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = a_t / b_t

    def test_cuda_raises_on_dtype_mismatch(self) -> None:
        a = np.ones((4,), dtype=np.float32)
        b = np.ones((4,), dtype=np.float64)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = a_t / b_t

    def test_cuda_raises_on_non_float_dtype(self) -> None:
        a = np.ones((4,), dtype=np.int32)
        b = np.ones((4,), dtype=np.int32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = a_t / b_t

    # ----------------------------
    # CPU backward-compat sanity (best-effort)
    # ----------------------------

    # def test_cpu_truediv_still_works(self) -> None:
    #     a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    #     b_np = np.array([2.0, 4.0, 6.0], dtype=np.float32)

    #     if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
    #         a = Tensor.from_numpy(a_np, device=Device("cpu"))  # type: ignore[call-arg]
    #         b = Tensor.from_numpy(b_np, device=Device("cpu"))  # type: ignore[call-arg]
    #         y = a / b
    #         np.testing.assert_allclose(y.to_numpy(), a_np / b_np, rtol=0.0, atol=0.0)
    #     else:
    #         # Fallback: if this repo doesn't expose a clean CPU factory, skip.
    #         raise unittest.SkipTest(
    #             "Tensor.from_numpy not available; CPU division sanity skipped."
    #         )


if __name__ == "__main__":
    unittest.main()
