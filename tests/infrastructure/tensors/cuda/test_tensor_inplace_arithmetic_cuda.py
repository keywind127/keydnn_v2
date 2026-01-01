# tests/infrastructure/tensors/test_tensor_inplace_arithmetic_cuda.py
"""
Unit tests for CUDA-backed in-place Tensor arithmetic:

Covers:
- Tensor.__iadd__   (Tensor += Tensor, Tensor += scalar)
- Tensor.__isub__   (Tensor -= Tensor, Tensor -= scalar)
- Tensor.__itruediv__ (Tensor /= Tensor, Tensor /= scalar)
- Tensor.__idiv__   (legacy alias for in-place division; explicit call)

Validates:
- In-place mutation updates the *same device buffer* (data pointer unchanged)
- Numerical correctness vs NumPy reference
- Proper dtype/shape checks (mismatch raises)
- Mixed-device inputs raise
- numel==0 is a no-op and does not error

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
    if src_c.nbytes == 0:
        return
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
    if dst_c.nbytes == 0:
        return
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
    """Best-effort creation of a CPU Tensor from a NumPy array for error-path tests."""
    x_c = np.ascontiguousarray(x)
    try:
        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            return Tensor.from_numpy(x_c, device=Device("cpu"))  # type: ignore[call-arg]
    except Exception:
        pass

    t = Tensor(
        shape=tuple(int(d) for d in x_c.shape),
        device=Device("cpu"),
        requires_grad=False,
    )  # type: ignore[call-arg]
    setattr(t, "_data", x_c)
    return t


class TestTensorInplaceArithmeticCuda(unittest.TestCase):
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
    # In-place: add
    # ----------------------------
    def test_iadd_tensor_tensor_cuda_f32_mutates_inplace(self) -> None:
        rng = np.random.default_rng(0)
        a0 = rng.standard_normal((64,), dtype=np.float32)
        b = rng.standard_normal((64,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a0)
        b_t = self._make_cuda_tensor_from_host(b)

        ptr_before = int(a_t.data)
        a_t += b_t
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "iadd must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 + b, rtol=1e-4, atol=1e-4)

    def test_iadd_tensor_scalar_cuda_f64_mutates_inplace(self) -> None:
        rng = np.random.default_rng(1)
        a0 = rng.standard_normal((9, 7)).astype(np.float64)
        alpha = 1.25

        a_t = self._make_cuda_tensor_from_host(a0)
        ptr_before = int(a_t.data)

        a_t += alpha
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "iadd scalar must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 + alpha, rtol=1e-10, atol=1e-10)

    # ----------------------------
    # In-place: sub
    # ----------------------------
    def test_isub_tensor_tensor_cuda_f32_mutates_inplace(self) -> None:
        rng = np.random.default_rng(2)
        a0 = rng.standard_normal((33,), dtype=np.float32)
        b = rng.standard_normal((33,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a0)
        b_t = self._make_cuda_tensor_from_host(b)

        ptr_before = int(a_t.data)
        a_t -= b_t
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "isub must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 - b, rtol=1e-4, atol=1e-4)

    def test_isub_tensor_scalar_cuda_f32_mutates_inplace(self) -> None:
        rng = np.random.default_rng(3)
        a0 = rng.standard_normal((4, 5), dtype=np.float32)
        alpha = 0.75

        a_t = self._make_cuda_tensor_from_host(a0)
        ptr_before = int(a_t.data)

        a_t -= alpha
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "isub scalar must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 - np.float32(alpha), rtol=1e-4, atol=1e-4)

    # ----------------------------
    # In-place: div
    # ----------------------------
    def test_itruediv_tensor_tensor_cuda_f32_mutates_inplace(self) -> None:
        rng = np.random.default_rng(4)
        a0 = rng.standard_normal((128,), dtype=np.float32)
        b = rng.standard_normal((128,), dtype=np.float32)
        b = np.where(np.abs(b) < 1e-3, np.float32(1.0), b)

        a_t = self._make_cuda_tensor_from_host(a0)
        b_t = self._make_cuda_tensor_from_host(b)

        ptr_before = int(a_t.data)
        a_t /= b_t
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "itruediv must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 / b, rtol=1e-3, atol=1e-3)

    def test_itruediv_tensor_scalar_cuda_f64_mutates_inplace(self) -> None:
        rng = np.random.default_rng(5)
        a0 = rng.standard_normal((7, 9)).astype(np.float64)
        alpha = 1.6

        a_t = self._make_cuda_tensor_from_host(a0)
        ptr_before = int(a_t.data)

        a_t /= alpha
        ptr_after = int(a_t.data)

        self.assertEqual(
            ptr_after, ptr_before, "itruediv scalar must not replace storage"
        )
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 / alpha, rtol=1e-12, atol=1e-12)

    def test_idiv_legacy_alias_calls_itruediv(self) -> None:
        rng = np.random.default_rng(6)
        a0 = rng.standard_normal((16,), dtype=np.float32)
        alpha = 2.0

        a_t = self._make_cuda_tensor_from_host(a0)
        ptr_before = int(a_t.data)

        # explicit legacy in-place alias
        a_t.__idiv__(alpha)  # type: ignore[attr-defined]
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before, "__idiv__ must not replace storage")
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a0 / np.float32(alpha), rtol=1e-3, atol=1e-3)

    # ----------------------------
    # numel==0 should be no-op
    # ----------------------------
    def test_inplace_numel_zero_is_ok_tensor_tensor(self) -> None:
        # Empty tensor => dev_ptr may be 0; inplace should early-return without native call.
        a0 = np.empty((0,), dtype=np.float32)
        b0 = np.empty((0,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a0)
        b_t = self._make_cuda_tensor_from_host(b0)

        ptr_before = int(a_t.data)
        a_t += b_t
        a_t -= b_t
        a_t /= b_t + 1.0  # keep it well-defined, still empty
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before)
        out = self._read_cuda_tensor_to_host(a_t)
        self.assertEqual(out.size, 0)

    def test_inplace_numel_zero_is_ok_scalar(self) -> None:
        a0 = np.empty((0,), dtype=np.float64)
        a_t = self._make_cuda_tensor_from_host(a0)

        ptr_before = int(a_t.data)
        a_t += 1.0
        a_t -= 2.0
        a_t /= 3.0
        ptr_after = int(a_t.data)

        self.assertEqual(ptr_after, ptr_before)
        out = self._read_cuda_tensor_to_host(a_t)
        self.assertEqual(out.size, 0)

    # ----------------------------
    # Error cases
    # ----------------------------
    def test_iadd_raises_on_shape_mismatch(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8,), dtype=np.float32)
        b = rng.standard_normal((9,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            a_t += b_t

    def test_isub_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((10,), dtype=np.float32)
        b = rng.standard_normal((10,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            a_t -= b_t

    def test_itruediv_raises_on_mixed_devices(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.standard_normal((6,), dtype=np.float32)
        b = rng.standard_normal((6,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_cpu = _make_cpu_tensor_from_numpy(b)

        with self.assertRaises(Exception):
            a_t /= b_cpu  # type: ignore[operator]

    def test_iadd_rejects_non_float_dtype(self) -> None:
        a = np.arange(16, dtype=np.int32)
        a_t = self._make_cuda_tensor_from_host(a)

        with self.assertRaises(TypeError):
            a_t += 1  # scalar path should reject int32 tensor dtype

    def test_itruediv_rejects_non_float_dtype(self) -> None:
        a = np.arange(16, dtype=np.int32)
        a_t = self._make_cuda_tensor_from_host(a)

        with self.assertRaises(TypeError):
            a_t /= 2  # scalar path should reject int32 tensor dtype


if __name__ == "__main__":
    unittest.main()
