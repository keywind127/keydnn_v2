# tests/infrastructure/tensors/test_tensor_imul_cuda.py
"""
Unit tests for CUDA-enabled Tensor.__imul__ (in-place multiply).

We validate:
- Tensor *= Tensor on CUDA mutates the LHS storage and matches NumPy
- Tensor *= scalar on CUDA mutates the LHS storage and matches NumPy
- __imul__ returns self (same object)
- dtype/shape mismatch raises
- safe fallback path still works when in-place is not taken (best-effort):
  - if requires_grad=True OR ctx exists, __imul__ should still produce correct
    result via out-of-place + copy_from.

Notes
-----
These tests skip on CPU-only environments where the KeyDNN CUDA native DLL
cannot be loaded.

Also, CUDA malloc(0) is not supported in your runtime (it raises), so for
"numel==0 is no-op" we construct an empty CUDA Tensor with data=0 (no alloc).
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


class TestTensorIMulCuda(unittest.TestCase):
    """Tests for CUDA-enabled Tensor.__imul__."""

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

    # -----------------------
    # Happy paths (CUDA in-place)
    # -----------------------

    def test_imul_tensor_tensor_cuda_mutates_self_f32(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 5), dtype=np.float32)
        b = rng.standard_normal((4, 5), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        old_ptr = int(a_t.data)
        old_id = id(a_t)

        ret = a_t.__imul__(b_t)  # a_t *= b_t

        self.assertIs(ret, a_t)
        self.assertEqual(id(a_t), old_id)
        self.assertEqual(
            int(a_t.data), old_ptr, "in-place CUDA should keep same devptr"
        )

        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a * b, rtol=1e-5, atol=1e-6)

    def test_imul_tensor_tensor_cuda_mutates_self_f64(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((3, 7)).astype(np.float64)
        b = rng.standard_normal((3, 7)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        old_ptr = int(a_t.data)
        ret = a_t.__imul__(b_t)

        self.assertIs(ret, a_t)
        self.assertEqual(int(a_t.data), old_ptr)

        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a * b, rtol=1e-12, atol=1e-12)

    def test_imul_tensor_scalar_cuda_mutates_self(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((16,), dtype=np.float32)
        s = 2.5

        a_t = self._make_cuda_tensor_from_host(a)

        old_ptr = int(a_t.data)
        ret = a_t.__imul__(s)

        self.assertIs(ret, a_t)
        self.assertEqual(int(a_t.data), old_ptr)

        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a * s, rtol=1e-5, atol=1e-6)

    # -----------------------
    # Error behavior
    # -----------------------

    def test_imul_raises_on_shape_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((4, 5), dtype=np.float32)
        b = rng.standard_normal((4, 6), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            a_t *= b_t

    def test_imul_raises_on_dtype_mismatch_cuda(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((4, 4), dtype=np.float32)
        b = rng.standard_normal((4, 4)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            a_t *= b_t

    # -----------------------
    # numel == 0 behavior
    # -----------------------

    def test_imul_numel_zero_is_noop_without_alloc(self) -> None:
        """
        numel==0 should be a no-op.

        Your cuda_malloc(0) raises, so we must not allocate.
        We construct an empty CUDA tensor with data=0 and verify __imul__ returns self.
        """
        a0 = np.empty((0,), dtype=np.float32)
        b0 = np.empty((0,), dtype=np.float32)

        a_t = Tensor._from_devptr(
            0,
            shape=tuple(a0.shape),
            dtype=a0.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )
        b_t = Tensor._from_devptr(
            0,
            shape=tuple(b0.shape),
            dtype=b0.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )

        ret = a_t.__imul__(b_t)
        self.assertIs(ret, a_t)
        self.assertEqual(int(a_t.data), 0)

    def test_imul_scalar_numel_zero_is_noop_without_alloc(self) -> None:
        a0 = np.empty((0,), dtype=np.float32)

        a_t = Tensor._from_devptr(
            0,
            shape=tuple(a0.shape),
            dtype=a0.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )

        ret = a_t.__imul__(2.0)
        self.assertIs(ret, a_t)
        self.assertEqual(int(a_t.data), 0)

    # -----------------------
    # Best-effort: fallback path correctness when in-place should be disabled
    # -----------------------

    def test_imul_fallback_path_when_requires_grad_true(self) -> None:
        """
        If requires_grad=True, your __imul__ should avoid true in-place mutation
        (graph safety) and fall back to out-of-place + copy_from.

        We can't reliably assert devptr changes (copy_from may or may not reallocate
        depending on your implementation), but we *can* assert numerical correctness.
        """
        rng = np.random.default_rng(6)
        a = rng.standard_normal((8,), dtype=np.float32)
        b = rng.standard_normal((8,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        a_t.requires_grad = True  # disable in-place fast path

        a_t *= b_t
        out = self._read_cuda_tensor_to_host(a_t)
        np.testing.assert_allclose(out, a * b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
