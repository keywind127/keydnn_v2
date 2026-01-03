# tests/infrastructure/ops/test_tensor_comparison_cuda_ops_ext.py
"""
Unit tests for CUDA Tensor-boundary comparison wrapper (tensor_comparison_cuda_ext.py).

These tests validate that the Tensor-facing wrapper:
- Accepts CUDA Tensors (device-pointer backed)
- Allocates output device memory for out-of-place ops
- Calls the underlying ctypes elementwise comparison kernels
- Returns CUDA Tensors with correct shape/dtype
- Produces numerically correct results vs NumPy reference
- Raises appropriate errors for invalid inputs (dtype/shape/device)

Coverage
--------
Elementwise (tensor-tensor):
- gt, ge, lt, le, eq, ne  -> float32 mask output

Scalar (tensor-scalar):
- gt_scalar, ge_scalar, lt_scalar, le_scalar, eq_scalar, ne_scalar -> float32 mask output

Edge cases:
- numel==0: returns empty CUDA tensor without cuda_malloc(0)
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

from src.keydnn.infrastructure.ops.tensor_comparison_cuda_ext import (
    # elementwise
    gt,
    ge,
    lt,
    le,
    eq,
    ne,
    # scalar
    gt_scalar,
    ge_scalar,
    lt_scalar,
    le_scalar,
    eq_scalar,
    ne_scalar,
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


class TestTensorComparisonCudaExt(unittest.TestCase):
    """Tests for Tensor-boundary CUDA comparison wrapper."""

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

    # ----------------------------
    # Helpers
    # ----------------------------
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

    def _make_empty_cuda_tensor(
        self, shape: Tuple[int, ...], dtype: np.dtype
    ) -> Tensor:
        """
        Create a CUDA tensor with numel==0 without calling cuda_malloc(0).
        Uses dev_ptr=0 (null). Wrapper must early-return / avoid allocations.
        """
        return Tensor._from_devptr(
            0,
            shape=tuple(int(d) for d in shape),
            dtype=np.dtype(dtype),
            device=self.cuda_device,
            requires_grad=False,
        )

    def _assert_mask(
        self, y_t: Tensor, ref: np.ndarray, shape: Tuple[int, ...]
    ) -> None:
        """Assert Tensor mask output is CUDA float32 with correct values."""
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        self.assertEqual(y.dtype, np.float32)
        np.testing.assert_allclose(y, ref.astype(np.float32), rtol=0.0, atol=0.0)

    # ----------------------------
    # Elementwise comparisons (tensor-tensor)
    # ----------------------------
    def test_all_elementwise_ops_f32_match_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64,), dtype=np.float32)
        b = rng.standard_normal((64,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        self._assert_mask(gt(a_t, b_t, device=self.device_index), (a > b), a.shape)
        self._assert_mask(ge(a_t, b_t, device=self.device_index), (a >= b), a.shape)
        self._assert_mask(lt(a_t, b_t, device=self.device_index), (a < b), a.shape)
        self._assert_mask(le(a_t, b_t, device=self.device_index), (a <= b), a.shape)
        self._assert_mask(eq(a_t, b_t, device=self.device_index), (a == b), a.shape)
        self._assert_mask(ne(a_t, b_t, device=self.device_index), (a != b), a.shape)

    def test_all_elementwise_ops_f64_match_numpy(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((7, 11)).astype(np.float64)
        b = rng.standard_normal((7, 11)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        self._assert_mask(gt(a_t, b_t, device=self.device_index), (a > b), a.shape)
        self._assert_mask(ge(a_t, b_t, device=self.device_index), (a >= b), a.shape)
        self._assert_mask(lt(a_t, b_t, device=self.device_index), (a < b), a.shape)
        self._assert_mask(le(a_t, b_t, device=self.device_index), (a <= b), a.shape)
        self._assert_mask(eq(a_t, b_t, device=self.device_index), (a == b), a.shape)
        self._assert_mask(ne(a_t, b_t, device=self.device_index), (a != b), a.shape)

    def test_eq_ne_exact_values_f32(self) -> None:
        a = np.array([0.0, 1.0, 2.0, 2.0, -0.0], dtype=np.float32)
        b = np.array([0.0, 2.0, 2.0, 3.0, 0.0], dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        self._assert_mask(eq(a_t, b_t, device=self.device_index), (a == b), a.shape)
        self._assert_mask(ne(a_t, b_t, device=self.device_index), (a != b), a.shape)

    # ----------------------------
    # Scalar comparisons (tensor-scalar)
    # ----------------------------
    def test_all_scalar_ops_f32_match_numpy(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((128,), dtype=np.float32)
        alpha = 0.25

        a_t = self._make_cuda_tensor_from_host(a)

        self._assert_mask(
            gt_scalar(a_t, alpha, device=self.device_index), (a > alpha), a.shape
        )
        self._assert_mask(
            ge_scalar(a_t, alpha, device=self.device_index), (a >= alpha), a.shape
        )
        self._assert_mask(
            lt_scalar(a_t, alpha, device=self.device_index), (a < alpha), a.shape
        )
        self._assert_mask(
            le_scalar(a_t, alpha, device=self.device_index), (a <= alpha), a.shape
        )
        self._assert_mask(
            eq_scalar(a_t, alpha, device=self.device_index), (a == alpha), a.shape
        )
        self._assert_mask(
            ne_scalar(a_t, alpha, device=self.device_index), (a != alpha), a.shape
        )

    def test_all_scalar_ops_f64_match_numpy(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((9, 10)).astype(np.float64)
        alpha = -1.0

        a_t = self._make_cuda_tensor_from_host(a)

        self._assert_mask(
            gt_scalar(a_t, alpha, device=self.device_index), (a > alpha), a.shape
        )
        self._assert_mask(
            ge_scalar(a_t, alpha, device=self.device_index), (a >= alpha), a.shape
        )
        self._assert_mask(
            lt_scalar(a_t, alpha, device=self.device_index), (a < alpha), a.shape
        )
        self._assert_mask(
            le_scalar(a_t, alpha, device=self.device_index), (a <= alpha), a.shape
        )
        self._assert_mask(
            eq_scalar(a_t, alpha, device=self.device_index), (a == alpha), a.shape
        )
        self._assert_mask(
            ne_scalar(a_t, alpha, device=self.device_index), (a != alpha), a.shape
        )

    def test_scalar_ops_reject_int_dtype(self) -> None:
        a = np.arange(16, dtype=np.int32).reshape(4, 4)
        a_t = self._make_cuda_tensor_from_host(a)

        with self.assertRaises(TypeError):
            _ = gt_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = ge_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = lt_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = le_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = eq_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = ne_scalar(a_t, 1.0, device=self.device_index)

    # ----------------------------
    # Error handling: dtype/shape/device
    # ----------------------------
    def test_binary_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((16,), dtype=np.float32)
        b = rng.standard_normal((16,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = gt(a_t, b_t, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = eq(a_t, b_t, device=self.device_index)

    def test_binary_raises_on_shape_mismatch(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((8, 8), dtype=np.float32)
        b = rng.standard_normal((8, 9), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = lt(a_t, b_t, device=self.device_index)

    def test_binary_raises_on_cpu_tensor(self) -> None:
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
                    f"Unable to construct CPU tensors; update test_binary_raises_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = gt(a_cpu, b_cpu, device=self.device_index)

    def test_scalar_ops_raise_on_cpu_tensor(self) -> None:
        a_np = np.ones((2, 3), dtype=np.float32)

        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            a_cpu = Tensor.from_numpy(a_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                a_cpu = Tensor(a_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                a_cpu._data = a_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensors; update test_scalar_ops_raise_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = ge_scalar(a_cpu, 1.0, device=self.device_index)

    # ----------------------------
    # numel==0 behavior
    # ----------------------------
    def test_numel_zero_tensor_tensor_returns_empty_cuda_tensor(self) -> None:
        a_t = self._make_empty_cuda_tensor((0,), np.float32)
        b_t = self._make_empty_cuda_tensor((0,), np.float32)

        y_t = gt(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (0,))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)
        self.assertEqual(int(y_t.data), 0)

        y2_t = eq(a_t, b_t, device=self.device_index)
        self.assertEqual(int(y2_t.data), 0)

    def test_numel_zero_scalar_returns_empty_cuda_tensor(self) -> None:
        a_t = self._make_empty_cuda_tensor((0, 3), np.float64)
        y_t = lt_scalar(a_t, 0.0, device=self.device_index)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), (0, 3))
        self.assertEqual(np.dtype(y_t.dtype), np.float32)
        self.assertEqual(int(y_t.data), 0)


if __name__ == "__main__":
    unittest.main()
