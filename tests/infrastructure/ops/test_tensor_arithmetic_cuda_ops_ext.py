"""
Unit tests for CUDA Tensor-boundary arithmetic wrapper (tensor_arithmetic_cuda_ext.py).

These tests validate that the Tensor-facing wrapper:
- Accepts CUDA Tensors (device-pointer backed)
- Allocates output device memory
- Calls the underlying ctypes elementwise kernels
- Returns CUDA Tensors with correct shape/dtype
- Produces numerically correct results vs NumPy reference
- Raises appropriate errors for invalid inputs (dtype/shape/device)

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

from src.keydnn.infrastructure.ops.tensor_arithmetic_cuda_ext import (
    neg,
    add,
    sub,
    div,
    gt,
    # NEW: scalar ops
    add_scalar,
    sub_scalar,
    div_scalar,
)


def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.

    This project may expose Device construction in different ways (e.g. Device("cuda", 0),
    Device.cuda(0), Device.from_str("cuda:0"), etc.). This helper tries common patterns.
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

    Returns
    -------
    (h2d, d2h) : ctypes functions
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


class TestTensorArithmeticCudaExt(unittest.TestCase):
    """
    Tests for Tensor-boundary CUDA arithmetic wrapper.

    The wrapper under test:
        keydnn.infrastructure.ops.tensor_arithmetic_cuda_ext
    """

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

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        """Copy CUDA Tensor data back to host NumPy."""
        host = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
        _d2h(self.lib, int(t.data), host)
        return host

    # ----------------------------
    # Correctness: unary
    # ----------------------------

    def test_neg_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((64,), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = neg(x_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, -x, rtol=1e-4, atol=1e-4)

    def test_neg_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((17, 13)).astype(np.float64)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = neg(x_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), x.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, -x, rtol=1e-10, atol=1e-10)

    # ----------------------------
    # Correctness: binary (tensor-tensor)
    # ----------------------------

    def test_add_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((8, 9), dtype=np.float32)
        b = rng.standard_normal((8, 9), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = add(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a + b, rtol=1e-4, atol=1e-4)

    def test_sub_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((11,)).astype(np.float64)
        b = rng.standard_normal((11,)).astype(np.float64)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = sub(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a - b, rtol=1e-10, atol=1e-10)

    def test_div_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((5, 7), dtype=np.float32)
        # avoid zeros to keep relative error stable
        b = rng.standard_normal((5, 7), dtype=np.float32)
        b = np.where(np.abs(b) < 1e-2, np.float32(0.5), b)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = div(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a / b, rtol=1e-3, atol=1e-3)

    # ----------------------------
    # Correctness: scalar (tensor-scalar)
    # ----------------------------

    def test_add_scalar_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(10)
        a = rng.standard_normal((33,), dtype=np.float32)
        alpha = np.float32(1.25)

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = add_scalar(a_t, float(alpha), device=self.device_index)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a + alpha, rtol=1e-4, atol=1e-4)

    def test_sub_scalar_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.standard_normal((6, 7)).astype(np.float64)
        alpha = 0.75  # python float OK

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = sub_scalar(a_t, float(alpha), device=self.device_index)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float64)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a - alpha, rtol=1e-10, atol=1e-10)

    def test_div_scalar_f32_matches_numpy(self) -> None:
        rng = np.random.default_rng(12)
        a = rng.standard_normal((4, 5), dtype=np.float32)
        alpha = np.float32(0.5)  # avoid tiny/zero

        a_t = self._make_cuda_tensor_from_host(a)
        y_t = div_scalar(a_t, float(alpha), device=self.device_index)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        np.testing.assert_allclose(y, a / alpha, rtol=1e-3, atol=1e-3)

    def test_scalar_ops_reject_int_dtype(self) -> None:
        a = np.arange(16, dtype=np.int32).reshape(4, 4)
        a_t = self._make_cuda_tensor_from_host(a)

        with self.assertRaises(TypeError):
            _ = add_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = sub_scalar(a_t, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = div_scalar(a_t, 2.0, device=self.device_index)

    # ----------------------------
    # Correctness: compare
    # ----------------------------

    def test_gt_f32_outputs_float32_and_matches_numpy(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((32,), dtype=np.float32)
        b = rng.standard_normal((32,), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = gt(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = (a > b).astype(np.float32)
        np.testing.assert_allclose(y, ref, rtol=0.0, atol=0.0)

    def test_gt_f64_outputs_float32_and_matches_numpy(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.standard_normal((6, 5)).astype(np.float64)
        b = rng.standard_normal((6, 5)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        y_t = gt(a_t, b_t, device=self.device_index)
        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), a.shape)
        self.assertEqual(np.dtype(y_t.dtype), np.float32)

        y = self._read_cuda_tensor_to_host(y_t)
        ref = (a > b).astype(np.float32)
        np.testing.assert_allclose(y, ref, rtol=0.0, atol=0.0)

    # ----------------------------
    # Error handling
    # ----------------------------

    def test_add_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((4, 4), dtype=np.float32)
        b = rng.standard_normal((4, 4)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = add(a_t, b_t, device=self.device_index)

    def test_add_raises_on_shape_mismatch(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((4, 4), dtype=np.float32)
        b = rng.standard_normal((4, 5), dtype=np.float32)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = add(a_t, b_t, device=self.device_index)

    def test_neg_raises_on_non_float_dtype(self) -> None:
        # Create an int32 CUDA buffer (still copyable), but wrapper should reject dtype.
        x = np.arange(16, dtype=np.int32)
        x_t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(TypeError):
            _ = neg(x_t, device=self.device_index)

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
                    f"Unable to construct CPU tensors in this repo; update test_scalar_ops_raise_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = add_scalar(a_cpu, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = sub_scalar(a_cpu, 1.0, device=self.device_index)
        with self.assertRaises(TypeError):
            _ = div_scalar(a_cpu, 2.0, device=self.device_index)

    def test_raises_on_cpu_tensor(self) -> None:
        # Construct CPU tensors in a best-effort way.
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
                    f"Unable to construct CPU tensors in this repo; update test_raises_on_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = add(a_cpu, b_cpu, device=self.device_index)

    def test_gt_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.standard_normal((10,), dtype=np.float32)
        b = rng.standard_normal((10,)).astype(np.float64)

        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(TypeError):
            _ = gt(a_t, b_t, device=self.device_index)


if __name__ == "__main__":
    unittest.main()
