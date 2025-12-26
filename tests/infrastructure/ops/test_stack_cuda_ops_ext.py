"""
Unit tests for CUDA Tensor-boundary stack wrapper (stack_cuda_ext.py).

These tests validate that the Tensor-facing wrapper:
- Accepts CUDA Tensors (device-pointer backed)
- Allocates output device memory
- Calls underlying CUDA stack kernels via ctypes wrapper
- Returns a CUDA Tensor with correct shape/dtype
- Produces numerically correct results vs NumPy reference
- Provides correct backward gradients by unstacking grad_out
- Raises appropriate errors for invalid inputs

The tests skip on CPU-only environments or when the KeyDNN CUDA DLL is unavailable.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple, List

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)

from src.keydnn.infrastructure.ops.stack_cuda_ext import (
    stack_forward,
    stack_backward,
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


class TestStackCudaExt(unittest.TestCase):
    """Tests for Tensor-boundary CUDA stack wrapper."""

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

    # ------------------------------------------------------------------
    # Forward tests
    # ------------------------------------------------------------------

    def test_stack_forward_f32_matches_numpy_multiple_axes(self) -> None:
        rng = np.random.default_rng(0)
        in_shape = (2, 3, 4)
        K = 5
        xs = [rng.standard_normal(in_shape, dtype=np.float32) for _ in range(K)]
        ts = [self._make_cuda_tensor_from_host(x) for x in xs]

        for axis in (0, 1, 2, 3, -1):
            with self.subTest(axis=axis):
                y = stack_forward(ts, axis=axis, device=self.device_index)
                self.assertTrue(y.device.is_cuda(), "output should be CUDA")
                self.assertEqual(np.dtype(y.dtype), np.float32)

                ref = np.stack(xs, axis=axis)
                self.assertEqual(tuple(y.shape), ref.shape)

                got = self._read_cuda_tensor_to_host(y)
                np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_stack_forward_f64_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        in_shape = (2, 2, 3)
        K = 3
        xs = [rng.standard_normal(in_shape).astype(np.float64) for _ in range(K)]
        ts = [self._make_cuda_tensor_from_host(x) for x in xs]

        axis = 1
        y = stack_forward(ts, axis=axis, device=self.device_index)
        self.assertTrue(y.device.is_cuda())
        self.assertEqual(np.dtype(y.dtype), np.float64)

        ref = np.stack(xs, axis=axis)
        self.assertEqual(tuple(y.shape), ref.shape)

        got = self._read_cuda_tensor_to_host(y)
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_stack_forward_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            _ = stack_forward([], axis=0, device=self.device_index)  # type: ignore[arg-type]

    def test_stack_forward_raises_on_shape_mismatch(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((2, 3), dtype=np.float32)
        b = rng.standard_normal((2, 4), dtype=np.float32)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = stack_forward([a_t, b_t], axis=0, device=self.device_index)

    def test_stack_forward_raises_on_dtype_mismatch(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((2, 3), dtype=np.float32)
        b = rng.standard_normal((2, 3)).astype(np.float64)
        a_t = self._make_cuda_tensor_from_host(a)
        b_t = self._make_cuda_tensor_from_host(b)

        with self.assertRaises(ValueError):
            _ = stack_forward([a_t, b_t], axis=0, device=self.device_index)

    def test_stack_forward_raises_on_cpu_tensor(self) -> None:
        # Best-effort CPU tensor construction. Adjust if your repo has Tensor.from_numpy().
        x_np = np.ones((2, 2), dtype=np.float32)

        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            x_cpu = Tensor.from_numpy(x_np, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                x_cpu = Tensor(x_np.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                x_cpu._data = x_np  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU tensor in this repo; update test_stack_forward_raises_on_cpu_tensor: {e}"
                ) from e

        # Mix CPU + CUDA should fail (and even all-CPU should fail because stack_forward expects CUDA)
        cuda_t = self._make_cuda_tensor_from_host(x_np)
        with self.assertRaises(TypeError):
            _ = stack_forward([cuda_t, x_cpu], axis=0, device=self.device_index)

    # ------------------------------------------------------------------
    # Backward tests
    # ------------------------------------------------------------------

    def test_stack_backward_f32_matches_numpy_multiple_axes(self) -> None:
        rng = np.random.default_rng(4)
        x_shape = (2, 3, 4)
        K = 4
        dy_shape_base = x_shape
        # We'll test multiple axes by generating dy each time.
        for axis in (0, 1, 2, 3, -1):
            with self.subTest(axis=axis):
                axis_n = axis
                # Build expected dy shape: x_shape[:axis] + (K,) + x_shape[axis:]
                # Need normalized axis for shape math.
                ndim = len(x_shape)
                if axis_n < 0:
                    axis_n = axis_n + (ndim + 1)
                out_shape = tuple(x_shape[:axis_n]) + (K,) + tuple(x_shape[axis_n:])

                dy = rng.standard_normal(out_shape, dtype=np.float32)
                dy_t = self._make_cuda_tensor_from_host(dy)

                grads = stack_backward(
                    dy_t,
                    x_shape=x_shape,
                    axis=axis,
                    K=K,
                    device=self.device_index,
                )
                self.assertEqual(len(grads), K)
                for g in grads:
                    self.assertTrue(g.device.is_cuda())
                    self.assertEqual(tuple(g.shape), x_shape)
                    self.assertEqual(np.dtype(g.dtype), np.float32)

                # Compare each to np.take
                for i in range(K):
                    got = self._read_cuda_tensor_to_host(grads[i])
                    ref = np.take(dy, i, axis=axis)
                    np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_stack_backward_raises_on_grad_shape_mismatch(self) -> None:
        rng = np.random.default_rng(5)
        x_shape = (2, 3)
        K = 3
        # wrong: missing the inserted K dimension
        dy = rng.standard_normal(x_shape, dtype=np.float32)
        dy_t = self._make_cuda_tensor_from_host(dy)

        with self.assertRaises(ValueError):
            _ = stack_backward(
                dy_t,
                x_shape=x_shape,
                axis=0,
                K=K,
                device=self.device_index,
            )


if __name__ == "__main__":
    unittest.main()
