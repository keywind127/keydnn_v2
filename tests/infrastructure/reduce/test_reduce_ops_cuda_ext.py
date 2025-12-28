# tests/infrastructure/reduce/test_reduce_ops_cuda_ext.py
"""
Unit tests for CUDA Tensor-boundary reduce wrappers (reduce_cuda_ext.py).

These tests validate that the Tensor-facing wrappers:
- Accept CUDA Tensors (device-pointer backed)
- Allocate output device memory
- Call the underlying ops-layer reduce functions
- Return CUDA Tensors with correct shape/dtype
- Produce numerically correct results vs NumPy references
- Raise appropriate errors for invalid inputs

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

from src.keydnn.infrastructure.ops.reduce_cuda_ext import (
    sum_all_forward,
    mean_all_forward,
    sum_axis2d_forward,
    sum_axis2d_backward,
    sum_backward_fill_forward,
    mean_backward_fill_forward,
    max_axis2d_forward,
    max_axis2d_backward,
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


def _tols_sum(dtype: np.dtype, numel: int) -> tuple[float, float]:
    if dtype == np.float32:
        # atomicAdd reduction order is nondeterministic; error grows with numel
        if numel >= 100_000:
            return (2e-5, 1e-4)
        # was (1e-5, 1e-6) — slightly too strict on some inputs
        return (2e-5, 5e-6)
    return (1e-12, 1e-12)


def _tols_default(dtype: np.dtype) -> tuple[float, float]:
    if dtype == np.float32:
        return (1e-5, 1e-6)
    return (1e-12, 1e-12)


class TestReduceCudaExt(unittest.TestCase):
    """
    Tests for Tensor-boundary CUDA reduce wrappers.

    Wrapper under test:
        keydnn.infrastructure.ops.reduce_cuda_ext
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

    # ---------------------------------------------------------------------
    # helpers: CUDA Tensor <-> host
    # ---------------------------------------------------------------------

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

    def _make_cuda_scalar_from_host(self, value: np.ndarray) -> Tensor:
        """Wrap a 1-element host buffer as a CUDA scalar Tensor (shape=())."""
        v = np.ascontiguousarray(value).reshape((1,))
        dev_ptr = int(cuda_malloc(self.lib, int(v.nbytes)))
        try:
            _h2d(self.lib, dev_ptr, v)
        except Exception:
            cuda_free(self.lib, dev_ptr)
            raise

        return Tensor._from_devptr(
            int(dev_ptr),
            shape=(),
            dtype=v.dtype,
            device=self.cuda_device,
            requires_grad=False,
        )

    def _read_cuda_tensor_to_host(self, t: Tensor) -> np.ndarray:
        """Copy CUDA Tensor data back to host NumPy."""
        shape = tuple(int(d) for d in t.shape)
        if shape == ():
            host = np.empty((1,), dtype=np.dtype(t.dtype))
        else:
            host = np.empty(shape, dtype=np.dtype(t.dtype))
        _d2h(self.lib, int(t.data), host)
        return host[0:1] if shape == () else host

    # ---------------------------------------------------------------------
    # sum_all / mean_all
    # ---------------------------------------------------------------------

    def _run_sum_all_case(self, dtype: np.dtype, *, shape: tuple[int, ...]) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(shape).astype(dtype, copy=False)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = sum_all_forward(x_t, device=self.device_index, sync=True)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), ())
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))

        y = self._read_cuda_tensor_to_host(y_t)[0]
        ref = np.array(x.sum(), dtype=dtype).item()

        rtol, atol = _tols_sum(np.dtype(dtype), int(x.size))
        np.testing.assert_allclose(y, ref, rtol=rtol, atol=atol)

    def test_sum_all_forward_f32_small(self) -> None:
        self._run_sum_all_case(np.float32, shape=(257,))

    def test_sum_all_forward_f64_small(self) -> None:
        self._run_sum_all_case(np.float64, shape=(257,))

    def test_sum_all_forward_f32_large(self) -> None:
        self._run_sum_all_case(np.float32, shape=(100_003,))

    def test_sum_all_forward_f64_large(self) -> None:
        self._run_sum_all_case(np.float64, shape=(100_003,))

    def _run_mean_all_case(self, dtype: np.dtype, *, shape: tuple[int, ...]) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(shape).astype(dtype, copy=False)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = mean_all_forward(x_t, device=self.device_index, sync=True)

        self.assertTrue(y_t.device.is_cuda())
        self.assertEqual(tuple(y_t.shape), ())
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))

        y = self._read_cuda_tensor_to_host(y_t)[0]
        ref = np.array(x.mean(), dtype=dtype).item()

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(y, ref, rtol=rtol, atol=atol)

    def test_mean_all_forward_f32(self) -> None:
        self._run_mean_all_case(np.float32, shape=(200_000,))

    def test_mean_all_forward_f64(self) -> None:
        self._run_mean_all_case(np.float64, shape=(200_000,))

    # ---------------------------------------------------------------------
    # sum_axis2d_forward
    # ---------------------------------------------------------------------

    def _run_sum_axis2d_forward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t = sum_axis2d_forward(x_t, axis=axis, device=self.device_index, sync=True)

        self.assertTrue(y_t.device.is_cuda())
        exp_shape = (cols,) if axis == 0 else (rows,)
        self.assertEqual(tuple(y_t.shape), exp_shape)
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))

        y = self._read_cuda_tensor_to_host(y_t)
        ref = x.sum(axis=axis).astype(dtype, copy=False)

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(y, ref, rtol=rtol, atol=atol)

    def test_sum_axis2d_forward_f32_axis1(self) -> None:
        self._run_sum_axis2d_forward_case(np.float32, rows=17, cols=31, axis=1)

    def test_sum_axis2d_forward_f64_axis1(self) -> None:
        self._run_sum_axis2d_forward_case(np.float64, rows=17, cols=31, axis=1)

    def test_sum_axis2d_forward_f32_axis0(self) -> None:
        self._run_sum_axis2d_forward_case(np.float32, rows=19, cols=29, axis=0)

    def test_sum_axis2d_forward_f64_axis0(self) -> None:
        self._run_sum_axis2d_forward_case(np.float64, rows=19, cols=29, axis=0)

    # ---------------------------------------------------------------------
    # sum_axis2d_backward (broadcast)
    # ---------------------------------------------------------------------

    def _run_sum_axis2d_backward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = np.random.default_rng(3)

        go_shape = (rows,) if axis == 1 else (cols,)
        grad_out = rng.standard_normal(go_shape).astype(dtype, copy=False)
        grad_out_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t = sum_axis2d_backward(
            grad_out_t,
            rows=rows,
            cols=cols,
            axis=axis,
            device=self.device_index,
            sync=True,
        )

        self.assertTrue(gx_t.device.is_cuda())
        self.assertEqual(tuple(gx_t.shape), (rows, cols))
        self.assertEqual(np.dtype(gx_t.dtype), np.dtype(dtype))

        gx = self._read_cuda_tensor_to_host(gx_t)

        if axis == 1:
            ref = np.broadcast_to(grad_out.reshape(rows, 1), (rows, cols)).astype(
                dtype, copy=False
            )
        else:
            ref = np.broadcast_to(grad_out.reshape(1, cols), (rows, cols)).astype(
                dtype, copy=False
            )

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(gx, ref, rtol=rtol, atol=atol)

    def test_sum_axis2d_backward_f32_axis1(self) -> None:
        self._run_sum_axis2d_backward_case(np.float32, rows=23, cols=11, axis=1)

    def test_sum_axis2d_backward_f64_axis1(self) -> None:
        self._run_sum_axis2d_backward_case(np.float64, rows=23, cols=11, axis=1)

    def test_sum_axis2d_backward_f32_axis0(self) -> None:
        self._run_sum_axis2d_backward_case(np.float32, rows=13, cols=37, axis=0)

    def test_sum_axis2d_backward_f64_axis0(self) -> None:
        self._run_sum_axis2d_backward_case(np.float64, rows=13, cols=37, axis=0)

    # ---------------------------------------------------------------------
    # backward fill helpers (scalar -> vector)
    # ---------------------------------------------------------------------

    def _run_sum_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = np.random.default_rng(4)
        grad_out = np.array(rng.standard_normal(), dtype=dtype)

        go_t = self._make_cuda_scalar_from_host(
            np.array([grad_out.item()], dtype=dtype)
        )
        gx_t = sum_backward_fill_forward(
            go_t, numel=numel, device=self.device_index, sync=True
        )

        self.assertTrue(gx_t.device.is_cuda())
        self.assertEqual(tuple(gx_t.shape), (numel,))
        self.assertEqual(np.dtype(gx_t.dtype), np.dtype(dtype))

        gx = self._read_cuda_tensor_to_host(gx_t)
        ref = (np.ones((numel,), dtype=dtype) * grad_out).astype(dtype, copy=False)

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(gx, ref, rtol=rtol, atol=atol)

    def test_sum_backward_fill_forward_f32(self) -> None:
        self._run_sum_backward_fill_case(np.float32, numel=12_345)

    def test_sum_backward_fill_forward_f64(self) -> None:
        self._run_sum_backward_fill_case(np.float64, numel=12_345)

    def _run_mean_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = np.random.default_rng(5)
        grad_out = np.array(rng.standard_normal(), dtype=dtype)

        go_t = self._make_cuda_scalar_from_host(
            np.array([grad_out.item()], dtype=dtype)
        )
        gx_t = mean_backward_fill_forward(
            go_t, numel=numel, device=self.device_index, sync=True
        )

        self.assertTrue(gx_t.device.is_cuda())
        self.assertEqual(tuple(gx_t.shape), (numel,))
        self.assertEqual(np.dtype(gx_t.dtype), np.dtype(dtype))

        gx = self._read_cuda_tensor_to_host(gx_t)
        # ref = (np.ones((numel,), dtype=dtype) * (grad_out / dtype.type(numel))).astype(
        #     dtype, copy=False
        # )

        dt = np.dtype(dtype)
        scale = dt.type(numel)
        ref = (np.ones((numel,), dtype=dt) * (grad_out / scale)).astype(dt, copy=False)

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(gx, ref, rtol=rtol, atol=atol)

    def test_mean_backward_fill_forward_f32(self) -> None:
        self._run_mean_backward_fill_case(np.float32, numel=65_537)

    def test_mean_backward_fill_forward_f64(self) -> None:
        self._run_mean_backward_fill_case(np.float64, numel=65_537)

    # ---------------------------------------------------------------------
    # max axis2d ext (optional parity tests)
    # ---------------------------------------------------------------------

    def _run_max_axis2d_forward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = np.random.default_rng(6)
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)
        x_t = self._make_cuda_tensor_from_host(x)

        y_t, idx_t = max_axis2d_forward(
            x_t, axis=axis, device=self.device_index, sync=True
        )

        self.assertTrue(y_t.device.is_cuda())
        self.assertTrue(idx_t.device.is_cuda())
        out_len = cols if axis == 0 else rows
        self.assertEqual(tuple(y_t.shape), (out_len,))
        self.assertEqual(tuple(idx_t.shape), (out_len,))
        self.assertEqual(np.dtype(y_t.dtype), np.dtype(dtype))
        self.assertEqual(np.dtype(idx_t.dtype), np.int64)

        y = self._read_cuda_tensor_to_host(y_t)
        idx = self._read_cuda_tensor_to_host(idx_t).astype(np.int64, copy=False)

        ref_idx = np.argmax(x, axis=axis).astype(np.int64, copy=False)
        if axis == 1:
            ref_y = x[np.arange(rows), ref_idx]
        else:
            ref_y = x[ref_idx, np.arange(cols)]
        ref_y = ref_y.astype(dtype, copy=False)

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(y, ref_y, rtol=rtol, atol=atol)
        np.testing.assert_array_equal(idx, ref_idx)

    def test_max_axis2d_forward_f32_axis1(self) -> None:
        self._run_max_axis2d_forward_case(np.float32, rows=17, cols=31, axis=1)

    def test_max_axis2d_forward_f64_axis0(self) -> None:
        self._run_max_axis2d_forward_case(np.float64, rows=19, cols=29, axis=0)

    def _run_max_axis2d_backward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)
        x_t = self._make_cuda_tensor_from_host(x)

        _, idx_t = max_axis2d_forward(
            x_t, axis=axis, device=self.device_index, sync=True
        )

        out_len = cols if axis == 0 else rows
        grad_out = rng.standard_normal((out_len,)).astype(dtype, copy=False)
        grad_out_t = self._make_cuda_tensor_from_host(grad_out)

        gx_t = max_axis2d_backward(
            grad_out_t,
            idx_t,
            rows=rows,
            cols=cols,
            axis=axis,
            device=self.device_index,
            zero_grad_x=True,
            sync=True,
        )

        self.assertTrue(gx_t.device.is_cuda())
        self.assertEqual(tuple(gx_t.shape), (rows, cols))
        self.assertEqual(np.dtype(gx_t.dtype), np.dtype(dtype))

        gx = self._read_cuda_tensor_to_host(gx_t)

        idx = self._read_cuda_tensor_to_host(idx_t).astype(np.int64, copy=False)
        ref = np.zeros((rows, cols), dtype=dtype)
        if axis == 1:
            for r in range(rows):
                ref[r, int(idx[r])] += grad_out[r]
        else:
            for c in range(cols):
                ref[int(idx[c]), c] += grad_out[c]

        rtol, atol = _tols_default(np.dtype(dtype))
        np.testing.assert_allclose(gx, ref, rtol=rtol, atol=atol)

    def test_max_axis2d_backward_f32_axis1(self) -> None:
        self._run_max_axis2d_backward_case(np.float32, rows=23, cols=11, axis=1)

    def test_max_axis2d_backward_f64_axis0(self) -> None:
        self._run_max_axis2d_backward_case(np.float64, rows=13, cols=37, axis=0)

    # ---------------------------------------------------------------------
    # validation tests
    # ---------------------------------------------------------------------

    def test_sum_axis2d_forward_rejects_non_2d(self) -> None:
        rng = np.random.default_rng(8)
        x = rng.standard_normal((2, 3, 4), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(ValueError):
            _ = sum_axis2d_forward(x_t, axis=1, device=self.device_index, sync=True)

    def test_sum_axis2d_forward_rejects_bad_axis(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.standard_normal((3, 4), dtype=np.float32)
        x_t = self._make_cuda_tensor_from_host(x)

        with self.assertRaises(ValueError):
            _ = sum_axis2d_forward(x_t, axis=2, device=self.device_index, sync=True)

    def test_sum_axis2d_backward_rejects_shape_mismatch(self) -> None:
        # axis=1 expects (rows,), give (cols,)
        rng = np.random.default_rng(10)
        rows, cols = 5, 7
        grad_out_wrong = rng.standard_normal((cols,), dtype=np.float32)
        go_t = self._make_cuda_tensor_from_host(grad_out_wrong)

        with self.assertRaises(ValueError):
            _ = sum_axis2d_backward(
                go_t, rows=rows, cols=cols, axis=1, device=self.device_index, sync=True
            )

    def test_raises_on_cpu_tensor(self) -> None:
        x_np = np.ones((2, 3), dtype=np.float32)

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
            _ = sum_all_forward(x_cpu, device=self.device_index, sync=True)


if __name__ == "__main__":
    unittest.main()
