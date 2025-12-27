# tests/infrastructure/tensors/test_tensor_fill_cuda.py
"""
Unit tests for Tensor.fill CUDA support.

These tests validate that:
- Tensor.fill(value) still works on CPU tensors (backward compatibility)
- Tensor.fill(value) works on CUDA tensors by invoking the CUDA fill ext path
- CUDA-filled tensors round-trip to host correctly via device->host memcpy
- Appropriate errors are raised for unsupported dtypes / devices (best-effort)

All CUDA tests are skipped if the KeyDNN CUDA native DLL is unavailable.
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


class TestTensorFillCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

    def _make_cuda_tensor_uninitialized(
        self, shape: Tuple[int, ...], dtype: np.dtype
    ) -> Tensor:
        shp = tuple(int(d) for d in shape)
        dt = np.dtype(dtype)
        n = int(np.prod(shp)) if len(shp) > 0 else 1
        nbytes = int(n) * int(dt.itemsize)
        dev_ptr = int(cuda_malloc(self.lib, int(nbytes))) if nbytes != 0 else 0
        return Tensor._from_devptr(
            int(dev_ptr),
            shape=shp,
            dtype=dt,
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
    # CUDA fill tests
    # ----------------------------

    def test_fill_cuda_f32(self) -> None:
        t = self._make_cuda_tensor_uninitialized((64,), np.float32)
        t.fill(2.5)

        out = self._read_cuda_tensor_to_host(t)
        ref = np.full((64,), 2.5, dtype=np.float32)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=0.0)

    def test_fill_cuda_f64(self) -> None:
        t = self._make_cuda_tensor_uninitialized((7, 5), np.float64)
        t.fill(-1.25)

        out = self._read_cuda_tensor_to_host(t)
        ref = np.full((7, 5), -1.25, dtype=np.float64)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=0.0)

    def test_fill_cuda_empty_noop(self) -> None:
        t = self._make_cuda_tensor_uninitialized((0,), np.float32)
        t.fill(123.0)
        out = self._read_cuda_tensor_to_host(t)
        self.assertEqual(out.size, 0)

    def test_fill_cuda_raises_on_non_float_dtype(self) -> None:
        # Even though we can allocate int32 device memory, the CUDA fill path only supports f32/f64.
        t = self._make_cuda_tensor_uninitialized((8,), np.int32)
        with self.assertRaises(TypeError):
            t.fill(1.0)

    # ----------------------------
    # CPU backward compatibility (best-effort)
    # ----------------------------

    def test_fill_cpu_still_works(self) -> None:
        x = np.zeros((3, 4), dtype=np.float32)

        # Best-effort CPU tensor creation; adjust if your repo has a canonical constructor.
        if hasattr(Tensor, "from_numpy") and callable(getattr(Tensor, "from_numpy")):
            t = Tensor.from_numpy(x, device=Device("cpu"))  # type: ignore[call-arg]
        else:
            try:
                t = Tensor(x.shape, device=Device("cpu"), requires_grad=False)  # type: ignore[call-arg]
                t._data = x  # type: ignore[attr-defined]
            except Exception as e:
                raise unittest.SkipTest(
                    f"Unable to construct CPU Tensor in this repo; update test_fill_cpu_still_works: {e}"
                ) from e

        t.fill(9.0)
        # Read from CPU backing array (most implementations store ndarray in _data)
        data = getattr(t, "_data", None)
        if data is None:
            raise unittest.SkipTest(
                "Unable to access CPU backing array for Tensor; update test."
            )
        np.testing.assert_allclose(data, np.full_like(x, 9.0), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
