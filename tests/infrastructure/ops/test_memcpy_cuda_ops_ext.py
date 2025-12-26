# tests/infrastructure/ops/test_memcpy_cuda_ext.py
"""
Unit tests for Tensor-boundary CUDA memcpy helpers (memcpy_cuda_ext.py).

We validate:
- copy_host_to_cuda: host ndarray -> CUDA Tensor
- copy_cuda_to_host: CUDA Tensor -> host ndarray
- copy_cuda_to_cuda: CUDA Tensor -> CUDA Tensor
and basic error handling.

These tests skip on CPU-only environments where the CUDA DLL is unavailable.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.ops.pool2d_cuda import _load_cuda_lib, cuda_set_device
from src.keydnn.infrastructure.ops.memcpy_cuda_ext import (
    copy_host_to_cuda,
    copy_cuda_to_host,
    copy_cuda_to_cuda,
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


class TestMemcpyCudaExt(unittest.TestCase):
    """Tests for Tensor-boundary CUDA memcpy helpers."""

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
            cls.cuda_device = _get_cuda_device(cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to construct CUDA Device: {e}") from e

    def test_copy_host_to_cuda_and_back_f32(self) -> None:
        """H2D then D2H roundtrip preserves values for float32."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((32, 17), dtype=np.float32)

        x_cuda = copy_host_to_cuda(
            x, device_tensor=self.cuda_device, device=self.device_index, sync=True
        )
        self.assertTrue(x_cuda.device.is_cuda())
        self.assertEqual(tuple(x_cuda.shape), x.shape)
        self.assertEqual(np.dtype(x_cuda.dtype), np.float32)

        y = copy_cuda_to_host(x_cuda, device=self.device_index, sync=True)
        np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)

    def test_copy_host_to_cuda_and_back_f64(self) -> None:
        """H2D then D2H roundtrip preserves values for float64."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((7, 5)).astype(np.float64)

        x_cuda = copy_host_to_cuda(
            x, device_tensor=self.cuda_device, device=self.device_index, sync=True
        )
        y = copy_cuda_to_host(x_cuda, device=self.device_index, sync=True)
        np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)

    def test_copy_cuda_to_host_into_preallocated_out(self) -> None:
        """D2H supports a provided output array with correct shape/dtype."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((10, 11), dtype=np.float32)

        x_cuda = copy_host_to_cuda(
            x, device_tensor=self.cuda_device, device=self.device_index, sync=True
        )

        out = np.empty_like(x)
        y = copy_cuda_to_host(x_cuda, out=out, device=self.device_index, sync=True)

        # Should return the same object
        self.assertIs(y, out)
        np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)

    def test_copy_cuda_to_cuda_produces_distinct_buffer(self) -> None:
        """D2D copy returns a new CUDA tensor with identical content."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal((13, 9), dtype=np.float32)

        x_cuda = copy_host_to_cuda(
            x, device_tensor=self.cuda_device, device=self.device_index, sync=True
        )
        y_cuda = copy_cuda_to_cuda(x_cuda, device=self.device_index, sync=True)

        self.assertTrue(y_cuda.device.is_cuda())
        self.assertEqual(tuple(y_cuda.shape), x.shape)
        self.assertEqual(np.dtype(y_cuda.dtype), np.float32)

        # Different device pointers (unless empty)
        if int(x_cuda.data) != 0 and int(y_cuda.data) != 0:
            self.assertNotEqual(int(x_cuda.data), int(y_cuda.data))

        y = copy_cuda_to_host(y_cuda, device=self.device_index, sync=True)
        np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)

    def test_copy_cuda_to_host_rejects_non_contiguous_out(self) -> None:
        """Non-contiguous out array should raise ValueError."""
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        x_cuda = copy_host_to_cuda(
            x, device_tensor=self.cuda_device, device=self.device_index, sync=True
        )

        out = np.empty((4, 3), dtype=np.float32).T  # makes it non-contiguous view (3,4)
        self.assertFalse(out.flags["C_CONTIGUOUS"])

        with self.assertRaises(ValueError):
            _ = copy_cuda_to_host(x_cuda, out=out, device=self.device_index, sync=True)

    def test_copy_cuda_to_host_rejects_cpu_tensor(self) -> None:
        """Passing a CPU tensor to copy_cuda_to_host should raise TypeError."""
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
                    f"Unable to construct CPU tensor in this repo; update test_copy_cuda_to_host_rejects_cpu_tensor: {e}"
                ) from e

        with self.assertRaises(TypeError):
            _ = copy_cuda_to_host(x_cpu, device=self.device_index, sync=True)

    def test_copy_host_to_cuda_rejects_non_ndarray(self) -> None:
        """copy_host_to_cuda should reject non-ndarray input."""
        with self.assertRaises(TypeError):
            _ = copy_host_to_cuda([1, 2, 3], device_tensor=self.cuda_device, device=self.device_index, sync=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
