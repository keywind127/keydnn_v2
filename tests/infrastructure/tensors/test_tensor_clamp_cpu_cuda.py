# tests/infrastructure/tensors/test_tensor_clamp_cpu_cuda.py
import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


def _make_tensor(arr: np.ndarray, *, device: Device, requires_grad: bool) -> Tensor:
    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device, requires_grad=requires_grad)
    t.copy_from_numpy(a)
    return t


def _cuda_available() -> bool:
    """
    Best-effort CUDA availability check.

    Try to construct a CUDA tensor and round-trip it to NumPy.
    If that fails, skip CUDA tests.
    """
    try:
        dev = Device("cuda:0")
        x = _make_tensor(
            np.array([1.0], dtype=np.float32), device=dev, requires_grad=False
        )
        _ = x.to_numpy()
        return True
    except Exception:
        return False


class TestTensorClampCPU(TestCase):
    def test_clamp_forward_cpu(self) -> None:
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        x = _make_tensor(x_np, device=Device("cpu"), requires_grad=False)

        # clamp is keyword-only in your implementation
        y = x.clamp(min=0.0, max=1.0)
        y_np = np.asarray(y.to_numpy(), dtype=np.float32)

        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(y_np, expected, rtol=0.0, atol=0.0)

    def test_clamp_backward_cpu(self) -> None:
        # Avoid exact boundary values for gradient ambiguity.
        x_np = np.array([-0.5, 0.5, 1.5], dtype=np.float32)  # outside, inside, outside
        x = _make_tensor(x_np, device=Device("cpu"), requires_grad=True)

        y = x.clamp(min=0.0, max=1.0)
        loss = y.sum()
        loss.backward()

        g = x.grad
        self.assertIsNotNone(g)
        g_np = np.asarray(g.to_numpy(), dtype=np.float32).reshape(-1)

        # Expected gradient: 0 outside [min,max], 1 inside.
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(g_np, expected, rtol=0.0, atol=0.0)


@unittest.skipUnless(_cuda_available(), "CUDA not available for KeyDNN tests")
class TestTensorClampCUDA(TestCase):
    def test_clamp_forward_cuda(self) -> None:
        dev = Device("cuda:0")
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        x = _make_tensor(x_np, device=dev, requires_grad=False)

        y = x.clamp(min=0.0, max=1.0)
        y_np = np.asarray(y.to_numpy(), dtype=np.float32)

        expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(y_np, expected, rtol=0.0, atol=0.0)

    def test_clamp_backward_cuda(self) -> None:
        dev = Device("cuda:0")
        x_np = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
        x = _make_tensor(x_np, device=dev, requires_grad=True)

        y = x.clamp(min=0.0, max=1.0)
        loss = y.sum()
        loss.backward()

        g = x.grad
        self.assertIsNotNone(g)
        g_np = np.asarray(g.to_numpy(), dtype=np.float32).reshape(-1)

        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(g_np, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
