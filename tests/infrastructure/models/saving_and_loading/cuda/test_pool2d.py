# tests/infrastructure/models/saving_and_loading/cuda/test_pool2d.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure._models import Sequential
from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
)


def _cuda_available() -> bool:
    """
    Best-effort detection: if the native CUDA loader can be imported and invoked,
    we assume CUDA ops are available for the test environment.
    """
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _make_input(*, device: Device) -> Tensor:
    """
    Deterministic input (N, C, H, W) = (1, 1, 4, 4)
    """
    x_np = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    x = Tensor(shape=x_np.shape, device=device, requires_grad=False, dtype=x_np.dtype)

    # CUDA tensors may start with devptr=0; ensure allocation before H2D copy.
    if device.is_cuda():
        x._ensure_cuda_alloc(dtype=np.dtype(x_np.dtype))

    x.copy_from_numpy(x_np)
    return x


class _PoolingRoundtripMixin:
    """
    Mixin that implements pooling JSON round-trip tests for a given device.
    Subclasses must set `DEVICE: Device`.
    """
    DEVICE: Device

    def _assert_state_is_empty(self, payload: dict) -> None:
        self.assertIn("state", payload)
        self.assertIsInstance(payload["state"], dict)
        self.assertEqual(len(payload["state"]), 0)

    def test_save_load_json_maxpool2d_roundtrip(self) -> None:
        model = Sequential(MaxPool2d(kernel_size=2, stride=2, padding=0))
        x = _make_input(device=self.DEVICE)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "maxpool.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self._assert_state_is_empty(payload)

            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 1)
        layer = loaded[0]
        self.assertIsInstance(layer, MaxPool2d)
        self.assertEqual(layer.kernel_size, (2, 2))
        self.assertEqual(layer.stride, (2, 2))
        self.assertEqual(layer.padding, (0, 0))

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_avgpool2d_roundtrip(self) -> None:
        model = Sequential(AvgPool2d(kernel_size=2, stride=2, padding=0))
        x = _make_input(device=self.DEVICE)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "avgpool.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self._assert_state_is_empty(payload)

            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 1)
        layer = loaded[0]
        self.assertIsInstance(layer, AvgPool2d)
        self.assertEqual(layer.kernel_size, (2, 2))
        self.assertEqual(layer.stride, (2, 2))
        self.assertEqual(layer.padding, (0, 0))

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_globalavgpool2d_roundtrip(self) -> None:
        model = Sequential(GlobalAvgPool2d())
        x = _make_input(device=self.DEVICE)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "gap.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self._assert_state_is_empty(payload)

            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 1)
        layer = loaded[0]
        self.assertIsInstance(layer, GlobalAvgPool2d)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_sequential_pooling_chain_preserves_order(self) -> None:
        model = Sequential(
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            AvgPool2d(kernel_size=2, stride=1, padding=0),
            GlobalAvgPool2d(),
        )
        x = _make_input(device=self.DEVICE)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "pool_chain.json"
            model.save_json(ckpt)
            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 3)
        self.assertIsInstance(loaded[0], MaxPool2d)
        self.assertIsInstance(loaded[1], AvgPool2d)
        self.assertIsInstance(loaded[2], GlobalAvgPool2d)

        mp: MaxPool2d = loaded[0]  # type: ignore[assignment]
        ap: AvgPool2d = loaded[1]  # type: ignore[assignment]

        self.assertEqual(mp.kernel_size, (2, 2))
        self.assertEqual(mp.stride, (2, 2))
        self.assertEqual(mp.padding, (0, 0))

        self.assertEqual(ap.kernel_size, (2, 2))
        self.assertEqual(ap.stride, (1, 1))
        self.assertEqual(ap.padding, (0, 0))

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)


class TestModelSaveLoadPoolingJSONCpu(_PoolingRoundtripMixin, unittest.TestCase):
    DEVICE = Device("cpu")


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelSaveLoadPoolingJSONCuda(_PoolingRoundtripMixin, unittest.TestCase):
    DEVICE = Device("cuda:0")


if __name__ == "__main__":
    unittest.main()
