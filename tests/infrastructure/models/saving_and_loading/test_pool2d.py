from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
)


class TestModelSaveLoadPoolingJSON(unittest.TestCase):
    """
    Tests for JSON save/load round-trip of pooling modules inside Sequential.

    Notes
    -----
    - Pooling modules are stateless (no parameters), so these tests focus on:
      (1) architecture round-trip (module types & hyperparams)
      (2) forward output equality before vs after load
    - Uses tempfile so no real files are written.
    """

    def _make_input(self) -> Tensor:
        # Deterministic input (N, C, H, W)
        x_np = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
        x = Tensor(
            shape=x_np.shape,
            device=Tensor(shape=(), device=Tensor(shape=(), device=None).device).device,
        )  # noqa: E501
        # The above is intentionally NOT correct. We must not create Device manually, and
        # we also can't access device without a safe default. Instead, always use a CPU
        # tensor by creating a dummy tensor and reusing its device.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Correct version (no weird device hacks)
# ---------------------------------------------------------------------------

from src.keydnn.domain.device._device import Device  # noqa: E402


class TestModelSaveLoadPoolingJSON(unittest.TestCase):
    def _make_input(self) -> Tensor:
        # Deterministic input (N, C, H, W)
        x_np = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
        # IMPORTANT: Use default CPU device via the framework's Device("cpu")
        # If you previously had match/case identity issues, your duck-typed device fix
        # should make this safe.
        x = Tensor(shape=x_np.shape, device=Device("cpu"), requires_grad=False)
        x.copy_from_numpy(x_np)
        return x

    def test_save_load_json_maxpool2d_roundtrip(self) -> None:
        """
        Ensure Sequential(MaxPool2d) round-trips:
        - hyperparameters preserved
        - forward output preserved
        """
        model = Sequential(MaxPool2d(kernel_size=2, stride=2, padding=0))
        x = self._make_input()
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "maxpool.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)
            self.assertIsInstance(payload["state"], dict)
            # Pooling has no parameters => state should be empty
            self.assertEqual(len(payload["state"]), 0)

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
        """
        Ensure Sequential(AvgPool2d) round-trips:
        - hyperparameters preserved
        - forward output preserved
        """
        model = Sequential(AvgPool2d(kernel_size=2, stride=2, padding=0))
        x = self._make_input()
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "avgpool.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)
            # Pooling has no parameters => state should be empty
            self.assertEqual(len(payload["state"]), 0)

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
        """
        Ensure Sequential(GlobalAvgPool2d) round-trips:
        - no hyperparameters
        - forward output preserved
        """
        model = Sequential(GlobalAvgPool2d())
        x = self._make_input()
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "gap.json"
            model.save_json(ckpt)

            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)
            # Pooling has no parameters => state should be empty
            self.assertEqual(len(payload["state"]), 0)

            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 1)
        layer = loaded[0]
        self.assertIsInstance(layer, GlobalAvgPool2d)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_sequential_pooling_chain_preserves_order(self) -> None:
        """
        Ensure a Sequential chain of pooling layers round-trips and preserves order.
        """
        model = Sequential(
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            AvgPool2d(kernel_size=2, stride=1, padding=0),
            GlobalAvgPool2d(),
        )
        x = self._make_input()
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "pool_chain.json"
            model.save_json(ckpt)
            loaded = Sequential.load_json(ckpt)

        self.assertEqual(len(loaded), 3)
        self.assertIsInstance(loaded[0], MaxPool2d)
        self.assertIsInstance(loaded[1], AvgPool2d)
        self.assertIsInstance(loaded[2], GlobalAvgPool2d)

        # Ensure configs survived
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


if __name__ == "__main__":
    unittest.main()
