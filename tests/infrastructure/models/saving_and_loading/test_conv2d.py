from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import json

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d


class TestModelSaveLoadConv2dJSON(unittest.TestCase):
    def _make_input(self, device, x_np: np.ndarray) -> Tensor:
        x = Tensor(shape=x_np.shape, device=device, requires_grad=False)
        x.copy_from_numpy(x_np.astype(np.float32, copy=False))
        return x

    def test_save_load_json_conv2d_weights_and_forward_match_bias_true(self) -> None:
        np.random.seed(0)

        model = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        conv: Conv2d = model[0]  # type: ignore[assignment]

        W = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3) / 10.0
        b = np.array([0.25, -0.75], dtype=np.float32)

        conv.weight.copy_from_numpy(W)
        self.assertIsNotNone(conv.bias)
        conv.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        x_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
        x = self._make_input(conv.weight.device, x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "conv2d_bias_true.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        loaded_conv: Conv2d = loaded[0]  # type: ignore[assignment]
        np.testing.assert_allclose(loaded_conv.weight.to_numpy(), W, rtol=0.0, atol=0.0)
        self.assertIsNotNone(loaded_conv.bias)
        np.testing.assert_allclose(
            loaded_conv.bias.to_numpy(),  # type: ignore[union-attr]
            b,
            rtol=0.0,
            atol=0.0,
        )

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_conv2d_weights_and_forward_match_bias_false(self) -> None:
        np.random.seed(0)

        model = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        conv: Conv2d = model[0]  # type: ignore[assignment]

        W = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3) / 7.0
        conv.weight.copy_from_numpy(W)
        self.assertIsNone(conv.bias)

        x_np = np.random.randn(2, 1, 5, 5).astype(np.float32)
        x = self._make_input(conv.weight.device, x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "conv2d_bias_false.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        loaded_conv: Conv2d = loaded[0]  # type: ignore[assignment]
        self.assertIsNone(loaded_conv.bias)
        np.testing.assert_allclose(loaded_conv.weight.to_numpy(), W, rtol=0.0, atol=0.0)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_sequential_two_conv2d_layers_preserves_order(self) -> None:
        np.random.seed(0)

        model = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        c0: Conv2d = model[0]  # type: ignore[assignment]
        c1: Conv2d = model[1]  # type: ignore[assignment]

        W0 = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3) / 11.0
        b0 = np.array([0.1, -0.2], dtype=np.float32)
        c0.weight.copy_from_numpy(W0)
        self.assertIsNotNone(c0.bias)
        c0.bias.copy_from_numpy(b0)  # type: ignore[union-attr]

        W1 = np.arange(1 * 2 * 1 * 1, dtype=np.float32).reshape(1, 2, 1, 1) / 3.0
        c1.weight.copy_from_numpy(W1)
        self.assertIsNone(c1.bias)

        x_np = np.random.randn(1, 1, 6, 6).astype(np.float32)
        x = self._make_input(c0.weight.device, x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "conv2d_two_layers.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], Conv2d)
        self.assertIsInstance(loaded[1], Conv2d)

        lc0: Conv2d = loaded[0]  # type: ignore[assignment]
        lc1: Conv2d = loaded[1]  # type: ignore[assignment]

        np.testing.assert_allclose(lc0.weight.to_numpy(), W0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(lc1.weight.to_numpy(), W1, rtol=0.0, atol=0.0)

        self.assertIsNotNone(lc0.bias)
        np.testing.assert_allclose(lc0.bias.to_numpy(), b0, rtol=0.0, atol=0.0)  # type: ignore[union-attr]
        self.assertIsNone(lc1.bias)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_conv2d_missing_key_raises(self) -> None:
        model = Sequential(
            Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=True)
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_conv2d.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            some_key = next(iter(state.keys()))
            del state[some_key]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)


if __name__ == "__main__":
    unittest.main()
