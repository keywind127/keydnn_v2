from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._models import Sequential
from src.keydnn.infrastructure._linear import Linear
from src.keydnn.infrastructure.flatten._flatten_module import (
    Flatten,
)


class TestModelSaveLoadFlattenJSON(unittest.TestCase):
    def test_save_load_json_flatten_arch_and_forward_match(self) -> None:
        """
        Ensure a model containing Flatten can be saved to a single JSON checkpoint
        and loaded back with identical forward outputs.

        Important
        ---------
        Do NOT manually construct a Device in tests. Let modules pick defaults,
        then build input tensors on the module's device when needed.
        """
        np.random.seed(0)

        # Model: (N, 2, 3) -> Flatten -> (N, 6) -> Linear -> (N, 4)
        model = Sequential(
            Flatten(),
            Linear(in_features=6, out_features=4, bias=True),
        )

        # Deterministic weights for Linear
        lin: Linear = model[1]  # type: ignore[assignment]
        W = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 10.0
        b = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        # Input tensor (on same device as linear)
        x_np = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) / 5.0
        x = Tensor(shape=x_np.shape, device=lin.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "flatten_linear.json"
            model.save_json(ckpt_path)

            self.assertTrue(ckpt_path.exists())
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        # Ensure Flatten is present and ordering preserved
        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], Flatten)
        self.assertIsInstance(loaded[1], Linear)

        loaded_lin: Linear = loaded[1]  # type: ignore[assignment]
        np.testing.assert_allclose(loaded_lin.weight.to_numpy(), W, rtol=0.0, atol=0.0)
        self.assertIsNotNone(loaded_lin.bias)
        np.testing.assert_allclose(
            loaded_lin.bias.to_numpy(),  # type: ignore[union-attr]
            b,
            rtol=0.0,
            atol=0.0,
        )

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_flatten_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if checkpoint state is missing a parameter key,
        while the architecture includes Flatten.

        We delete one Linear parameter entry from the saved JSON state.
        """
        model = Sequential(
            Flatten(),
            Linear(in_features=6, out_features=4, bias=True),
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_flatten.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            # Remove one parameter entry (prefer a Linear key if present)
            lin_keys = [
                k for k in state.keys() if k.endswith(".weight") or k.endswith(".bias")
            ]
            key_to_remove = lin_keys[0] if lin_keys else next(iter(state.keys()))
            del state[key_to_remove]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)


if __name__ == "__main__":
    unittest.main()
