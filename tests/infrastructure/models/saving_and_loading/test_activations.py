from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._models import Sequential
from src.keydnn.infrastructure._activations import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
)


class TestModelSaveLoadActivationsJSON(unittest.TestCase):
    def test_save_load_json_activation_stack_preserves_types_and_hparams(self) -> None:
        """
        Ensure a Sequential containing only activation modules can be saved and loaded
        with identical architecture (types + hyperparameters).

        Notes
        -----
        - Stateless activations must round-trip with empty config.
        - LeakyReLU must preserve alpha.
        - Softmax must preserve axis.
        """
        model = Sequential(
            Sigmoid(),
            ReLU(),
            LeakyReLU(alpha=0.123),
            Tanh(),
            Softmax(axis=1),
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "activations.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        # Order + types preserved
        self.assertEqual(len(loaded), 5)
        self.assertIsInstance(loaded[0], Sigmoid)
        self.assertIsInstance(loaded[1], ReLU)
        self.assertIsInstance(loaded[2], LeakyReLU)
        self.assertIsInstance(loaded[3], Tanh)
        self.assertIsInstance(loaded[4], Softmax)

        # Hyperparams preserved
        loaded_lrelu: LeakyReLU = loaded[2]  # type: ignore[assignment]
        self.assertAlmostEqual(float(getattr(loaded_lrelu, "alpha")), 0.123, places=7)

        loaded_softmax: Softmax = loaded[4]  # type: ignore[assignment]
        self.assertEqual(int(getattr(loaded_softmax, "_axis")), 1)

    def test_save_load_json_activation_forward_matches(self) -> None:
        """
        Ensure forward outputs match exactly before vs after save/load for a pure
        activation stack.
        """
        np.random.seed(0)

        model = Sequential(
            Sigmoid(),
            ReLU(),
            LeakyReLU(alpha=0.2),
            Tanh(),
            Softmax(axis=1),
        )

        x_np = np.array(
            [
                [1.0, -2.0, 0.5, 3.0],
                [-1.0, 2.0, -0.25, 0.0],
            ],
            dtype=np.float32,
        )

        from src.keydnn.domain.device._device import Device

        x = Tensor(
            shape=x_np.shape,
            device=Device("cpu"),
            requires_grad=False,
        )
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "activations_forward.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_activation_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if the checkpoint state is missing a key.

        Even though these activations are parameter-free, the checkpoint still
        contains a 'state' dict. We delete a required top-level key to force a
        clear failure.
        """
        model = Sequential(Sigmoid(), ReLU())

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_activations.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))

            # Remove required key to force failure path
            del payload["arch"]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)


if __name__ == "__main__":
    unittest.main()
