from pathlib import Path
import tempfile
import unittest
import json

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure.fully_connected._linear import Linear


class TestModelSaveLoadLinearJSON(unittest.TestCase):
    def test_save_load_json_linear_weights_and_forward_match(self) -> None:
        """
        Ensure a model containing a Linear layer can be saved to a single JSON
        checkpoint and loaded back with identical weights and forward outputs.

        Uses tempfile to avoid writing to the real filesystem.

        Important
        ---------
        Do NOT manually construct a Device in tests. Let the framework pick the
        default CPU device to avoid class-identity mismatches in match/case.
        """
        np.random.seed(0)

        # Build a tiny model (top-level container) that includes Linear.
        # Do NOT pass `device=...` here: let Linear create its own CPU device.
        model = Sequential(
            Linear(in_features=3, out_features=2, bias=True),
        )

        # Access first layer
        lin: Linear = model[0]  # type: ignore[assignment]

        # Overwrite weights/bias with deterministic values (avoid init randomness)
        W = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        b = np.array([0.5, -1.5], dtype=np.float32)

        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        # Create deterministic input
        x_np = np.array(
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]],
            dtype=np.float32,
        )

        # Construct input tensor on the SAME device the layer uses
        x = Tensor(shape=x_np.shape, device=lin.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "linear_model.json"
            model.save_json(ckpt_path)

            # Sanity: JSON exists and has the right shape
            self.assertTrue(ckpt_path.exists())
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        loaded_lin: Linear = loaded[0]  # type: ignore[assignment]

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

    def test_save_load_json_linear_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if the checkpoint state is missing a parameter.

        We delete an arbitrary parameter entry from the saved JSON state.
        """
        model = Sequential(Linear(in_features=3, out_features=2, bias=True))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            # Remove one parameter entry to force failure
            some_key = next(iter(state.keys()))
            del state[some_key]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)

    def test_save_load_json_linear_no_bias_roundtrip(self) -> None:
        """
        Ensure Linear(bias=False) round-trips through JSON:
        - architecture restores with bias disabled
        - weights restore correctly
        - forward output matches
        """
        np.random.seed(0)

        model = Sequential(Linear(in_features=3, out_features=2, bias=False))
        lin: Linear = model[0]  # type: ignore[assignment]

        # Deterministic weights
        W = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -4.0]], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNone(lin.bias)

        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = Tensor(shape=x_np.shape, device=lin.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "linear_nobias.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        loaded_lin: Linear = loaded[0]  # type: ignore[assignment]
        self.assertIsNone(loaded_lin.bias)
        np.testing.assert_allclose(loaded_lin.weight.to_numpy(), W, rtol=0.0, atol=0.0)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_sequential_two_layers_preserves_order(self) -> None:
        """
        Ensure a multi-layer Sequential round-trips and preserves layer order.

        This catches bugs where `_modules` is restored but `_layers` is not,
        or where ordering becomes non-deterministic.
        """
        np.random.seed(0)

        model = Sequential(
            Linear(in_features=3, out_features=4, bias=True),
            Linear(in_features=4, out_features=2, bias=True),
        )

        lin0: Linear = model[0]  # type: ignore[assignment]
        lin1: Linear = model[1]  # type: ignore[assignment]

        # Deterministic weights/bias
        W0 = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
        b0 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        lin0.weight.copy_from_numpy(W0)
        self.assertIsNotNone(lin0.bias)
        lin0.bias.copy_from_numpy(b0)  # type: ignore[union-attr]

        W1 = np.arange(8, dtype=np.float32).reshape(2, 4) / 20.0
        b1 = np.array([0.25, -0.75], dtype=np.float32)
        lin1.weight.copy_from_numpy(W1)
        self.assertIsNotNone(lin1.bias)
        lin1.bias.copy_from_numpy(b1)  # type: ignore[union-attr]

        x_np = np.array([[1.0, -1.0, 2.0], [0.0, 0.5, -0.5]], dtype=np.float32)
        x = Tensor(shape=x_np.shape, device=lin0.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "seq_two_layers.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        # Ensure layers restored
        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], Linear)
        self.assertIsInstance(loaded[1], Linear)

        loaded0: Linear = loaded[0]  # type: ignore[assignment]
        loaded1: Linear = loaded[1]  # type: ignore[assignment]

        # Ensure weights map to the correct layer (order preserved)
        np.testing.assert_allclose(loaded0.weight.to_numpy(), W0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(loaded1.weight.to_numpy(), W1, rtol=0.0, atol=0.0)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_load_json_unsupported_format_raises(self) -> None:
        """
        Ensure load_json rejects unknown checkpoint format values.
        """
        model = Sequential(Linear(in_features=3, out_features=2, bias=True))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "fmt.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            payload["format"] = "some.other.format.v999"
            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                Sequential.load_json(ckpt_path)

    def test_load_json_shape_mismatch_raises(self) -> None:
        """
        Ensure load fails if a parameter payload shape doesn't match the model's parameter shape.
        This simulates a corrupted/tampered checkpoint.
        """
        model = Sequential(Linear(in_features=3, out_features=2, bias=True))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "shape_bad.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            # Pick one param and corrupt its shape metadata
            some_key = next(iter(state.keys()))
            state[some_key]["shape"] = [999, 999]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                Sequential.load_json(ckpt_path)

    def test_load_json_restores_layers_list(self) -> None:
        """
        Regression test: after deserialization, Sequential._layers must be rebuilt
        so indexing and forward() actually use child modules.
        """
        model = Sequential(
            Linear(in_features=3, out_features=2, bias=True),
            Linear(in_features=2, out_features=1, bias=False),
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "layers_restore.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        # This would fail if `_layers` was not reconstructed
        self.assertEqual(len(loaded), 2)
        _ = loaded[0]
        _ = loaded[1]


if __name__ == "__main__":
    unittest.main()
