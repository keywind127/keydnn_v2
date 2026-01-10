from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure.recurrent._rnn_module import RNN


class TestModelSaveLoadRNNJSON(unittest.TestCase):
    def _make_time_major_input(self, *, device, T: int, N: int, D: int) -> Tensor:
        """
        Create a deterministic (T, N, D) input Tensor on the provided device.
        """
        x_np = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D) / 10.0
        x = Tensor(shape=x_np.shape, device=device, requires_grad=False)
        x.copy_from_numpy(x_np)
        return x

    def test_save_load_json_rnn_weights_and_forward_match(self) -> None:
        """
        Ensure an RNN inside a Sequential can be saved to JSON and loaded back with:
        - identical weights
        - identical forward outputs

        Notes
        -----
        We force the RNN to return a single Tensor so Sequential works:
        keras_compat=True, return_sequences=False, return_state=False -> returns h_T
        """
        np.random.seed(0)

        model = Sequential(
            RNN(
                input_size=3,
                hidden_size=4,
                bias=True,
                return_sequences=False,
                return_state=False,
                keras_compat=True,
            )
        )

        rnn: RNN = model[0]  # type: ignore[assignment]
        cell = rnn.cell

        # Use the device owned by the cell parameters (avoid constructing a new Device in tests)
        dev = cell.W_ih.device

        # Overwrite weights/bias with deterministic values
        W_ih = (np.arange(3 * 4, dtype=np.float32).reshape(3, 4) - 5.0) / 10.0
        W_hh = (np.arange(4 * 4, dtype=np.float32).reshape(4, 4) - 7.0) / 10.0
        b_ih = np.array([0.10, -0.20, 0.30, -0.40], dtype=np.float32)
        b_hh = np.array([-0.01, 0.02, -0.03, 0.04], dtype=np.float32)

        cell.W_ih.copy_from_numpy(W_ih)
        cell.W_hh.copy_from_numpy(W_hh)
        self.assertIsNotNone(cell.b_ih)
        self.assertIsNotNone(cell.b_hh)
        cell.b_ih.copy_from_numpy(b_ih)  # type: ignore[union-attr]
        cell.b_hh.copy_from_numpy(b_hh)  # type: ignore[union-attr]

        # Input (T, N, D)
        x = self._make_time_major_input(device=dev, T=5, N=2, D=3)

        # Forward before save
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "rnn.json"
            model.save_json(ckpt_path)

            # Sanity-check payload
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        loaded_rnn: RNN = loaded[0]  # type: ignore[assignment]
        loaded_cell = loaded_rnn.cell

        # Check config survived
        self.assertTrue(loaded_rnn.keras_compat)
        self.assertFalse(loaded_rnn.return_sequences)
        self.assertFalse(loaded_rnn.return_state)

        # Check weights survived
        np.testing.assert_allclose(
            loaded_cell.W_ih.to_numpy(), W_ih, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            loaded_cell.W_hh.to_numpy(), W_hh, rtol=0.0, atol=0.0
        )
        self.assertIsNotNone(loaded_cell.b_ih)
        self.assertIsNotNone(loaded_cell.b_hh)
        np.testing.assert_allclose(
            loaded_cell.b_ih.to_numpy(),  # type: ignore[union-attr]
            b_ih,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            loaded_cell.b_hh.to_numpy(),  # type: ignore[union-attr]
            b_hh,
            rtol=0.0,
            atol=0.0,
        )

        # Forward after load
        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_rnn_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if the checkpoint state is missing a parameter.
        """
        model = Sequential(
            RNN(
                input_size=3,
                hidden_size=4,
                bias=True,
                return_sequences=False,
                return_state=False,
                keras_compat=True,
            )
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_rnn.json"
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

    def test_save_load_json_rnn_bias_false_roundtrip(self) -> None:
        """
        Ensure RNN(bias=False) round-trips properly and still matches forward output.
        """
        np.random.seed(0)

        model = Sequential(
            RNN(
                input_size=2,
                hidden_size=3,
                bias=False,
                return_sequences=False,
                return_state=False,
                keras_compat=True,
            )
        )

        rnn: RNN = model[0]  # type: ignore[assignment]
        cell = rnn.cell
        dev = cell.W_ih.device

        # deterministic weights (no biases)
        W_ih = (np.arange(2 * 3, dtype=np.float32).reshape(2, 3) - 2.0) / 10.0
        W_hh = (np.arange(3 * 3, dtype=np.float32).reshape(3, 3) - 3.0) / 10.0
        cell.W_ih.copy_from_numpy(W_ih)
        cell.W_hh.copy_from_numpy(W_hh)
        self.assertIsNone(cell.b_ih)
        self.assertIsNone(cell.b_hh)

        x = self._make_time_major_input(device=dev, T=4, N=2, D=2)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "rnn_bias_false.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        loaded_rnn: RNN = loaded[0]  # type: ignore[assignment]
        loaded_cell = loaded_rnn.cell

        # weights match
        np.testing.assert_allclose(
            loaded_cell.W_ih.to_numpy(), W_ih, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            loaded_cell.W_hh.to_numpy(), W_hh, rtol=0.0, atol=0.0
        )
        self.assertIsNone(loaded_cell.b_ih)
        self.assertIsNone(loaded_cell.b_hh)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
