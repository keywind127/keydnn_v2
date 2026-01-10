# tests/infrastructure/models/saving_and_loading/cuda/test_activations.py
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure._activations import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
)


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _make_input(device: Device) -> Tensor:
    x_np = np.array(
        [
            [1.0, -2.0, 0.5, 3.0],
            [-1.0, 2.0, -0.25, 0.0],
        ],
        dtype=np.float32,
    )
    x = Tensor(shape=x_np.shape, device=device, requires_grad=False, dtype=x_np.dtype)

    if device.is_cuda():
        x._ensure_cuda_alloc(dtype=np.dtype(x_np.dtype))

    x.copy_from_numpy(x_np)
    return x


class TestModelSaveLoadActivationsJSON(unittest.TestCase):
    def test_save_load_json_activation_stack_preserves_types_and_hparams(self) -> None:
        """
        Ensure a Sequential containing only activation modules can be saved and loaded
        with identical architecture (types + hyperparameters).

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
            self.assertIsInstance(payload["state"], dict)
            # Activations are parameter-free -> state should be empty
            self.assertEqual(len(payload["state"]), 0)

            loaded = Sequential.load_json(ckpt_path)

        self.assertEqual(len(loaded), 5)
        self.assertIsInstance(loaded[0], Sigmoid)
        self.assertIsInstance(loaded[1], ReLU)
        self.assertIsInstance(loaded[2], LeakyReLU)
        self.assertIsInstance(loaded[3], Tanh)
        self.assertIsInstance(loaded[4], Softmax)

        loaded_lrelu: LeakyReLU = loaded[2]  # type: ignore[assignment]
        self.assertAlmostEqual(float(getattr(loaded_lrelu, "alpha")), 0.123, places=7)

        loaded_softmax: Softmax = loaded[4]  # type: ignore[assignment]
        self.assertEqual(int(getattr(loaded_softmax, "_axis")), 1)

    def test_save_load_json_activation_forward_matches_cpu(self) -> None:
        """
        Ensure forward outputs match exactly before vs after save/load for a pure
        activation stack on CPU.
        """
        np.random.seed(0)

        model = Sequential(
            Sigmoid(),
            ReLU(),
            LeakyReLU(alpha=0.2),
            Tanh(),
            Softmax(axis=1),
        )

        x = _make_input(Device("cpu"))
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "activations_forward_cpu.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    @unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
    def test_save_load_json_activation_forward_matches_cuda(self) -> None:
        """
        Same forward-preservation check as CPU, but executed on CUDA tensors.

        This validates:
        - activation ops work on CUDA
        - save/load works while the model is used with CUDA tensors
          (activations are parameter-free, so this mostly tests arch round-trip + CUDA forward)
        """
        np.random.seed(0)

        model = Sequential(
            Sigmoid(),
            ReLU(),
            LeakyReLU(alpha=0.2),
            Tanh(),
            Softmax(axis=1),
        )

        x = _make_input(Device("cuda:0"))
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "activations_forward_cuda.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_activation_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if the checkpoint is missing a required top-level key.
        """
        model = Sequential(Sigmoid(), ReLU())

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_activations.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            del payload["arch"]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)


if __name__ == "__main__":
    unittest.main()
