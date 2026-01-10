# tests/infrastructure/models/saving_and_loading/cuda/test_flatten.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.models._sequential import Sequential
from src.keydnn.infrastructure.fully_connected._linear import Linear
from src.keydnn.infrastructure.flatten._flatten_module import Flatten

from src.keydnn.domain.device._device import Device


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _make_input_on_device(device: Device, x_np: np.ndarray) -> Tensor:
    x_np = x_np.astype(np.float32, copy=False)
    x = Tensor(shape=x_np.shape, device=device, requires_grad=False, dtype=x_np.dtype)

    if device.is_cuda():
        x._ensure_cuda_alloc(dtype=np.dtype(x_np.dtype))

    x.copy_from_numpy(x_np)
    return x


class TestModelSaveLoadFlattenJSONCpu(unittest.TestCase):
    def test_save_load_json_flatten_arch_and_forward_match_cpu(self) -> None:
        """
        CPU: Flatten -> Linear save/load round-trip preserves weights and forward output.

        Note: On CPU tests we avoid constructing Device manually; modules pick defaults
        and we build input tensors on the module's device.
        """
        np.random.seed(0)

        model = Sequential(
            Flatten(),
            Linear(in_features=6, out_features=4, bias=True),
        )

        lin: Linear = model[1]  # type: ignore[assignment]
        W = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 10.0
        b = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        x_np = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) / 5.0
        x = Tensor(shape=x_np.shape, device=lin.device, requires_grad=False)
        x.copy_from_numpy(x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "flatten_linear_cpu.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

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
        CPU: missing parameter key in checkpoint should raise KeyError,
        even when Flatten is present in architecture.
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

            lin_keys = [
                k for k in state.keys() if k.endswith(".weight") or k.endswith(".bias")
            ]
            key_to_remove = lin_keys[0] if lin_keys else next(iter(state.keys()))
            del state[key_to_remove]

            ckpt_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelSaveLoadFlattenJSONCuda(unittest.TestCase):
    def test_save_load_json_flatten_arch_and_forward_match_cuda(self) -> None:
        """
        CUDA: Flatten -> Linear save/load round-trip preserves weights and forward output.

        We explicitly construct Device("cuda:0") here because tests must choose a CUDA device.
        """
        np.random.seed(0)
        dev = Device("cuda:0")

        model = Sequential(
            Flatten(),
            Linear(in_features=6, out_features=4, bias=True, device=dev),
        )

        lin: Linear = model[1]  # type: ignore[assignment]
        self.assertTrue(lin.device.is_cuda())

        W = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 10.0
        b = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        x_np = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) / 5.0
        x = _make_input_on_device(dev, x_np)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "flatten_linear_cuda.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

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


if __name__ == "__main__":
    unittest.main()
