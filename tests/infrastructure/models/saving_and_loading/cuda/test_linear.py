# tests/infrastructure/models/saving_and_loading/cuda/test_linear.py
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import json

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure._models import Sequential
from src.keydnn.infrastructure.fully_connected._linear import Linear


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


def _alloc_cuda_tensor_from_numpy(arr: np.ndarray, *, device: Device) -> Tensor:
    """
    Allocate a CUDA Tensor and copy NumPy data into it.

    Rationale
    ---------
    In this codebase, CUDA tensors commonly start with devptr=0 and require an
    explicit allocation step via `_ensure_cuda_alloc(...)` before copies/ops.
    """
    t = Tensor(shape=arr.shape, device=device, requires_grad=False, dtype=arr.dtype)
    # Ensure device memory exists (no-op if already allocated).
    t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    t.copy_from_numpy(arr)
    return t


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelSaveLoadLinearJSONCuda(unittest.TestCase):
    def test_save_load_json_linear_cuda_weights_and_forward_match(self) -> None:
        """
        Ensure a model containing a CUDA Linear layer can be saved to JSON and
        loaded back with identical weights and forward outputs.
        """
        np.random.seed(0)
        dev = Device("cuda:0")

        model = Sequential(
            Linear(in_features=3, out_features=2, bias=True, device=dev),
        )

        lin: Linear = model[0]  # type: ignore[assignment]
        self.assertTrue(lin.device.is_cuda(), f"Expected CUDA device, got {lin.device}")

        W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b = np.array([0.5, -1.5], dtype=np.float32)

        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]], dtype=np.float32)
        x = _alloc_cuda_tensor_from_numpy(x_np, device=lin.device)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "linear_cuda.json"
            model.save_json(ckpt_path)

            self.assertTrue(ckpt_path.exists())
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")
            self.assertIn("arch", payload)
            self.assertIn("state", payload)

            loaded = Sequential.load_json(ckpt_path)

        loaded_lin: Linear = loaded[0]  # type: ignore[assignment]

        # Depending on your deserialization policy, this might come back as CPU.
        # If you serialize device and restore it, this should be CUDA.
        self.assertTrue(
            loaded_lin.device.is_cuda(),
            f"Expected loaded model on CUDA; got {loaded_lin.device}",
        )

        np.testing.assert_allclose(loaded_lin.weight.to_numpy(), W, rtol=0.0, atol=0.0)
        self.assertIsNotNone(loaded_lin.bias)
        np.testing.assert_allclose(
            loaded_lin.bias.to_numpy(),  # type: ignore[union-attr]
            b,
            rtol=0.0,
            atol=0.0,
        )

        x2 = _alloc_cuda_tensor_from_numpy(x_np, device=loaded_lin.device)
        y_after = loaded.forward(x2).to_numpy()

        # If this flakes due to CUDA math/order, relax to 1e-6.
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_linear_cuda_no_bias_roundtrip(self) -> None:
        """
        Ensure Linear(bias=False) round-trips on CUDA:
        - architecture restores with bias disabled
        - weights restore correctly
        - forward output matches
        """
        np.random.seed(0)
        dev = Device("cuda:0")

        model = Sequential(Linear(in_features=3, out_features=2, bias=False, device=dev))
        lin: Linear = model[0]  # type: ignore[assignment]
        self.assertTrue(lin.device.is_cuda())

        W = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -4.0]], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNone(lin.bias)

        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = _alloc_cuda_tensor_from_numpy(x_np, device=lin.device)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "linear_cuda_nobias.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        loaded_lin: Linear = loaded[0]  # type: ignore[assignment]
        self.assertTrue(loaded_lin.device.is_cuda())
        self.assertIsNone(loaded_lin.bias)
        np.testing.assert_allclose(loaded_lin.weight.to_numpy(), W, rtol=0.0, atol=0.0)

        x2 = _alloc_cuda_tensor_from_numpy(x_np, device=loaded_lin.device)
        y_after = loaded.forward(x2).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_sequential_two_layers_cuda_preserves_order(self) -> None:
        """
        Ensure a multi-layer CUDA Sequential round-trips and preserves layer order.
        """
        np.random.seed(0)
        dev = Device("cuda:0")

        model = Sequential(
            Linear(in_features=3, out_features=4, bias=True, device=dev),
            Linear(in_features=4, out_features=2, bias=True, device=dev),
        )

        lin0: Linear = model[0]  # type: ignore[assignment]
        lin1: Linear = model[1]  # type: ignore[assignment]
        self.assertTrue(lin0.device.is_cuda())
        self.assertTrue(lin1.device.is_cuda())

        W0 = (np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0)
        b0 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        lin0.weight.copy_from_numpy(W0)
        self.assertIsNotNone(lin0.bias)
        lin0.bias.copy_from_numpy(b0)  # type: ignore[union-attr]

        W1 = (np.arange(8, dtype=np.float32).reshape(2, 4) / 20.0)
        b1 = np.array([0.25, -0.75], dtype=np.float32)
        lin1.weight.copy_from_numpy(W1)
        self.assertIsNotNone(lin1.bias)
        lin1.bias.copy_from_numpy(b1)  # type: ignore[union-attr]

        x_np = np.array([[1.0, -1.0, 2.0], [0.0, 0.5, -0.5]], dtype=np.float32)
        x = _alloc_cuda_tensor_from_numpy(x_np, device=lin0.device)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "seq_two_layers_cuda.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], Linear)
        self.assertIsInstance(loaded[1], Linear)

        loaded0: Linear = loaded[0]  # type: ignore[assignment]
        loaded1: Linear = loaded[1]  # type: ignore[assignment]
        self.assertTrue(loaded0.device.is_cuda())
        self.assertTrue(loaded1.device.is_cuda())

        np.testing.assert_allclose(loaded0.weight.to_numpy(), W0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(loaded1.weight.to_numpy(), W1, rtol=0.0, atol=0.0)

        x2 = _alloc_cuda_tensor_from_numpy(x_np, device=loaded0.device)
        y_after = loaded.forward(x2).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_linear_cuda_missing_key_raises(self) -> None:
        """
        Ensure load fails clearly if the checkpoint state is missing a parameter.
        """
        dev = Device("cuda:0")
        model = Sequential(Linear(in_features=3, out_features=2, bias=True, device=dev))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "bad_cuda.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            some_key = next(iter(state.keys()))
            del state[some_key]

            ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            with self.assertRaises(KeyError):
                Sequential.load_json(ckpt_path)

    def test_load_json_cuda_unsupported_format_raises(self) -> None:
        """
        Ensure load_json rejects unknown checkpoint format values (CUDA path too).
        """
        dev = Device("cuda:0")
        model = Sequential(Linear(in_features=3, out_features=2, bias=True, device=dev))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "fmt_cuda.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            payload["format"] = "some.other.format.v999"
            ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            with self.assertRaises(ValueError):
                Sequential.load_json(ckpt_path)

    def test_load_json_cuda_shape_mismatch_raises(self) -> None:
        """
        Ensure load fails if a parameter payload shape doesn't match the model's parameter shape.
        """
        dev = Device("cuda:0")
        model = Sequential(Linear(in_features=3, out_features=2, bias=True, device=dev))

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "shape_bad_cuda.json"
            model.save_json(ckpt_path)

            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
            state = payload["state"]

            some_key = next(iter(state.keys()))
            state[some_key]["shape"] = [999, 999]

            ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            with self.assertRaises(ValueError):
                Sequential.load_json(ckpt_path)

    def test_load_json_cuda_restores_layers_list(self) -> None:
        """
        Regression test: after deserialization, Sequential._layers must be rebuilt
        so indexing and forward() actually use child modules (CUDA too).
        """
        dev = Device("cuda:0")
        model = Sequential(
            Linear(in_features=3, out_features=2, bias=True, device=dev),
            Linear(in_features=2, out_features=1, bias=False, device=dev),
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "layers_restore_cuda.json"
            model.save_json(ckpt_path)
            loaded = Sequential.load_json(ckpt_path)

        self.assertEqual(len(loaded), 2)
        _ = loaded[0]
        _ = loaded[1]


if __name__ == "__main__":
    unittest.main()
