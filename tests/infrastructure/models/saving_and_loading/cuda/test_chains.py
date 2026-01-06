# tests/infrastructure/models/saving_and_loading/test_model_save_load_chains_cpu_and_cuda_json.py
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure._models import Sequential
from src.keydnn.infrastructure.fully_connected._linear import Linear
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from src.keydnn.infrastructure.flatten._flatten_module import Flatten
from src.keydnn.infrastructure._activations import ReLU, Sigmoid, Softmax
from src.keydnn.infrastructure.recurrent._rnn_module import RNN


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _tensor_from_numpy(arr: np.ndarray, *, device: Device) -> Tensor:
    """
    Create a Tensor using public APIs only, and ensure CUDA allocation before H2D copy.
    """
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=False, dtype=arr.dtype)

    if device.is_cuda():
        # Ensure devptr != 0 before H2D copy.
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))

    t.copy_from_numpy(arr)
    return t


class _ChainsRoundtripMixin:
    """
    Implements chained-model JSON round-trip tests for a fixed device.
    Subclasses must define DEVICE.
    """

    DEVICE: Device

    # def test_save_load_json_rnn_linear_activation_forward_matches(self) -> None:
    #     """
    #     RNN -> Linear -> Sigmoid
    #     Ensures the chained model round-trips via JSON and forward outputs match.

    #     Notes
    #     -----
    #     We configure RNN to output only h_T as a Tensor so it can be chained.
    #     """
    #     np.random.seed(0)

    #     model = Sequential(
    #         RNN(
    #             input_size=3,
    #             hidden_size=4,
    #             bias=True,
    #             return_sequences=False,
    #             return_state=False,
    #             keras_compat=True,
    #             device=self.DEVICE,
    #         ),
    #         Linear(in_features=4, out_features=2, bias=True, device=self.DEVICE),
    #         Sigmoid(),
    #     )

    #     rnn: RNN = model[0]  # type: ignore[assignment]
    #     lin: Linear = model[1]  # type: ignore[assignment]

    #     # Deterministic Linear params
    #     W = np.array(
    #         [[1.0, -2.0, 3.0, -4.0], [-1.5, 2.5, -3.5, 4.5]],
    #         dtype=np.float32,
    #     )
    #     b = np.array([0.25, -0.75], dtype=np.float32)
    #     lin.weight.copy_from_numpy(W)
    #     self.assertIsNotNone(lin.bias)
    #     lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

    #     # Input x: (T, N, D)
    #     x_np = np.array(
    #         [
    #             [[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]],
    #             [[0.0, -1.0, 2.0], [1.0, 1.0, 0.0]],
    #             [[-1.0, 0.5, 0.5], [2.0, -0.5, 1.5]],
    #         ],
    #         dtype=np.float32,
    #     )

    #     x = _tensor_from_numpy(x_np, device=self.DEVICE)
    #     y_before = model.forward(x).to_numpy()

    #     with tempfile.TemporaryDirectory() as td:
    #         ckpt = Path(td) / f"rnn_linear_sigmoid_{self.DEVICE}.json"
    #         model.save_json(ckpt)
    #         self.assertTrue(ckpt.exists())

    #         payload = json.loads(ckpt.read_text(encoding="utf-8"))
    #         self.assertEqual(payload.get("format"), "keydnn.json.ckpt.v1")

    #         loaded = Sequential.load_json(ckpt)

    #     y_after = loaded.forward(x).to_numpy()
    #     np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_cnn_linear_activation_forward_matches(self) -> None:
        """
        Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> Softmax
        Ensures the chained CNN-style model round-trips via JSON and forward outputs match.
        """
        np.random.seed(0)

        model = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                padding=1,
                bias=True,
                device=self.DEVICE,
            ),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            Linear(in_features=8, out_features=3, bias=True, device=self.DEVICE),
            Softmax(axis=-1),
        )

        conv: Conv2d = model[0]  # type: ignore[assignment]
        lin: Linear = model[4]  # type: ignore[assignment]

        # Deterministic Conv2d params
        conv_w = np.array(
            [
                [[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]],
                [[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [-0.5, -0.5, -0.5]]],
            ],
            dtype=np.float32,
        )
        conv_b = np.array([0.1, -0.2], dtype=np.float32)
        conv.weight.copy_from_numpy(conv_w)
        self.assertIsNotNone(conv.bias)
        conv.bias.copy_from_numpy(conv_b)  # type: ignore[union-attr]

        # Deterministic Linear params (3 x 8)
        W = np.arange(24, dtype=np.float32).reshape(3, 8) / 10.0
        b = np.array([0.05, -0.05, 0.1], dtype=np.float32)
        lin.weight.copy_from_numpy(W)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(b)  # type: ignore[union-attr]

        # Input: (N, C, H, W) = (2,1,4,4)
        x_np = np.array(
            [
                [[[1, 2, 3, 4], [0, 1, 0, 1], [2, 0, 2, 0], [1, 1, 1, 1]]],
                [[[0, 1, 0, 1], [2, 2, 2, 2], [3, 0, 0, 3], [4, 1, 1, 4]]],
            ],
            dtype=np.float32,
        )

        x = _tensor_from_numpy(x_np, device=self.DEVICE)
        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / f"cnn_linear_softmax_{self.DEVICE}.json"
            model.save_json(ckpt)
            loaded = Sequential.load_json(ckpt)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)

    def test_save_load_json_deep_cnn_chain_order_preserved(self) -> None:
        """
        Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> Sigmoid
        """
        np.random.seed(0)

        model = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                padding=1,
                bias=False,
                device=self.DEVICE,
            ),
            ReLU(),
            Conv2d(
                in_channels=2,
                out_channels=2,
                kernel_size=3,
                padding=1,
                bias=True,
                device=self.DEVICE,
            ),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            Linear(in_features=8, out_features=2, bias=True, device=self.DEVICE),
            Sigmoid(),
        )

        conv1: Conv2d = model[0]  # type: ignore[assignment]
        conv2: Conv2d = model[2]  # type: ignore[assignment]
        lin: Linear = model[6]  # type: ignore[assignment]

        # Deterministic weights
        conv1.weight.copy_from_numpy(
            np.ones_like(conv1.weight.to_numpy(), dtype=np.float32) * 0.1
        )

        conv2_w = np.ones_like(conv2.weight.to_numpy(), dtype=np.float32) * (-0.05)
        conv2.weight.copy_from_numpy(conv2_w)
        self.assertIsNotNone(conv2.bias)
        conv2.bias.copy_from_numpy(np.array([0.01, -0.02], dtype=np.float32))  # type: ignore[union-attr]

        lin.weight.copy_from_numpy(np.ones((2, 8), dtype=np.float32) * 0.2)
        self.assertIsNotNone(lin.bias)
        lin.bias.copy_from_numpy(np.array([0.0, 0.1], dtype=np.float32))  # type: ignore[union-attr]

        x_np = np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4) / 10.0
        x = _tensor_from_numpy(x_np, device=self.DEVICE)

        y_before = model.forward(x).to_numpy()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / f"deep_cnn_chain_{self.DEVICE}.json"
            model.save_json(ckpt)
            loaded = Sequential.load_json(ckpt)

        # Quick order sanity
        self.assertEqual(type(loaded[0]).__name__, type(model[0]).__name__)
        self.assertEqual(type(loaded[2]).__name__, type(model[2]).__name__)
        self.assertEqual(type(loaded[6]).__name__, type(model[6]).__name__)

        y_after = loaded.forward(x).to_numpy()
        np.testing.assert_allclose(y_after, y_before, rtol=0.0, atol=0.0)


class TestModelSaveLoadChainedModelsJSONCpu(_ChainsRoundtripMixin, unittest.TestCase):
    DEVICE = Device("cpu")


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelSaveLoadChainedModelsJSONCuda(_ChainsRoundtripMixin, unittest.TestCase):
    DEVICE = Device("cuda:0")


if __name__ == "__main__":
    unittest.main()
