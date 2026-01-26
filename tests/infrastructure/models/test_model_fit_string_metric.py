from __future__ import annotations

import os
import unittest
from typing import Any, Dict, List

import numpy as np

RUN_SLOW = os.environ.get("KEYDNN_RUN_SLOW", "0") == "1"

from src.keydnn.domain.device._device import Device


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _tensor_from_numpy(arr: np.ndarray, *, device, requires_grad: bool = False):
    from src.keydnn.infrastructure.tensor._tensor import Tensor

    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device, requires_grad=requires_grad)
    t.copy_from_numpy(a)
    return t


def _xor_data_numpy():
    x_np = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)
    return x_np, y_np


def _accuracy_from_pred_np(y_true_np: np.ndarray, pred_np: np.ndarray) -> float:
    y_hat = (pred_np >= 0.5).astype(np.float32)
    return float((y_hat == y_true_np).mean())


class TestModelFitMetricStringArgsContract(unittest.TestCase):
    def test_fit_resolves_metrics_string_acc_and_passes_callable_to_train_on_batch(
        self,
    ):
        """
        Contract:
        - fit(metrics=["acc"]) should resolve to a callable metric
        - fit should pass the resolved callable + a stable metric_names entry to train_on_batch
        - history should include "acc"
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        m = Model()

        m.forward = lambda x: x

        dummy_x = _tensor_from_numpy(
            np.array([[0.0]], dtype=np.float32),
            device=Device("cpu"),
        )
        m.build(dummy_x[:1])

        calls: List[Dict[str, Any]] = []

        def fake_train_on_batch(
            xb, yb, *, loss, optimizer, metrics=None, metric_names=None, **kwargs
        ):

            calls.append(
                {
                    "metrics": metrics,
                    "metric_names": metric_names,
                    "loss": loss,
                    "optimizer": optimizer,
                }
            )

            self.assertIsNotNone(metrics)
            self.assertTrue(isinstance(metrics, (list, tuple)))
            self.assertEqual(len(metrics), 1)
            self.assertTrue(callable(metrics[0]))

            self.assertIsNotNone(metric_names)
            self.assertTrue(isinstance(metric_names, (list, tuple)))
            self.assertEqual(len(metric_names), 1)
            self.assertEqual(str(metric_names[0]).lower(), "acc")

            return {"loss": 1.0, str(metric_names[0]): 0.5}

        m.train_on_batch = fake_train_on_batch

        batches = [("xb1", "yb1"), ("xb2", "yb2")]

        hist = m.fit(
            batches,
            None,
            loss=lambda yp, yt: 1.0,
            optimizer=object(),
            metrics=["acc"],
            epochs=3,
            verbose=0,
        )

        self.assertTrue(hasattr(hist, "history"))
        self.assertTrue(hasattr(hist, "epoch"))
        self.assertEqual(len(hist.epoch), 3)
        self.assertIn("loss", hist.history)
        self.assertIn("acc", hist.history)
        self.assertEqual(len(hist.history["loss"]), 3)
        self.assertEqual(len(hist.history["acc"]), 3)
        self.assertEqual(hist.history["loss"], [1.0, 1.0, 1.0])
        self.assertEqual(hist.history["acc"], [0.5, 0.5, 0.5])

        self.assertEqual(len(calls), 3 * len(batches))

    def test_fit_rejects_unknown_metric_string(self):
        """
        fit(metrics=["..."]) should raise ValueError for unknown built-in metric names.
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        m = Model()

        m.forward = lambda x: x

        dummy_x = _tensor_from_numpy(
            np.array([[0.0]], dtype=np.float32),
            device=Device("cpu"),
        )
        m.build(dummy_x[:1])

        batches = [("xb", "yb")]

        with self.assertRaises(ValueError):
            _ = m.fit(
                batches,
                None,
                loss=lambda yp, yt: 1.0,
                optimizer=object(),
                metrics=["not_a_metric"],
                epochs=1,
                verbose=0,
            )

    def test_fit_metric_names_length_mismatch_raises(self):
        """
        If metric_names is provided, its length must match metrics length.
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        m = Model()

        m.forward = lambda x: x

        dummy_x = _tensor_from_numpy(
            np.array([[0.0]], dtype=np.float32),
            device=Device("cpu"),
        )
        m.build(dummy_x[:1])

        batches = [("xb", "yb")]

        with self.assertRaises(ValueError):
            _ = m.fit(
                batches,
                None,
                loss=lambda yp, yt: 1.0,
                optimizer=object(),
                metrics=["acc"],
                metric_names=["acc", "extra"],
                epochs=1,
                verbose=0,
            )


class _FitMetricStringAccIntegrationMixin:
    DEVICE_STR: str = "cpu"

    def _device(self):
        from src.keydnn.domain.device._device import Device

        return Device(self.DEVICE_STR)

    def _build_model(self, device):
        from src.keydnn.infrastructure.models._sequential import Sequential
        from src.keydnn.infrastructure.fully_connected._linear import Linear
        from src.keydnn.infrastructure.activations._modules import Sigmoid

        hidden_dim = 8
        if self.DEVICE_STR.startswith("cuda"):
            return Sequential(
                Linear(2, hidden_dim, device=device),
                Sigmoid(),
                Linear(hidden_dim, 1, device=device),
                Sigmoid(),
            )
        return Sequential(
            Linear(2, hidden_dim),
            Sigmoid(),
            Linear(hidden_dim, 1),
            Sigmoid(),
        )

    def _xor_tensors(self, device):
        x_np, y_np = _xor_data_numpy()

        repeats = 256
        x_big = np.repeat(x_np, repeats=repeats, axis=0)
        y_big = np.repeat(y_np, repeats=repeats, axis=0)

        x = _tensor_from_numpy(x_big, device=device, requires_grad=False)
        y = _tensor_from_numpy(y_big, device=device, requires_grad=False)

        if hasattr(x, "device"):
            self.assertEqual(str(x.device), self.DEVICE_STR)
        if hasattr(y, "device"):
            self.assertEqual(str(y.device), self.DEVICE_STR)

        return x_np, y_np, x, y

    def test_fit_with_metrics_acc_string_learns_xor(self):
        """
        Integration:
        - fit(loss="mse", optimizer="sgd", metrics=["acc"]) should learn XOR
        - history should contain "acc"
        """
        device = self._device()
        x_base, y_base, x, y = self._xor_tensors(device)

        model = self._build_model(device)

        hist = model.fit(
            x,
            y,
            loss="mse",
            optimizer="sgd",
            optimizer_kwargs={"lr": 1.0},
            metrics=["acc"],
            batch_size=32,
            epochs=2000,
            shuffle=True,
            verbose=0,
        )

        self.assertTrue(hasattr(hist, "history"))
        self.assertIn("loss", hist.history)
        self.assertIn("acc", hist.history)
        self.assertEqual(len(hist.history["acc"]), len(hist.epoch))

        x_eval = _tensor_from_numpy(x_base, device=device, requires_grad=False)
        pred = model(x_eval)
        pred_np = np.asarray(pred.to_numpy(), dtype=np.float32)

        acc = _accuracy_from_pred_np(y_base, pred_np)
        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low with metrics='acc': {acc:.3f}, pred={pred_np.reshape(-1).tolist()}",
        )


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
class TestModelFitMetricStringAccCPU(
    _FitMetricStringAccIntegrationMixin, unittest.TestCase
):
    DEVICE_STR = "cpu"


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelFitMetricStringAccCUDA(
    _FitMetricStringAccIntegrationMixin, unittest.TestCase
):
    DEVICE_STR = "cuda:0"


if __name__ == "__main__":
    unittest.main()
