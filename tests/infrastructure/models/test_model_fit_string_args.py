from __future__ import annotations

import os
import unittest

import numpy as np

from src.keydnn.domain.device._device import Device

RUN_SLOW = os.environ.get("KEYDNN_RUN_SLOW", "0") == "1"


def _cuda_available() -> bool:
    try:
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _tensor_from_numpy(arr: np.ndarray, *, device):
    from src.keydnn.infrastructure.tensor._tensor import Tensor

    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device)
    t.copy_from_numpy(a)
    return t


def _as_float(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)

    if hasattr(x, "to_numpy"):
        v = x.to_numpy()
        v = np.asarray(v)
        return float(v.reshape(-1)[0])

    v = np.asarray(x)
    return float(v.reshape(-1)[0])


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


# ======================================================================================
# Slow integration tests (opt-in) for string args
# ======================================================================================


class _FitStringArgsMixin:
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
        x = _tensor_from_numpy(x_np, device=device)
        y = _tensor_from_numpy(y_np, device=device)

        # Guardrails against silent CPU fallback
        if hasattr(x, "device"):
            self.assertEqual(str(x.device), self.DEVICE_STR)
        if hasattr(y, "device"):
            self.assertEqual(str(y.device), self.DEVICE_STR)

        return x_np, y_np, x, y

    def test_fit_accepts_string_loss_and_optimizer(self):
        """
        Verifies:
        - fit accepts loss="mse" and optimizer="sgd"
        - optimizer_kwargs are forwarded (lr)
        - training solves XOR (integration)
        """
        try:
            from src.keydnn.infrastructure.models._sequential import (
                Sequential,
            )  # noqa: F401
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        device = self._device()
        x_np, y_np, x, y = self._xor_tensors(device)

        model = self._build_model(device)

        def acc_metric(y_true, y_pred):
            yp = np.asarray(y_pred.to_numpy(), dtype=np.float32)
            # metric should use provided y_true (but XOR fixed anyway)
            return float(_accuracy_from_pred_np(y_np, yp))

        epochs = 2000
        history = model.fit(
            [(x, y)],  # iterable-of-batches path
            None,
            loss="mse",
            optimizer="sgd",
            optimizer_kwargs={"lr": 1.0},
            metrics=[acc_metric],
            metric_names=["acc"],
            epochs=epochs,
            verbose=0,
        )

        self.assertTrue(hasattr(history, "history"))
        self.assertTrue(hasattr(history, "epoch"))
        self.assertIn("loss", history.history)
        self.assertIn("acc", history.history)

        self.assertEqual(len(history.epoch), epochs)
        self.assertEqual(len(history.history["loss"]), epochs)
        self.assertEqual(len(history.history["acc"]), epochs)

        self.assertTrue(all(np.isfinite(_as_float(v)) for v in history.history["loss"]))
        self.assertTrue(all(0.0 <= _as_float(v) <= 1.0 for v in history.history["acc"]))

        pred_np = np.asarray(model(x).to_numpy(), dtype=np.float32)
        acc = _accuracy_from_pred_np(y_np, pred_np)
        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low after fit(string args): {acc:.3f}, pred={pred_np.reshape(-1).tolist()}",
        )

    def test_fit_raises_on_unknown_string_loss(self):
        try:
            from src.keydnn.infrastructure.models._sequential import (
                Sequential,
            )  # noqa: F401
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        device = self._device()
        _x_np, _y_np, x, y = self._xor_tensors(device)
        model = self._build_model(device)

        with self.assertRaises(ValueError):
            _ = model.fit(
                [(x, y)],
                None,
                loss="definitely_not_a_loss",
                optimizer="sgd",
                optimizer_kwargs={"lr": 1.0},
                epochs=1,
                verbose=0,
            )

    def test_fit_raises_on_unknown_string_optimizer(self):
        try:
            from src.keydnn.infrastructure.models._sequential import (
                Sequential,
            )  # noqa: F401
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        device = self._device()
        _x_np, _y_np, x, y = self._xor_tensors(device)
        model = self._build_model(device)

        with self.assertRaises(ValueError):
            _ = model.fit(
                [(x, y)],
                None,
                loss="mse",
                optimizer="definitely_not_an_optimizer",
                optimizer_kwargs={"lr": 1.0},
                epochs=1,
                verbose=0,
            )


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
class TestModelFitStringArgsCPU(_FitStringArgsMixin, unittest.TestCase):
    DEVICE_STR = "cpu"


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelFitStringArgsCUDA(_FitStringArgsMixin, unittest.TestCase):
    DEVICE_STR = "cuda:0"


# ======================================================================================
# Fast contract tests (always run): ensure fit resolves strings without running real training
# ======================================================================================


class FakeScalar:
    def __init__(self, value: float):
        self._v = float(value)

    def backward(self) -> None:
        pass

    def to_numpy(self):
        return [self._v]


class FakeOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


class TestModelFitStringArgsContract(unittest.TestCase):
    def test_fit_resolves_string_loss_and_optimizer_and_calls_train_on_batch(self):
        """
        Contract test:
        - passing loss="mse" triggers resolver (no exception)
        - passing optimizer="sgd" triggers resolver (no exception)
        - fit still aggregates history lengths as usual

        We stub train_on_batch to avoid depending on real tensor/loss/optimizer internals.
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        m = Model()

        # NEW: provide a minimal forward so build() can run
        m.forward = lambda x: x  # type: ignore[method-assign]

        dummy_x = _tensor_from_numpy(
            np.array([[0.0]], dtype=np.float32),
            device=Device("cpu"),
        )
        m.build(dummy_x[:1])

        # Provide parameters() so optimizer resolver can construct SGD(model.parameters(), ...)
        # If Model already has parameters(), this override is harmless but not required.
        m.parameters = lambda: []  # type: ignore[assignment]

        # Monkeypatch train_on_batch so training is deterministic.
        train_calls = {"n": 0}

        def fake_train_on_batch(*args, **kwargs):
            _ = args, kwargs
            train_calls["n"] += 1
            return {"loss": 1.0, "acc": 0.5}

        m.train_on_batch = fake_train_on_batch  # type: ignore[assignment]

        batches = [("xb1", "yb1"), ("xb2", "yb2"), ("xb3", "yb3")]

        hist = m.fit(
            batches,
            None,
            loss="mse",
            optimizer="sgd",
            optimizer_kwargs={"lr": 0.1},
            epochs=5,
            verbose=0,
        )

        self.assertEqual(train_calls["n"], 5 * len(batches))

        self.assertTrue(hasattr(hist, "history"))
        self.assertTrue(hasattr(hist, "epoch"))
        self.assertEqual(len(hist.epoch), 5)

        self.assertIn("loss", hist.history)
        self.assertIn("acc", hist.history)
        self.assertEqual(len(hist.history["loss"]), 5)
        self.assertEqual(len(hist.history["acc"]), 5)

        self.assertTrue(all(v == 1.0 for v in hist.history["loss"]))
        self.assertTrue(all(v == 0.5 for v in hist.history["acc"]))


if __name__ == "__main__":
    unittest.main()
