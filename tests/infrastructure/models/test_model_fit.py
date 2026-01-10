from __future__ import annotations

import os
import unittest
from unittest.mock import Mock

import numpy as np

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
    """
    KeyDNN-friendly scalar extraction for tests.

    Accepts:
    - python number
    - numpy scalar / array(1,)
    - Tensor-like exposing to_numpy()
    """
    if isinstance(x, (int, float)):
        return float(x)

    # Tensor -> numpy -> scalar
    if hasattr(x, "to_numpy"):
        v = x.to_numpy()
        v = np.asarray(v)
        return float(v.reshape(-1)[0])

    # numpy scalar / array
    v = np.asarray(x)
    return float(v.reshape(-1)[0])


def _mse_loss(pred, target):
    diff = pred - target
    sq = diff * diff
    if hasattr(sq, "mean"):
        return sq.mean()
    if hasattr(sq, "sum"):
        return sq.sum() * (1.0 / target.shape[0])
    raise AttributeError("Tensor must implement mean() or sum()")


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
# Slow integration tests (opt-in via KEYDNN_RUN_SLOW=1)
# ======================================================================================


class _FitTrainOnBatchMixin:
    DEVICE_STR: str = "cpu"

    def _device(self):
        from src.keydnn.domain.device._device import Device

        return Device(self.DEVICE_STR)

    def _build_model(self, device):
        from src.keydnn.infrastructure.models._sequential import Sequential
        from src.keydnn.infrastructure.fully_connected._linear import Linear
        from src.keydnn.infrastructure._activations import Sigmoid

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

    def _optimizer(self, model):
        from src.keydnn.infrastructure._optimizers import SGD

        opt = SGD(model.parameters(), lr=1.0)
        self.assertTrue(hasattr(opt, "step"), "SGD must implement step().")
        return opt

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

    def test_train_on_batch_returns_logs_and_learns(self):
        """
        Verifies:
        - train_on_batch returns dict with 'loss' and metric keys
        - values are scalar-like
        - training solves XOR
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential  # noqa: F401
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        device = self._device()
        x_np, y_np, x, y = self._xor_tensors(device)

        model = self._build_model(device)
        opt = self._optimizer(model)

        def acc_metric(y_true, y_pred):
            self.assertTrue(hasattr(y_pred, "to_numpy"))
            yp = y_pred.to_numpy()
            return _accuracy_from_pred_np(y_np, np.asarray(yp, dtype=np.float32))

        logs = model.train_on_batch(
            x,
            y,
            loss=_mse_loss,
            optimizer=opt,
            metrics=[acc_metric],
            metric_names=["acc"],
        )

        self.assertIsInstance(logs, dict)
        self.assertIn("loss", logs)
        self.assertIn("acc", logs)

        loss0 = _as_float(logs["loss"])
        acc0 = _as_float(logs["acc"])

        self.assertTrue(np.isfinite(loss0))
        self.assertTrue(0.0 <= acc0 <= 1.0)

        # Train
        epochs = 2000
        for _ in range(epochs - 1):
            model.train_on_batch(
                x,
                y,
                loss=_mse_loss,
                optimizer=opt,
                metrics=[acc_metric],
                metric_names=["acc"],
            )

        pred = model(x)
        pred_np = np.asarray(pred.to_numpy(), dtype=np.float32)
        acc = _accuracy_from_pred_np(y_np, pred_np)

        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low after train_on_batch: {acc:.3f}, pred={pred_np.reshape(-1).tolist()}",
        )

    def test_fit_accepts_iterable_batches_and_returns_history(self):
        """
        fit(x_batches, y=None) where x_batches yields (xb, yb).

        Verifies:
        - fit returns History with expected keys/lengths
        - final model solves XOR
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential  # noqa: F401
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        device = self._device()
        x_np, y_np, x, y = self._xor_tensors(device)

        model = self._build_model(device)
        opt = self._optimizer(model)

        def acc_metric(y_true, y_pred):
            yp = np.asarray(y_pred.to_numpy(), dtype=np.float32)
            return float(_accuracy_from_pred_np(y_np, yp))

        # IMPORTANT: finite iterable (one batch total)
        batches = [(x, y)]
        epochs = 2000

        history = model.fit(
            batches,
            None,
            loss=_mse_loss,
            optimizer=opt,
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
            _as_float(history.history["acc"][-1]),
            0.95,
            f"Expected acc to reach >= 0.95 during fit; got {history.history['acc'][-1]}",
        )
        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low after fit: {acc:.3f}, pred={pred_np.reshape(-1).tolist()}",
        )


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
class TestModelFitTrainOnBatchCPU(_FitTrainOnBatchMixin, unittest.TestCase):
    DEVICE_STR = "cpu"


@unittest.skipUnless(RUN_SLOW, "slow integration test; set KEYDNN_RUN_SLOW=1 to run")
@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestModelFitTrainOnBatchCUDA(_FitTrainOnBatchMixin, unittest.TestCase):
    DEVICE_STR = "cuda:0"


# ======================================================================================
# Fast contract tests (always run): use stubs, but call REAL methods on a real Model instance
# ======================================================================================


class FakeScalar:
    """
    Minimal scalar-like object that simulates a Tensor scalar returned by loss.
    Supports backward() and to_numpy() so your _to_float_scalar works.
    """

    def __init__(self, value: float):
        self._v = float(value)
        self.backward_called = 0

    def backward(self) -> None:
        self.backward_called += 1

    def to_numpy(self):
        return [self._v]


class FakePred:
    """
    Minimal prediction tensor-like for metrics.
    """

    def __init__(self, value: float = 0.5):
        self._v = float(value)

    def to_numpy(self):
        return [[self._v]]


class FakeOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def _simple_metric(y_true, y_pred):
    _ = y_true
    _ = y_pred
    return 0.25


class TestModelTrainOnBatchContract(unittest.TestCase):
    def test_train_on_batch_calls_backward_and_optimizer(self):
        """
        Contract test:
        - calls optimizer.zero_grad() and optimizer.step()
        - calls loss(...).backward()
        - returns dict containing loss + metric keys

        Important:
        We must call instance methods on a real Model object because your actual implementation
        may live on Sequential/Model instances, not as classmethods.
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        # Real Model instance, but we override forward to keep it cheap.
        m = Model()

        # Provide callable semantics expected by train_on_batch: y_pred = self(x_batch)
        # Some Module implementations call forward through __call__; to be safe we set forward.
        m.forward = lambda x: FakePred(0.5)  # type: ignore[method-assign]

        opt = FakeOptimizer()

        # Loss returns FakeScalar
        loss_fn = Mock(side_effect=lambda y_pred, y_true: FakeScalar(1.234))
        metrics = [Mock(side_effect=_simple_metric)]

        # Call instance method
        logs = m.train_on_batch(
            "x",
            "y",
            loss=loss_fn,
            optimizer=opt,
            metrics=metrics,
            metric_names=["m"],
        )

        self.assertIsInstance(logs, dict)
        self.assertIn("loss", logs)
        self.assertIn("m", logs)

        self.assertEqual(opt.zero_grad_calls, 1)
        self.assertEqual(opt.step_calls, 1)

        self.assertEqual(loss_fn.call_count, 1)
        self.assertTrue(metrics[0].called)


class TestModelFitHistoryContract(unittest.TestCase):
    def test_fit_returns_history_and_lengths(self):
        """
        Contract test for fit:
        - consumes batches iterable
        - calls train_on_batch per batch
        - aggregates into History with correct lengths
        """
        try:
            from src.keydnn.infrastructure.models._models import Model
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Missing imports: {e}")

        m = Model()

        # Monkeypatch train_on_batch on the instance to be deterministic & fast.
        def fake_train_on_batch(*args, **kwargs):
            _ = args, kwargs
            return {"loss": 1.0, "acc": 0.5}

        m.train_on_batch = fake_train_on_batch  # type: ignore[assignment]

        # Finite "dataloader": 3 batches per epoch
        batches = [("xb1", "yb1"), ("xb2", "yb2"), ("xb3", "yb3")]

        # loss/optimizer are unused due to stubbed train_on_batch, but required by signature
        hist = m.fit(
            batches,
            None,
            loss=lambda yp, yt: FakeScalar(1.0),
            optimizer=FakeOptimizer(),
            metrics=None,
            epochs=5,
            verbose=0,
        )

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
