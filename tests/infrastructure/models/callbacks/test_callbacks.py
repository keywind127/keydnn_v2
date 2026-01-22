import os
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.models._models import Model
from src.keydnn.infrastructure.utils._preprocessing import numpy_to_tensor

from src.keydnn.infrastructure.models.callbacks._base import Callback, CallbackList
from src.keydnn.infrastructure.models.callbacks._model_checkpoint import ModelCheckpoint
from src.keydnn.infrastructure.models.callbacks._early_stopping import EarlyStopping


class _DummyScalar:
    """
    A scalar-like object that mimics a tensor loss output enough for Model.train_on_batch.

    - `.item()` returns a Python float
    - `.backward()` exists (no-op)
    """

    def __init__(self, v: float) -> None:
        self._v = float(v)

    def item(self) -> float:
        return float(self._v)

    def backward(self) -> None:
        return


class _SequenceLoss:
    """
    Loss callable that yields a pre-defined sequence of scalar values.

    This allows deterministic "improving then plateau" behavior without relying on autograd.
    """

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(map(float, values))
        self._i = 0

    def __call__(self, y_pred: Any, y_true: Any) -> _DummyScalar:
        if self._i >= len(self._values):
            # If tests accidentally consume more steps than expected, fail loudly.
            raise RuntimeError(
                "Loss sequence exhausted; increase test sequence length."
            )
        v = self._values[self._i]
        self._i += 1
        return _DummyScalar(v)


class _NoOpOptimizer:
    def zero_grad(self) -> None:
        return

    def step(self) -> None:
        return


class _PassthroughModel(Model):
    """
    Minimal model to exercise Model.fit + callbacks without real parameter updates.
    """

    def forward(self, x: Any) -> Any:
        return x


class _RestorableWeightModel(_PassthroughModel):
    """
    A model with a simple in-memory "weight" that can be snapshotted/restored
    via to_json_payload/from_json_payload_.

    This is used to test EarlyStopping(restore_best_weights=True) independent of
    full module serialization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w: float = 0.0

    def to_json_payload(self) -> Dict[str, Any]:
        # Mimic your checkpoint payload shape, but keep it minimal for the test.
        return {
            "format": "keydnn.json.ckpt.v1",
            "arch": {"dummy": True},
            "state": {"w": float(self.w)},
        }

    def from_json_payload_(self, payload: Dict[str, Any]) -> None:
        if payload.get("format") != "keydnn.json.ckpt.v1":
            raise ValueError("bad format")
        self.w = float(payload["state"]["w"])


class _MutateWeightEachEpoch(Callback):
    """
    Mutates model.w at epoch end so that "best weights" corresponds to best epoch.
    """

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(map(float, values))

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        # Set weight to a known per-epoch value.
        self.model.w = float(self._values[epoch])


class TestCallbackList(unittest.TestCase):
    def test_callback_list_stop_training_propagates(self) -> None:
        class _Stopper(Callback):
            def on_epoch_end(
                self, epoch: int, logs: Optional[Dict[str, float]] = None
            ) -> None:
                self.stop_training = True  # type: ignore[attr-defined]

        cb = CallbackList([_Stopper()])
        cb.set_model(object())
        cb.on_epoch_end(0, logs={"loss": 1.0})
        self.assertTrue(cb.stop_training)


class TestModelCheckpoint(unittest.TestCase):
    def test_model_checkpoint_saves_best_only(self) -> None:
        model = _PassthroughModel()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # We'll patch model.save_json to write a tiny JSON marker file.
            saved_paths: list[Path] = []

            def _save_json(p: str | Path) -> None:
                p = Path(p)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"ok": True}), encoding="utf-8")
                saved_paths.append(p)

            model.save_json = _save_json  # type: ignore[method-assign]

            cb = ModelCheckpoint(
                td_path / "ckpt_epoch{epoch:03d}_valloss{val_loss:.6f}.json",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=0,
            )
            cb.set_model(model)

            cb.on_train_begin()

            # val_loss improves at epoch 0 -> 1, then worsens.
            logs_by_epoch = [
                {"val_loss": 1.0},
                {"val_loss": 0.9},
                {"val_loss": 0.91},
                {"val_loss": 0.92},
            ]

            for e, logs in enumerate(logs_by_epoch):
                cb.on_epoch_end(e, logs=logs)

            # Expect saves on epoch 0 and epoch 1 only.
            self.assertEqual(len(saved_paths), 2)
            for p in saved_paths:
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)

            # Ensure formatting happened (epoch is 1-based in the callback implementation
            # in my earlier patch; if yours is 0-based, adjust this assertion).
            # We'll just sanity-check the filenames contain "epoch" and "valloss".
            names = [p.name for p in saved_paths]
            self.assertTrue(any("epoch" in n and "valloss" in n for n in names))


class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_stops_after_patience(self) -> None:
        es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=1,
            min_delta=0.0,
            restore_best_weights=False,
        )

        # Simulate training:
        es.on_train_begin()

        logs_by_epoch = [
            {"val_loss": 1.0},  # best
            {"val_loss": 0.9},  # best
            {"val_loss": 0.91},  # no improve (wait=1)
            {
                "val_loss": 0.92
            },  # no improve (wait=2) => stop if condition is wait > patience
        ]
        for e, logs in enumerate(logs_by_epoch):
            es.on_epoch_end(e, logs=logs)
            if getattr(es, "stop_training", False):
                break

        self.assertTrue(getattr(es, "stop_training", False))
        self.assertEqual(es.best, 0.9)
        self.assertEqual(es.best_epoch, 1)

    def test_early_stopping_restores_best_weights_in_memory(self) -> None:
        model = _RestorableWeightModel()
        # Each epoch, we mutate w to a known value. Best epoch should restore that.
        mutator = _MutateWeightEachEpoch([10.0, 20.0, 30.0, 40.0])

        es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=1,
            min_delta=0.0,
            restore_best_weights=True,
        )

        # Wire callbacks like fit() does.
        cb_list = CallbackList([mutator, es])
        cb_list.set_model(model)

        cb_list.on_train_begin(logs={})

        # val_loss best at epoch 1 (0.9), then degrades twice -> stop.
        logs_by_epoch = [
            {"val_loss": 1.0},
            {"val_loss": 0.9},  # best epoch (w=20)
            {"val_loss": 0.91},
            {"val_loss": 0.92},  # triggers stop
        ]

        for e, logs in enumerate(logs_by_epoch):
            cb_list.on_epoch_begin(e, logs={})
            cb_list.on_epoch_end(e, logs=logs)
            if cb_list.stop_training:
                break

        # Mutator would have set w to 40.0 at epoch 3,
        # but EarlyStopping should restore best weights (epoch 1 => w=20.0).
        self.assertTrue(cb_list.stop_training)
        self.assertAlmostEqual(model.w, 20.0, places=7)


class TestFitWithCallbacks(unittest.TestCase):
    def test_fit_runs_callbacks_and_early_stops(self) -> None:
        model = _PassthroughModel()
        opt = _NoOpOptimizer()

        # One batch per epoch (batch_size == len(x)), and one validation batch per epoch.
        x = np.zeros((8, 3), dtype=np.float32)
        y = np.zeros((8, 3), dtype=np.float32)
        xv = np.zeros((8, 3), dtype=np.float32)
        yv = np.zeros((8, 3), dtype=np.float32)

        device = Device("cpu")

        x = numpy_to_tensor(x, device=device)
        y = numpy_to_tensor(y, device=device)
        xv = numpy_to_tensor(xv, device=device)
        yv = numpy_to_tensor(yv, device=device)

        # Loss called twice per epoch (train + val) due to validation_data.
        # val_loss improves epoch 0->1 then worsens twice => stop with patience=1.
        # We'll keep train loss arbitrary.
        loss_values = [
            2.0,
            1.0,  # epoch0: train, val
            2.0,
            0.9,  # epoch1: train, val (best)
            2.0,
            0.91,  # epoch2: train, val (no improve)
            2.0,
            0.92,  # epoch3: train, val (no improve => stop)
            2.0,
            0.93,  # extra (should not be consumed)
        ]
        loss = _SequenceLoss(loss_values)

        es = EarlyStopping(
            monitor="val_loss", mode="min", patience=1, restore_best_weights=False
        )

        model.build(x[:1])

        hist = model.fit(
            x,
            y,
            loss=loss,
            optimizer=opt,
            batch_size=len(x),
            epochs=10,
            shuffle=False,
            verbose=0,
            validation_data=(xv, yv),
            callbacks=[es],
        )

        # Expect stop at epoch index 3 (4 epochs recorded) given the "wait > patience" rule.
        self.assertLess(len(hist.epoch), 10)
        self.assertEqual(len(hist.epoch), 4)

        self.assertIn("loss", hist.history)
        self.assertIn("val_loss", hist.history)
        self.assertEqual(len(hist.history["val_loss"]), len(hist.epoch))

    def test_fit_with_model_checkpoint_saves_only_on_improve(self) -> None:
        model = _PassthroughModel()
        opt = _NoOpOptimizer()

        x = np.zeros((8, 3), dtype=np.float32)
        y = np.zeros((8, 3), dtype=np.float32)
        xv = np.zeros((8, 3), dtype=np.float32)
        yv = np.zeros((8, 3), dtype=np.float32)

        device = Device("cpu")

        x = numpy_to_tensor(x, device=device)
        y = numpy_to_tensor(y, device=device)
        xv = numpy_to_tensor(xv, device=device)
        yv = numpy_to_tensor(yv, device=device)

        # Train+val per epoch.
        loss_values = [
            2.0,
            1.0,  # epoch0 val=1.0 (save)
            2.0,
            0.9,  # epoch1 val=0.9 (save)
            2.0,
            0.91,  # epoch2 val=0.91 (skip)
            2.0,
            0.92,  # epoch3 val=0.92 (skip)
        ]
        loss = _SequenceLoss(loss_values)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            saved: list[Path] = []

            def _save_json(p: str | Path) -> None:
                p = Path(p)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{"ok": true}', encoding="utf-8")
                saved.append(p)

            model.save_json = _save_json  # type: ignore[method-assign]

            ckpt = ModelCheckpoint(
                td_path / "ckpt_epoch{epoch:03d}_valloss{val_loss:.6f}.json",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=0,
            )

            model.build(x[:1])

            _ = model.fit(
                x,
                y,
                loss=loss,
                optimizer=opt,
                batch_size=len(x),
                epochs=4,
                shuffle=False,
                verbose=0,
                validation_data=(xv, yv),
                callbacks=[ckpt],
            )

            # Expect 2 checkpoint saves (epoch0 and epoch1).
            self.assertEqual(len(saved), 2)
            self.assertTrue(all(p.exists() for p in saved))
            self.assertTrue(
                any("valloss0.900000" in p.name or "0.900000" in p.name for p in saved)
            )


if __name__ == "__main__":
    unittest.main()
