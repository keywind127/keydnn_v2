"""
EarlyStopping callback.

This module provides a Keras-style `EarlyStopping` callback for `Model.fit()`-like
training loops. It monitors a metric in the epoch-end `logs` dict (e.g.,
"val_loss" or "accuracy") and requests early termination when the metric has not
improved for a configurable number of epochs (`patience`).

In-memory best-weight restore
-----------------------------
If `restore_best_weights=True`, the callback attempts to snapshot the model state
when a new best value is observed and restore that best state when stopping is
triggered.

This restore mechanism is best-effort and intentionally filesystem-free. It
relies on optional model methods:

- `model.to_json_payload() -> dict`
- `model.from_json_payload_(payload: dict) -> None`

If the model does not provide these methods, the callback gracefully degrades to
"stop-only" behavior.

Contracts
---------
- Expected `logs` shape: mapping of metric name to scalar-like values.
- Early termination: the callback sets `self.stop_training = True`. The training
  loop should stop after the current epoch (Keras-like behavior).

Design goals
------------
- Minimal coupling to the rest of the framework.
- Deterministic behavior based on explicit epoch indices and logs.
"""

from __future__ import annotations
from typing import Optional, Dict
import copy

from ._base import Callback
from ._base import _as_float, _is_better


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    This callback monitors a metric in `logs` (e.g., "val_loss") at epoch end.
    If the metric fails to improve for more than `patience` epochs, it sets
    `self.stop_training = True`, which the training loop should respect
    (typically stopping after the current epoch).  # NOTE: original text unchanged

    Restore best weights (in-memory)
    --------------------------------
    If `restore_best_weights=True`, EarlyStopping snapshots the model at the best
    epoch and restores it when stopping criteria are met.

    The snapshot/restore is performed entirely in memory using the optional
    model methods:
    - `model.to_json_payload() -> dict`
    - `model.from_json_payload_(payload: dict) -> None`

    If these methods are not available, `restore_best_weights=True` degrades
    gracefully to "stop-only" behavior.

    Parameters
    ----------
    monitor : str, optional
        Metric name to monitor, e.g. "val_loss" or "loss". Default "val_loss".
    mode : str, optional
        One of {"min", "max", "auto"}:
        - "min": lower is better (e.g., loss)
        - "max": higher is better (e.g., accuracy)
        - "auto": infer direction from `monitor` name ("acc"/"accuracy" -> max; else min)
    patience : int, optional
        Number of epochs with no improvement to tolerate before stopping.
        Default 0.
    min_delta : float, optional
        Minimum absolute change required to count as improvement. Default 0.0.
    restore_best_weights : bool, optional
        If True, attempt to restore model weights from the best epoch using the
        in-memory payload interface. Default False.

    Attributes
    ----------
    best : float
        Best metric value observed so far.
    best_epoch : Optional[int]
        Zero-based epoch index where the best value was observed.
    stopped_epoch : Optional[int]
        Zero-based epoch index where stopping was triggered.
    stop_training : bool
        Set to True when stopping criteria are met.

    Notes
    -----
    - Patience is evaluated using `wait > patience` (Keras-like in spirit, but
      exact off-by-one behavior depends on the training loop cadence).
    - Improvement logic is shared across callbacks via `_is_better`.
    """

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: str = "auto",
        patience: int = 0,
        min_delta: float = 0.0,
        restore_best_weights: bool = False,
    ) -> None:
        """
        Create a new EarlyStopping callback.

        Parameters
        ----------
        monitor : str, optional
            Metric name to monitor, e.g. "val_loss" or "loss".
        mode : str, optional
            One of {"min", "max", "auto"} determining optimization direction.
        patience : int, optional
            Number of epochs with no improvement to tolerate before stopping.
        min_delta : float, optional
            Minimum absolute change required to count as improvement.
        restore_best_weights : bool, optional
            Whether to restore model weights from the best epoch using an
            in-memory snapshot.
        """
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)

        self.wait = 0
        self.stopped_epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.stop_training = False

        self.best = float("inf")
        self._best_payload: Optional[dict] = None

    def _resolve_mode(self) -> str:
        """
        Resolve the effective optimization mode.

        Returns
        -------
        str
            Either "min" or "max", inferred from the configured `mode`
            and `monitor` name if necessary.

        Raises
        ------
        ValueError
            If the configured mode is unsupported.
        """
        if self.mode in ("min", "max"):
            return self.mode
        if self.mode == "auto":
            m = self.monitor.lower()
            if "acc" in m or "accuracy" in m:
                return "max"
            return "min"
        raise ValueError(f"Unsupported mode: {self.mode!r}")

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize internal state at the start of training.

        This resets the best metric value, wait counter, bookkeeping
        attributes, and any stored in-memory snapshot.

        Parameters
        ----------
        logs : dict[str, float], optional
            Initial training logs (unused).
        """
        mode = self._resolve_mode()
        if mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")
        self.wait = 0
        self.stopped_epoch = None
        self.best_epoch = None
        self.stop_training = False
        self._best_payload = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Check monitored metric at epoch end and update stopping state.

        If the monitored metric improves, internal best-tracking state is
        updated and (optionally) an in-memory model snapshot is taken.
        Otherwise, the patience counter is incremented and training may
        be stopped once patience is exceeded.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Metrics logged for the completed epoch.
        """
        logs = logs or {}
        current = _as_float(logs.get(self.monitor, float("nan")))
        mode = self._resolve_mode()

        if _is_better(current, self.best, mode, self.min_delta):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch

            if self.restore_best_weights:
                model = getattr(self, "model", None)
                if model is not None:
                    to_config = getattr(model, "save_json", None)
                    to_payload = getattr(model, "to_json_payload", None)
                    if callable(to_payload):
                        self._best_payload = copy.deepcopy(to_payload())

            return

        self.wait += 1
        if self.wait > self.patience:
            self.stopped_epoch = epoch
            self.stop_training = True

            if self.restore_best_weights and self._best_payload is not None:
                model = getattr(self, "model", None)
                from_payload = getattr(model, "from_json_payload_", None)
                if callable(from_payload):
                    from_payload(self._best_payload)
