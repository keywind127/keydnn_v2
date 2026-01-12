"""
Keras-style training callback primitives.

This module defines the minimal callback protocol used by `Model.fit()` and
small helper utilities shared across callback implementations.

What this module provides
-------------------------
- Scalar helpers:
  - `_as_float(x)`: best-effort conversion to a Python float for log values.
  - `_is_better(current, best, mode, min_delta)`: improvement predicate used
    by callbacks such as EarlyStopping / ModelCheckpoint.
- `Callback`: a lightweight base class with Keras-inspired hook names.
- `CallbackList`: a dispatcher that forwards events to multiple callbacks and
  exposes a shared `stop_training` flag.

Logs contract
-------------
The training loop passes `logs` as a mapping of metric names to Python floats,
for example:
    {"loss": 0.123, "accuracy": 0.98, "val_loss": 0.130}

Callbacks may read and/or mutate `logs`. The training loop is responsible for
constructing `logs` and for deciding which keys exist.

Stopping contract
-----------------
A callback may request early termination by setting an attribute named
`stop_training` to True. `CallbackList` detects this after each batch/epoch
and sets its own `stop_training` flag. The training loop should then stop
after the current epoch (Keras-like behavior).

Design goals
------------
- Minimal and framework-agnostic: no dependency on tensors, autograd, devices.
- Deterministic: callbacks operate on explicit epoch/batch indices.
- Easy integration: hooks are optional and default to no-ops.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import math


Number = Union[int, float]


def _as_float(x: Any) -> float:
    """
    Convert a scalar-like value to a Python float.

    This helper is intentionally defensive. While `Model.fit()` generally
    supplies `logs` as floats already, callbacks may be used in contexts
    where values come from other scalar-like types.

    Parameters
    ----------
    x : Any
        Scalar-like value.

    Returns
    -------
    float
        Converted float, or NaN if conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
    """
    Determine whether a new metric value is an improvement over the best-so-far.

    Parameters
    ----------
    current : float
        Newly observed metric value.
    best : float
        Best value seen so far.
    mode : str
        Optimization direction:
        - "min": improvement means strictly smaller than (best - min_delta)
        - "max": improvement means strictly larger than (best + min_delta)
    min_delta : float
        Minimum absolute change required to count as improvement.

    Returns
    -------
    bool
        True if the new value is considered better; otherwise False.

    Notes
    -----
    - NaN is never considered an improvement.
    - This predicate is shared by multiple callbacks to keep behavior consistent.
    """
    if math.isnan(current):
        return False

    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)

    raise ValueError(f"Unsupported mode: {mode!r} (expected 'min' or 'max')")


class Callback:
    """
    Base class for Keras-style training callbacks.

    Subclasses may override any subset of the hook methods below. All hook
    methods are optional and default to no-ops.

    Supported hooks
    ---------------
    - `set_model(model)`: called once before training begins.
    - `on_train_begin(logs)`, `on_train_end(logs)`
    - `on_epoch_begin(epoch, logs)`, `on_epoch_end(epoch, logs)`
    - `on_batch_begin(batch, logs)`, `on_batch_end(batch, logs)`

    Attributes
    ----------
    model : Any
        The model instance being trained. Set by `set_model(model)`.

    Notes
    -----
    - `logs` is typically a dict of Python floats, but callers may pass any
      mapping-like structure as long as metric values are scalar-like.
    - To request early termination, a callback may set `self.stop_training = True`.
    """

    def set_model(self, model: Any) -> None:
        """
        Attach the model being trained to this callback.

        This method is called once before training begins and allows callbacks
        to store a reference to the model for later use.

        Parameters
        ----------
        model : Any
            The model instance that will be trained.
        """
        self.model = model  # type: ignore[attr-defined]

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Hook called at the very beginning of training.

        Parameters
        ----------
        logs : dict[str, float], optional
            Initial training logs, typically empty or containing global metrics.
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Hook called at the very end of training.

        Parameters
        ----------
        logs : dict[str, float], optional
            Final training logs.
        """
        pass

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Hook called at the beginning of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Logs available at the start of the epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Hook called at the end of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Logs aggregated over the epoch.
        """
        pass

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Hook called at the beginning of a batch.

        Parameters
        ----------
        batch : int
            Zero-based batch index within the current epoch.
        logs : dict[str, float], optional
            Logs available at the start of the batch.
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Hook called at the end of a batch.

        Parameters
        ----------
        batch : int
            Zero-based batch index within the current epoch.
        logs : dict[str, float], optional
            Logs aggregated over the batch.
        """
        pass


class CallbackList:
    """
    Dispatch training events to a sequence of callbacks.

    `CallbackList` is used by training loops (e.g., `Model.fit()`) to forward
    events to multiple callbacks while tracking a shared `stop_training` flag.

    Attributes
    ----------
    callbacks : List[Callback]
        Registered callbacks to receive events.
    stop_training : bool
        Set to True if any callback requests termination (by setting an attribute
        `stop_training` to True). The training loop should stop after the
        current epoch completes.
    """

    def __init__(self, callbacks: Optional[Sequence[Callback]] = None) -> None:
        """
        Create a new callback dispatcher.

        Parameters
        ----------
        callbacks : sequence of Callback, optional
            Initial callbacks to register.
        """
        self.callbacks: List[Callback] = list(callbacks or [])
        self.stop_training: bool = False

    def set_model(self, model: Any) -> None:
        """
        Attach the model to all registered callbacks.

        Parameters
        ----------
        model : Any
            The model instance being trained.
        """
        for cb in self.callbacks:
            cb.set_model(model)

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Forward the training-begin event to all callbacks.

        Parameters
        ----------
        logs : dict[str, float], optional
            Initial training logs.
        """
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Forward the training-end event to all callbacks.

        Parameters
        ----------
        logs : dict[str, float], optional
            Final training logs.
        """
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Forward the epoch-begin event to all callbacks.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Logs available at the start of the epoch.
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Forward the epoch-end event to all callbacks and update stop flags.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Logs aggregated over the epoch.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
            if getattr(cb, "stop_training", False):
                self.stop_training = True

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Forward the batch-begin event to all callbacks.

        Parameters
        ----------
        batch : int
            Zero-based batch index.
        logs : dict[str, float], optional
            Logs available at the start of the batch.
        """
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Forward the batch-end event to all callbacks and update stop flags.

        Parameters
        ----------
        batch : int
            Zero-based batch index.
        logs : dict[str, float], optional
            Logs aggregated over the batch.
        """
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)
            if getattr(cb, "stop_training", False):
                self.stop_training = True
