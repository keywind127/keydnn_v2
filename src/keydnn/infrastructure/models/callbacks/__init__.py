"""
Training callback utilities.

This package provides Keras-style callbacks for `Model.fit()`.

Public exports
--------------
- `Callback`, `CallbackList`: callback base primitives used by training loops.
- `ModelCheckpoint`: filesystem checkpointing via `Model.save_json(path)`.
- `EarlyStopping`: early termination based on a monitored metric, with optional
  in-memory best-weight restore via `to_json_payload()` / `from_json_payload_()`.

Callbacks are intentionally lightweight and framework-agnostic: they operate on
epoch/batch indices and a `logs` mapping of metric names to Python floats.
"""

from ._model_checkpoint import ModelCheckpoint
from ._early_stopping import EarlyStopping
from ._base import Callback, CallbackList

__all__ = [
    ModelCheckpoint.__name__,
    EarlyStopping.__name__,
    CallbackList.__name__,
    Callback.__name__,
]
