"""
High-level training API exports.

This module defines the public-facing training-related symbols exposed by the
infrastructure layer. It aggregates commonly used model abstractions and
callback utilities so they can be imported from a single, stable entry point.

Exports
-------
Models
~~~~~~
- `Model`: Base model abstraction.
- `Sequential`: Simple linear stack of layers.
- `History`: Training history container produced by `Model.fit()`.

Callbacks
~~~~~~~~~
- `Callback`: Base class for training callbacks.
- `CallbackList`: Callback dispatcher used by training loops.
- `ModelCheckpoint`: Filesystem-based model checkpointing callback.
- `EarlyStopping`: Metric-based early termination callback.

Design intent
-------------
- Provide a clean, discoverable API surface for users.
- Avoid exposing internal helper modules directly.
- Keep import paths stable even if internal organization changes.
"""

from ._sequential import Sequential
from ._history import History
from ._models import Model
from .callbacks import (
    Callback,
    CallbackList,
    ModelCheckpoint,
    EarlyStopping,
)
