"""
Training callbacks exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for callback
interfaces and implementations defined in the internal presentation
layer (`keydnn.presentation.apis.callbacks`).

Callbacks allow users to hook into various stages of the training
lifecycle (e.g., epoch begin/end, batch updates) to implement custom
logging, monitoring, checkpointing, or control logic.

Examples
--------
>>> from keydnn.callbacks import Callback

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.callbacks import *
