"""
Optimization algorithms exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for optimizer
implementations defined in the internal presentation layer
(`keydnn.presentation.apis.optimizers`). Optimizers are responsible for
updating model parameters during training based on computed gradients.

Users are encouraged to import optimizers from this module rather than
relying on internal package paths.

Examples
--------
>>> from keydnn.optimizers import SGD, Adam

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.optimizers import *
