"""
Activation functions exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for activation
functions implemented in the internal presentation layer
(`keydnn.presentation.apis.activations`).

Users are encouraged to import activation functions from this module
rather than relying on internal package paths. This abstraction allows
the internal architecture to evolve without breaking user code.

Examples
--------
>>> from keydnn.activations import ReLU, Softmax

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.activations import *
