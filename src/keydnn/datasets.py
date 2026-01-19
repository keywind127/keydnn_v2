"""
Dataset loading and preprocessing utilities exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for dataset helpers
implemented in the internal presentation layer
(`keydnn.presentation.apis.datasets`). These utilities typically include
dataset loaders, download helpers, and lightweight preprocessing
routines intended for experimentation and training workflows.

Users are encouraged to import dataset utilities from this module rather
than relying on internal package paths.

Examples
--------
>>> from keydnn.datasets import load_mnist

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.datasets import *
