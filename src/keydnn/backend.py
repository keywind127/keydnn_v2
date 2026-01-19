"""
Backend utilities and device abstractions exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for backend-related
functionality implemented in the internal presentation layer
(`keydnn.presentation.apis.backend`). These utilities typically include
device management, execution backends (e.g., CPU, CUDA), and low-level
runtime helpers required by higher-level components.

Users should import backend functionality from this module rather than
relying on internal package paths.

Examples
--------
>>> from keydnn.backend import cuda_available

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.backend import *
