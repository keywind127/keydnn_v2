"""
Tensor abstractions and utilities exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for tensor classes
and tensor-related utilities implemented in the internal presentation
layer (`keydnn.presentation.apis.tensors`). Tensors are the core data
structure used throughout KeyDNN for numerical computation and automatic
differentiation.

Users should import tensor-related functionality from this module rather
than relying on internal package paths.

Examples
--------
>>> from keydnn.tensors import Tensor

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.tensors import *
