"""
Neural network layer modules exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for layer classes
implemented in the internal presentation layer
(`keydnn.presentation.apis.layers`). Layers are the fundamental building
blocks of KeyDNN models and include common components such as convolution,
normalization, pooling, and fully connected layers.

Users are encouraged to import layer classes from this module rather than
relying on internal package paths, ensuring long-term API stability.

Examples
--------
>>> from keydnn.layers import Dense, Conv2D

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.layers import *
