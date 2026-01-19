"""
Model abstractions and containers exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for model classes
implemented in the internal presentation layer
(`keydnn.presentation.apis.models`). These abstractions define how layers
are composed, trained, evaluated, and serialized within KeyDNN.

Users are encouraged to import model classes from this module rather than
relying on internal package paths.

Examples
--------
>>> from keydnn.models import Sequential

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.models import *
