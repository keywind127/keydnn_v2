"""
Utility functions exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for general-purpose
utilities implemented in the internal presentation layer
(`keydnn.presentation.apis.utils`). These helpers support common tasks
such as data preprocessing, serialization, and miscellaneous workflow
operations used throughout the library.

Users are encouraged to import utilities from this module rather than
relying on internal package paths.

Examples
--------
>>> from keydnn.utils import one_hot

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.utils import *
