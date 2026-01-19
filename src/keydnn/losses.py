"""
Loss functions exposed by the KeyDNN public API.

This module provides a stable, user-facing namespace for loss function
modules implemented in the internal presentation layer
(`keydnn.presentation.apis.losses`). Loss functions define the objective
optimized during training and are used to quantify model prediction
errors.

Users should import loss functions from this module rather than relying
on internal package paths.

Examples
--------
>>> from keydnn.losses import mse_loss

Notes
-----
This module is a thin re-export layer and contains no implementation
logic of its own.
"""

from __future__ import annotations
from .presentation.apis.losses import *
