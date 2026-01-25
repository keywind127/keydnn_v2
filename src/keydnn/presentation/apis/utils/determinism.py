"""
Public determinism configuration API for KeyDNN.

This module exposes a stable entry point for configuring determinism-related
runtime policies (e.g., CPU threading). CUDA backend determinism is handled
separately when the CUDA configuration surface is available.
"""

from ....infrastructure.random._determinism import (
    set_deterministic,
    get_deterministic,
)

__all__ = [
    "set_deterministic",
    "get_deterministic",
]
