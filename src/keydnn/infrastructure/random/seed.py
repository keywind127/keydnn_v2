"""
Random number generation utilities for KeyDNN (CPU / NumPy backend).

This module provides a minimal, framework-level interface for controlling
randomness in KeyDNN's CPU execution path. It centralizes seeding of all
Python- and NumPy-based random number generators to ensure reproducible
behavior across:

- parameter initialization
- NumPy-based operators and utilities
- any Python-side randomness used by the framework

Notes
-----
- This module intentionally targets only Python and NumPy RNG sources.
- GPU / CUDA randomness is handled separately in the infrastructure CUDA
  subsystem and is not affected by this module.
- With no multiprocessing data loader, seeding here is sufficient to achieve
  deterministic behavior for typical CPU-only training and inference.
"""

from __future__ import annotations

import random
import numpy as np

_GLOBAL_SEED: int | None = None


def seed(seed: int) -> None:
    """
    Seed RNG sources used by KeyDNN's CPU / NumPy execution path.

    This function seeds all random number generators that affect CPU-side
    execution, including:

    - Python's standard-library `random` module
    - NumPy's global random number generator (`np.random`)

    Parameters
    ----------
    seed : int
        Global seed value used to initialize all supported RNG sources.

    Raises
    ------
    TypeError
        If `seed` is not an integer.

    Notes
    -----
    - Calling this function multiple times with the same seed is idempotent
      with respect to subsequent random number generation.
    - This function should typically be called once at the start of a script,
      test, or experiment to ensure reproducibility.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)

    global _GLOBAL_SEED
    _GLOBAL_SEED = seed


def get_seed() -> int | None:
    """
    Return the last seed set via :func:`seed`.

    Returns
    -------
    int or None
        The most recent seed value passed to :func:`seed`, or ``None`` if
        no seed has been set during the current process lifetime.
    """
    return _GLOBAL_SEED
