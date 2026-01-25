"""
Determinism configuration utilities for KeyDNN (CPU backend).

This module defines the framework-level policy for controlling deterministic
behavior that is not governed by random number generators. It centralizes
configuration related to execution order, threading, and backend behavior
that may otherwise introduce run-to-run numerical variation.

Currently, this module focuses on CPU-side determinism, primarily by
configuring thread-related environment variables used by common numerical
libraries (e.g., OpenMP, MKL, OpenBLAS).

Notes
-----
- This module does not perform random seeding; RNG control is handled
  separately by the random utilities.
- CUDA backend determinism (cuDNN / cuBLAS) is intentionally out of scope
  until KeyDNN exposes native backend configuration APIs.
- Some environment variables may need to be set before importing NumPy or
  BLAS libraries to take full effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeterminismState:
    """
    Immutable record of the active determinism configuration.

    This data class captures the most recent determinism policy applied via
    :func:`set_deterministic`, allowing the framework to introspect and
    report reproducibility-related settings.

    Attributes
    ----------
    enabled : bool
        Whether deterministic behavior is enabled.
    cpu_threads : int or None
        The number of CPU threads configured for numerical libraries, or
        ``None`` if thread-related environment variables were not modified.
    """

    enabled: bool
    cpu_threads: int | None


_STATE: DeterminismState | None = None


def set_deterministic(
    enabled: bool, *, cpu_threads: int | None = 1
) -> DeterminismState:
    """
    Configure KeyDNN's determinism policy.

    This function centralizes reproducibility-related settings that are not
    strictly based on random number generation, such as thread-level execution
    determinism and backend configuration choices.

    When enabled, this function can restrict CPU parallelism by setting
    thread-control environment variables used by common numerical libraries.
    This helps reduce nondeterminism caused by varying execution order in
    parallel reductions.

    Parameters
    ----------
    enabled : bool
        If True, configure the runtime to prefer deterministic behavior.
    cpu_threads : int or None, optional
        Desired number of CPU threads for numerical libraries. When
        ``enabled=True``, the default is ``1`` for maximum reproducibility.
        If ``None``, thread-related environment variables are left unchanged.

    Returns
    -------
    DeterminismState
        The applied determinism configuration.

    Raises
    ------
    TypeError
        If `enabled` is not a boolean.
    ValueError
        If `cpu_threads` is not a positive integer or ``None``.

    Notes
    -----
    - Thread-related settings are applied via environment variables and may
      need to be set before importing NumPy or BLAS libraries for full effect.
    - This function does not guarantee bitwise-identical results across
      different hardware or library implementations.
    - CUDA backend determinism (cuDNN / cuBLAS) is intentionally not configured
      here and will be handled separately when native backend APIs are exposed.
    """
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")

    if cpu_threads is not None:
        if not isinstance(cpu_threads, int) or cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive int or None")

        # Common thread-control environment variables used by BLAS/OpenMP stacks.
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    global _STATE
    _STATE = DeterminismState(enabled=enabled, cpu_threads=cpu_threads)
    return _STATE


def get_deterministic() -> DeterminismState | None:
    """
    Return the most recent determinism configuration.

    Returns
    -------
    DeterminismState or None
        The last determinism configuration applied via
        :func:`set_deterministic`, or ``None`` if no configuration has been
        applied during the current process lifetime.
    """
    return _STATE
