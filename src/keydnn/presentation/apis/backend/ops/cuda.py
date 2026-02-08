"""
keydnn.presentation.apis.cuda

Public CUDA availability utilities for KeyDNN.

This module exposes lightweight, safe-to-import helpers in the presentation
layer for checking whether the CUDA native backend can be loaded in the
current Python process.

Design notes
------------
- This check is intentionally conservative: it reports availability only if
  the native CUDA backend loader successfully returns a loaded handle.
- Any exception is treated as "CUDA unavailable" to keep the API ergonomic
  for callers who want a simple boolean gate (e.g., tests, feature flags,
  conditional device selection).
- The underlying loader may depend on system configuration (e.g., CUDA/cuDNN
  runtime DLLs, PATH search, driver presence). This function does not attempt
  to diagnose or report the specific failure reason.
"""

from .....infrastructure.native_cuda.python._native_loader import (
    load_keydnn_cuda_native,
)


def cuda_available() -> bool:
    """
    Return whether KeyDNN's CUDA native backend appears to be available.

    This function attempts to load the KeyDNN CUDA native library via
    `load_keydnn_cuda_native()`. If the library loads successfully, CUDA is
    considered available; otherwise, the function returns `False`.

    The check is best used as a fast feature gate for optional CUDA execution.

    Returns
    -------
    bool
        `True` if the native CUDA backend can be loaded in the current process;
        `False` otherwise.

    Notes
    -----
    - All exceptions raised by the underlying loader are swallowed and mapped
      to `False`. If you need diagnostics, call `load_keydnn_cuda_native()`
      directly to surface the underlying error.
    """
    try:
        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


__all__ = [
    "cuda_available",
]
