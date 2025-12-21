"""
KeyDNN native shared library loader and deployment policy.

This module centralizes all logic for resolving and loading KeyDNN's
native C++ backend via `ctypes`, including platform-specific filename
conventions, OpenMP vs non-OpenMP variant selection, and Windows DLL
dependency handling.

Design goals
------------
- Provide a **stable, cross-platform loading API** for native kernels.
- Prefer **OpenMP-enabled libraries by default** to maximize CPU
  parallelism, while safely falling back to single-threaded variants
  when OpenMP runtimes are unavailable.
- Keep Python ↔ C boundary crossings minimal and predictable.
- Make native acceleration an **optional optimization layer** that
  never compromises correctness or portability.

Resolution policy
-----------------
Unless an explicit `lib_path` is provided, the loader searches for native
libraries located next to this module using the following priority order:

1. OpenMP-enabled variant (`*_omp`)
2. Single-threaded native variant (`*_noomp`)
3. Backward-compatible default name (`*default`)

This policy ensures that production deployments automatically benefit
from OpenMP parallelism when available, while development and test
environments remain robust to missing runtime dependencies.

Windows-specific considerations
-------------------------------
On Windows (Python 3.8+), dependent DLL discovery is restricted by default.
This module explicitly registers additional DLL search paths using
`os.add_dll_directory(...)`, including:

- The directory containing the target KeyDNN native library
- The MinGW-w64 runtime directory (`KEYDNN_MINGW_BIN`, if defined)

Directory registration handles are retained for the lifetime of the
loaded library to prevent premature unloading.

Scope
-----
This module is intentionally narrow in scope:
- It does *not* expose kernel symbols directly.
- It does *not* decide which kernels are called.
- It only guarantees that the correct native library is located and
  loaded consistently across platforms.

Higher-level dispatch logic (e.g., dtype routing, kernel selection,
fallback behavior) lives in the surrounding `conv2d_ctypes` and operator
dispatch layers.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional


def _variant_lib_name(variant: str) -> str:
    """
    Return the platform-specific filename for a given KeyDNN native library variant.

    Parameters
    ----------
    variant : str
        One of:
        - "omp"     : OpenMP-enabled native library
        - "noomp"   : single-threaded native library (no OpenMP)
        - "default" : back-compat library name (historically the only native lib)

    Returns
    -------
    str
        The filename (not a full path) for the requested variant on the current OS.

    Notes
    -----
    - Windows:  *.dll
    - macOS:    *.dylib
    - Linux:    *.so
    """
    v = variant.lower()

    if sys.platform.startswith("win"):
        if v == "omp":
            return "keydnn_native_omp.dll"
        if v == "noomp":
            return "keydnn_native_noomp.dll"
        if v == "default":
            return "keydnn_native.dll"
        raise ValueError(f"Unknown variant: {variant!r}")

    if sys.platform == "darwin":
        if v == "omp":
            return "libkeydnn_native_omp.dylib"
        if v == "noomp":
            return "libkeydnn_native_noomp.dylib"
        if v == "default":
            return "libkeydnn_native.dylib"
        raise ValueError(f"Unknown variant: {variant!r}")

    # Linux / others
    if v == "omp":
        return "libkeydnn_native_omp.so"
    if v == "noomp":
        return "libkeydnn_native_noomp.so"
    if v == "default":
        return "libkeydnn_native.so"
    raise ValueError(f"Unknown variant: {variant!r}")


def load_keydnn_native(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """
    Load the KeyDNN native shared library via ctypes.

    Resolution policy
    -----------------
    1) If `lib_path` is provided, it is treated as the exact library to load.
    2) Otherwise, KeyDNN searches next to this Python module, preferring:
       - OpenMP variant ("omp")
       - Non-OpenMP variant ("noomp")
       - Back-compat default name ("default")

    Parameters
    ----------
    lib_path : Optional[str]
        Absolute or relative path to a specific native library file.
        If provided, this path always wins and no variant search occurs.

    Returns
    -------
    ctypes.CDLL
        A loaded ctypes handle to the native library.

    Raises
    ------
    FileNotFoundError
        If `lib_path` is provided but the file does not exist.
    OSError
        If none of the candidate libraries can be loaded.
    """
    base_dir = Path(__file__).resolve().parent

    # Explicit path always wins
    if lib_path is not None:
        return _load_cdll_with_windows_dirs(Path(lib_path).resolve())

    # Default deployment policy: prefer OMP, then NOOMP, then DEFAULT
    candidates = [
        base_dir / _variant_lib_name("omp"),
        base_dir / _variant_lib_name("noomp"),
        base_dir / _variant_lib_name("default"),
    ]

    errors: list[str] = []
    for p in candidates:
        if not p.exists():
            errors.append(f"- {p} (missing)")
            continue
        try:
            return _load_cdll_with_windows_dirs(p)
        except Exception as e:
            errors.append(f"- {p} (failed to load: {e})")

    raise OSError(
        "Failed to load any KeyDNN native library. Tried:\n" + "\n".join(errors)
    )


def _load_cdll_with_windows_dirs(dll_path: Path) -> ctypes.CDLL:
    """
    Load a native library with extra Windows DLL search path registration.

    On Windows (Python 3.8+), dependent DLL lookup is restricted unless you:
    - place dependencies next to the target DLL, or
    - add directories via `os.add_dll_directory(...)`.

    This helper registers:
    1) The directory containing the target DLL
    2) KEYDNN_MINGW_BIN (if set), to find MinGW runtime DLLs

    Parameters
    ----------
    dll_path : Path
        Absolute path to the target native library.

    Returns
    -------
    ctypes.CDLL
        Loaded library handle. The handle retains DLL-directory registration
        handles to keep them alive.

    Raises
    ------
    FileNotFoundError
        If `dll_path` does not exist.
    OSError
        If the DLL exists but cannot be loaded (often due to missing dependencies).
    """
    if not dll_path.exists():
        raise FileNotFoundError(f"Native library not found: {dll_path}")

    add_dirs: list[str] = []
    handles = []

    if sys.platform.startswith("win"):
        add_dirs.append(str(dll_path.parent))

        mingw_bin = os.environ.get("KEYDNN_MINGW_BIN", "")
        if mingw_bin:
            add_dirs.append(mingw_bin)

        if hasattr(os, "add_dll_directory"):
            for d in add_dirs:
                if d and Path(d).exists():
                    handles.append(os.add_dll_directory(d))

    lib = ctypes.CDLL(str(dll_path))
    # Keep the directory handles alive for the lifetime of the library handle.
    setattr(lib, "_keydnn_dll_dir_handles", handles)
    return lib
