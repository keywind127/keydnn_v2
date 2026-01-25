"""
Windows DLL loader for the KeyDNN CUDA native backend.

This module provides a cached loader for the KeyDNN CUDA native DLL on Windows.
It resolves the expected `KeyDNNV2CudaNative.dll` path within the repository,
configures DLL dependency resolution for the current Python process, and returns
a reusable `ctypes.CDLL` handle.

Key behaviors
-------------
- Cached singleton: :func:`load_keydnn_cuda_native` is decorated with
  ``lru_cache(maxsize=1)`` so the DLL is loaded at most once per process.
- Dependency resolution: attempts to register dependent DLL directories via
  :func:`os.add_dll_directory` (preferred on modern Windows / Python).
- Robust fallback: if :func:`os.add_dll_directory` fails with WinError 206
  ("The filename or extension is too long"), falls back to prepending the
  directory to the process-local ``PATH``.
- Environment variable support:
  - ``CUDA_PATH``: if set, ``<CUDA_PATH>\\bin`` is added to the DLL search path.
  - ``CUDNN_PATH``: if set, ``<CUDNN_PATH>\\bin`` is added to the DLL search path.
- Conservative fallback search (Windows-only): if ``CUDA_PATH`` is not set,
  the loader attempts to locate CUDA under the standard installation root:
  ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin`` (newest first).
- Actionable failures: if loading fails due to missing dependencies, a rich
  :class:`OSError` is raised with guidance on what to configure.

Notes
-----
- This module is Windows-specific. It relies on Windows DLL search behavior and
  :func:`os.add_dll_directory`.
- Thread-safety: the cached loader is intended to be called during startup, not
  concurrently from multiple threads.
- Environment variable changes apply only to the current process and must be
  set before Python starts for global/system persistence.
"""

from __future__ import annotations

from functools import lru_cache


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_keydnn_cuda_native():
    """
    Load and cache the KeyDNN CUDA native DLL (Windows).

    This function locates the compiled `KeyDNNV2CudaNative.dll` within the
    repository, configures DLL dependency resolution for the current process,
    and returns a `ctypes.CDLL` handle.

    Environment variables
    ---------------------
    CUDA_PATH : str, optional
        If set, ``<CUDA_PATH>\\bin`` is added to the DLL search path to help
        locate CUDA runtime dependencies.
    CUDNN_PATH : str, optional
        If set, ``<CUDNN_PATH>\\bin`` is added to the DLL search path to help
        locate cuDNN dependencies.

    Returns
    -------
    ctypes.CDLL
        Loaded DLL handle for the KeyDNN CUDA native backend.

    Raises
    ------
    FileNotFoundError
        If the expected KeyDNN native DLL does not exist at the resolved path.
    OSError
        If dependency resolution fails, or if the DLL fails to load.

    Notes
    -----
    - Uses ``lru_cache(maxsize=1)`` so the DLL is loaded at most once per process.
    - Uses `_add_dll_dir_or_path` for CUDA/cuDNN bins and the DLL's own directory
      to ensure transitive dependencies resolve correctly.
    """
    import ctypes
    import os
    import platform
    from pathlib import Path

    if platform.system().lower() != "windows":
        raise OSError("KeyDNN CUDA native backend loader is supported on Windows only.")

    # ------------------------------------------------------------------
    # Resolve KeyDNN native DLL path (repo-relative)
    # ------------------------------------------------------------------
    p = (
        Path(__file__).resolve().parents[2]
        / "native_cuda"
        / "keydnn_v2_cuda_native"
        / "x64"
        / "Release"
        / "KeyDNNV2CudaNative.dll"
    ).resolve()

    if not p.exists():
        raise FileNotFoundError(f"KeyDNN CUDA native DLL not found at: {p}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_dll_dir_or_path(dir_path: str) -> bool:
        """
        Add a directory for DLL dependency resolution.

        This helper uses `os.add_dll_directory(dir_path)` when available, which
        is the recommended approach on modern Windows and Python versions.

        Some environments can raise WinError 206 ("The filename or extension
        is too long") when calling `os.add_dll_directory`. In that specific case,
        this helper falls back to process-local PATH modification by prepending
        `dir_path` to `os.environ["PATH"]`.

        Parameters
        ----------
        dir_path : str
            Directory to add for DLL dependency lookup. Non-existent or empty
            paths are ignored.

        Returns
        -------
        bool
            True if the directory existed and was added (via add_dll_directory
            or PATH fallback); False otherwise.

        Raises
        ------
        OSError
            Re-raised if `os.add_dll_directory` fails for reasons other than
            WinError 206.
        """
        if not dir_path:
            return False
        if not os.path.isdir(dir_path):
            return False

        try:
            os.add_dll_directory(dir_path)
        except OSError as e:
            # WinError 206: The filename or extension is too long
            if getattr(e, "winerror", None) == 206:
                cur = os.environ.get("PATH", "")
                parts = cur.split(os.pathsep) if cur else []
                if dir_path not in parts:
                    os.environ["PATH"] = dir_path + (os.pathsep + cur if cur else "")
            else:
                raise
        return True

    def _try_add_cuda_from_default_locations() -> str | None:
        """
        Try to locate CUDA under standard Windows install paths.

        Searches the canonical CUDA installation root:
        ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin``

        Returns
        -------
        str or None
            The CUDA `bin` directory path if found and added; otherwise None.
        """
        base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if not base.exists():
            return None

        # Newest version first (lexicographic "v12.4" > "v11.8" works fine here).
        for vdir in sorted(base.glob("v*"), reverse=True):
            cand = vdir / "bin"
            if cand.is_dir():
                if _add_dll_dir_or_path(str(cand)):
                    return str(cand)
        return None

    def _clean_env_path(s: str) -> str:
        """
        Clean environmental variable path by stripping leading/trailing whitespaces, quotation marks and trailing slashes.

        Parameters
        ----------
        s : str
            Environmental path to be cleaned.

        Returns
        -------
        str
            The cleaned environmental path.
        """
        return s.strip().strip('"').rstrip("\\/")

    # ------------------------------------------------------------------
    # Configure dependency search paths
    # ------------------------------------------------------------------
    added_paths: list[str] = []

    # 1) CUDA runtime deps
    cuda_env = _clean_env_path(os.environ.get("CUDA_PATH", ""))
    cuda_bin_added: str | None = None
    if cuda_env:
        cuda_bin = os.path.join(cuda_env, "bin")
        if _add_dll_dir_or_path(cuda_bin):
            cuda_bin_added = cuda_bin
            added_paths.append(cuda_bin)
    else:
        cuda_bin_added = _try_add_cuda_from_default_locations()
        if cuda_bin_added:
            added_paths.append(cuda_bin_added)

    # 2) cuDNN deps (optional explicit override)
    cudnn_env = _clean_env_path(os.environ.get("CUDNN_PATH", ""))
    if cudnn_env:
        cudnn_bin = os.path.join(cudnn_env, "bin")
        if _add_dll_dir_or_path(cudnn_bin):
            added_paths.append(cudnn_bin)

    # 3) Add the DLL folder itself (transitive deps adjacent to DLL)
    if _add_dll_dir_or_path(str(p.parent)):
        added_paths.append(str(p.parent))

    # ------------------------------------------------------------------
    # Load DLL with actionable error reporting
    # ------------------------------------------------------------------
    try:
        return ctypes.CDLL(str(p))
    except OSError as e:
        lines: list[str] = [
            "Failed to load KeyDNN CUDA native backend.",
            "",
            f"Attempted DLL:",
            f"  {p}",
            "",
        ]

        lines.append("DLL search paths added for this process:")
        if added_paths:
            for ap in added_paths:
                lines.append(f"  - {ap}")
        else:
            lines.append("  (none)")

        lines.extend(["", "Potential causes and fixes:"])

        if not cuda_env and cuda_bin_added is None:
            lines.append(
                "- CUDA runtime not found. Set CUDA_PATH to your CUDA install root, e.g.:"
            )
            lines.append(
                "    CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x"
            )
            lines.append("  and ensure <CUDA_PATH>\\bin contains cudart64*.dll.")

        if not cudnn_env:
            lines.append(
                "- cuDNN not explicitly configured. If cuDNN is not installed into CUDA's bin, set CUDNN_PATH, e.g.:"
            )
            lines.append("    CUDNN_PATH=C:\\path\\to\\cudnn")
            lines.append("  and ensure <CUDNN_PATH>\\bin contains cudnn*.dll.")

        lines.extend(
            [
                "- If you changed environment variables, restart the Python process so they are applied cleanly.",
                "- Verify CUDA / cuDNN version compatibility with your installed GPU drivers.",
                "",
                "Original error:",
                f"  {e}",
            ]
        )

        raise OSError("\n".join(lines)) from e
