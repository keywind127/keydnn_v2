"""
infrastructure/native_cuda/python/_native_loader.py (DLL loader excerpt)

This snippet provides a cached Windows DLL loader for the KeyDNN CUDA native
backend.

It resolves the expected DLL path within the repository, ensures that dependent
DLL directories (CUDA runtime and cuDNN) are discoverable by the current Python
process, and returns a `ctypes.CDLL` handle that can be reused across the
codebase.

Key behaviors
-------------
- Cached singleton: `load_keydnn_cuda_native()` is decorated with `lru_cache`
  so the DLL is loaded only once per process.
- Dependency resolution: attempts to register directories via
  `os.add_dll_directory` (preferred on modern Windows/Python).
- Robust fallback: if `os.add_dll_directory` fails with WinError 206, it falls
  back to prepending the directory onto `PATH` for the current process.
- Explicit failure: raises `FileNotFoundError` if the target DLL path does not
  exist.

Platform notes
--------------
- Windows-only: depends on Windows DLL search behavior and `ctypes.CDLL`.
- The WinError 206 workaround exists to handle environments where adding a DLL
  directory fails due to overly-long paths.
"""

from functools import lru_cache

# ---------------------------------------------------------------------
# DLL loading
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
        If set, `<CUDA_PATH>/bin` is added to the DLL search path to help locate
        CUDA runtime dependencies.
    CUDNN_PATH : str, optional
        If set, `<CUDNN_PATH>/bin` is added to the DLL search path to help locate
        cuDNN dependencies.

    Returns
    -------
    ctypes.CDLL
        Loaded DLL handle for the KeyDNN CUDA native backend.

    Raises
    ------
    FileNotFoundError
        If the expected DLL does not exist at the resolved path.
    OSError
        If DLL directory configuration fails for reasons other than WinError 206,
        or if the DLL fails to load.

    Notes
    -----
    - Uses `lru_cache(maxsize=1)` so the DLL is loaded at most once per process.
    - Calls `_add_dll_dir_or_path` for CUDA/cuDNN bins and the DLL's own
      directory to ensure transitive dependencies resolve correctly.
    """
    import os
    import ctypes
    from pathlib import Path

    # your existing path logic:
    p = (
        Path(__file__).resolve().parents[2]
        / "native_cuda"
        / "keydnn_v2_cuda_native"
        / "x64"
        / "Release"
        / "KeyDNNV2CudaNative.dll"
    )
    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"KeyDNN CUDA native DLL not found at: {p}")

    def _add_dll_dir_or_path(dir_path: str) -> None:
        """
        Add a directory for DLL dependency resolution.

        This helper uses `os.add_dll_directory(dir_path)` when available, which
        is the recommended approach on modern Windows and Python versions.

        Some Windows setups can raise WinError 206 ("The filename or extension
        is too long") when calling `os.add_dll_directory`. In that specific
        case, this helper falls back to process-local PATH modification by
        prepending `dir_path` to `os.environ["PATH"]`.

        Parameters
        ----------
        dir_path : str
            Directory to add for DLL dependency lookup. Non-existent or empty
            paths are ignored.

        Raises
        ------
        OSError
            Re-raised if `os.add_dll_directory` fails for reasons other than
            WinError 206.
        """
        if not dir_path:
            return
        if not os.path.isdir(dir_path):
            return

        try:
            os.add_dll_directory(dir_path)
        except OSError as e:
            # WinError 206: The filename or extension is too long
            if getattr(e, "winerror", None) == 206:
                # Fallback: PATH-based resolution for this process
                cur = os.environ.get("PATH", "")
                parts = cur.split(os.pathsep) if cur else []
                if dir_path not in parts:
                    os.environ["PATH"] = (
                        dir_path + os.pathsep + cur if cur else dir_path
                    )
            else:
                raise

    # ---- ensure CUDA runtime deps are discoverable ----
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        _add_dll_dir_or_path(cuda_bin)

    cudnn_path = os.environ.get("CUDNN_PATH", "")
    if cudnn_path:
        cudnn_bin = os.path.join(cudnn_path, "bin")
        _add_dll_dir_or_path(cudnn_bin)

    # Also add the DLL folder itself (this is where you currently crash with WinError 206)
    _add_dll_dir_or_path(str(p.parent))

    return ctypes.CDLL(str(p))
