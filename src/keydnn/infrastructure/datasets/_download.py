"""
Download and integrity utilities for dataset assets.

This module provides minimal, dependency-free helpers for downloading dataset
files from remote URLs into a local cache directory, with optional SHA256
verification.

Design notes
------------
- Uses standard library only (urllib, hashlib) to avoid external dependencies.
- Downloads are performed atomically via a temporary `.part` file that is
  renamed upon successful completion.
- Optional SHA256 verification allows callers to pin exact dataset versions
  and detect corrupted or tampered files.
- This module is intentionally generic and dataset-agnostic.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA256 checksum of a file.

    Parameters
    ----------
    path : Path
        Path to the file to hash.
    chunk_size : int, optional
        Number of bytes to read per iteration. Defaults to 1 MiB.

    Returns
    -------
    str
        Hex-encoded SHA256 digest of the file contents.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url(url: str, dst: Path, *, expected_sha256: Optional[str] = None) -> None:
    """
    Download a file from a URL into a local path with optional integrity checking.

    The file is first downloaded to a temporary `.part` file and then atomically
    renamed to the final destination. If the destination already exists and
    `expected_sha256` is provided, the download is skipped when the checksum
    matches.

    Parameters
    ----------
    url : str
        Source URL to download from.
    dst : Path
        Destination path of the downloaded file.
    expected_sha256 : str, optional
        Expected SHA256 checksum. If provided, the downloaded file is verified
        and a RuntimeError is raised on mismatch.

    Raises
    ------
    RuntimeError
        If checksum verification fails.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If already exists and hash matches, skip.
    if dst.exists() and expected_sha256 is not None:
        if sha256_file(dst) == expected_sha256:
            return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def _reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        # Intentionally quiet by default.
        pass

    urllib.request.urlretrieve(url, tmp.as_posix(), reporthook=_reporthook)
    tmp.replace(dst)

    if expected_sha256 is not None:
        got = sha256_file(dst)
        if got != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {dst.name}: expected {expected_sha256}, got {got}"
            )
