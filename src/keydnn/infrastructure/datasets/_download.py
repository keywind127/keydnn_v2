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
import sys
import time
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


def _make_progress_hook(label: str):
    """
    Create a progress-reporting hook compatible with urllib.request.urlretrieve.

    The hook renders a lightweight, single-line progress indicator to stdout
    while the download is in progress. If the server provides a total content
    length, a percentage-based progress bar is shown; otherwise, a simple
    downloaded-bytes counter is displayed.

    This function is intentionally internal and dependency-free, avoiding
    third-party progress bar libraries (e.g., tqdm) to keep infrastructure
    utilities lightweight and portable.
    """
    start_time = time.time()
    last_print = 0.0

    def hook(blocknum: int, blocksize: int, totalsize: int) -> None:
        nonlocal last_print

        downloaded = blocknum * blocksize
        now = time.time()

        # Throttle updates to avoid spamming stdout
        if now - last_print < 0.1:
            return
        last_print = now

        if totalsize > 0:
            frac = min(downloaded / totalsize, 1.0)
            bar_width = 30
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            percent = frac * 100.0

            sys.stdout.write(f"\rDownloading {label}: [{bar}] {percent:6.2f}%")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloading {label}: {mb:6.2f} MB")

        sys.stdout.flush()

    return hook


def download_url(url: str, dst: Path, *, expected_sha256: Optional[str] = None) -> None:
    """
    Download a file from a URL into a local path with optional integrity checking.

    The file is first downloaded to a temporary `.part` file and then atomically
    renamed to the final destination. If the destination already exists and
    `expected_sha256` is provided, the download is skipped when the checksum
    matches.

    During download, a lightweight terminal progress indicator is displayed
    when supported by the remote server.

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

    label = dst.name
    hook = _make_progress_hook(label)

    try:
        urllib.request.urlretrieve(url, tmp.as_posix(), reporthook=hook)
        sys.stdout.write("\n")
    except Exception:
        sys.stdout.write("\n")
        raise

    tmp.replace(dst)

    if expected_sha256 is not None:
        got = sha256_file(dst)
        if got != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {dst.name}: expected {expected_sha256}, got {got}"
            )
