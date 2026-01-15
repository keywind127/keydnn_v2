# src/keydnn/infrastructure/datasets/_download.py
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url(url: str, dst: Path, *, expected_sha256: Optional[str] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If already exists and hash matches, skip.
    if dst.exists() and expected_sha256 is not None:
        if sha256_file(dst) == expected_sha256:
            return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def _reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        # Optional: keep it quiet by default, or print progress when totalsize known.
        pass

    urllib.request.urlretrieve(url, tmp.as_posix(), reporthook=_reporthook)
    tmp.replace(dst)

    if expected_sha256 is not None:
        got = sha256_file(dst)
        if got != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {dst.name}: expected {expected_sha256}, got {got}"
            )
