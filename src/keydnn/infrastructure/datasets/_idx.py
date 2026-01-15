# src/keydnn/infrastructure/datasets/_idx.py
from __future__ import annotations

import struct
from pathlib import Path
import gzip
import numpy as np


def _read_u32_be(b: bytes, offset: int) -> int:
    return struct.unpack_from(">I", b, offset)[0]


def load_idx_images_gz(path: Path) -> np.ndarray:
    # returns uint8 array (N, 28, 28)
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = _read_u32_be(data, 0)
    if magic != 2051:
        raise ValueError(f"Bad IDX image magic {magic} in {path.name}")

    n = _read_u32_be(data, 4)
    rows = _read_u32_be(data, 8)
    cols = _read_u32_be(data, 12)

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(n, rows, cols)
    return images


def load_idx_labels_gz(path: Path) -> np.ndarray:
    # returns uint8 array (N,)
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = _read_u32_be(data, 0)
    if magic != 2049:
        raise ValueError(f"Bad IDX label magic {magic} in {path.name}")

    n = _read_u32_be(data, 4)
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    labels = labels.reshape(n)
    return labels
