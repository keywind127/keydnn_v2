"""
IDX file format loaders.

This module implements minimal parsers for the IDX binary format used by
classic datasets such as MNIST. Both image and label files are supported,
with gzip-compressed inputs handled transparently.

Design notes
------------
- Parsing is strict and validates IDX magic numbers.
- Data is returned as NumPy arrays with native dtypes (uint8).
- No normalization or reshaping beyond the IDX specification is performed.
"""

from __future__ import annotations

import struct
from pathlib import Path
import gzip
import numpy as np


def _read_u32_be(b: bytes, offset: int) -> int:
    """
    Read a big-endian unsigned 32-bit integer from a byte buffer.

    Parameters
    ----------
    b : bytes
        Byte buffer.
    offset : int
        Offset into the buffer.

    Returns
    -------
    int
        Parsed unsigned 32-bit integer.
    """
    return struct.unpack_from(">I", b, offset)[0]


def load_idx_images_gz(path: Path) -> np.ndarray:
    """
    Load an IDX image file from a gzip-compressed source.

    Parameters
    ----------
    path : Path
        Path to a `.gz` file containing IDX image data.

    Returns
    -------
    np.ndarray
        Image tensor of shape (N, H, W) with dtype uint8.

    Raises
    ------
    ValueError
        If the IDX magic number is invalid.
    """
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
    """
    Load an IDX label file from a gzip-compressed source.

    Parameters
    ----------
    path : Path
        Path to a `.gz` file containing IDX label data.

    Returns
    -------
    np.ndarray
        Label vector of shape (N,) with dtype uint8.

    Raises
    ------
    ValueError
        If the IDX magic number is invalid.
    """
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = _read_u32_be(data, 0)
    if magic != 2049:
        raise ValueError(f"Bad IDX label magic {magic} in {path.name}")

    n = _read_u32_be(data, 4)
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    labels = labels.reshape(n)
    return labels
