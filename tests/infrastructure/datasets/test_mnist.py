import unittest
from unittest import TestCase
from unittest.mock import patch

import gzip
import struct
import tempfile
from pathlib import Path

import numpy as np


import gzip
import struct
from pathlib import Path


def _idx_images_payload(n: int, rows: int = 28, cols: int = 28) -> bytes:
    header = struct.pack(">IIII", 2051, n, rows, cols)
    count = n * rows * cols

    # Pure-Python deterministic pixels (no NumPy)
    pixels = bytes((i % 256 for i in range(count)))

    return header + pixels


def _idx_labels_payload(n: int) -> bytes:
    header = struct.pack(">II", 2049, n)
    labels = bytes((i % 10 for i in range(n)))
    return header + labels


def _write_gzip_bytes(dst: Path, raw_payload: bytes) -> None:
    """
    Write *exactly one* gzip member whose decompressed bytes equal raw_payload.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    compressed = gzip.compress(raw_payload)  # returns full gzip stream bytes
    with dst.open("wb") as f:
        f.write(compressed)


def _validate_idx_images_gz(path: Path) -> None:
    """
    Ensure gzip contains exactly:
      16-byte header + n*rows*cols bytes, no trailing bytes.
    """
    with gzip.open(path, "rb") as f:
        data = f.read()

    if len(data) < 16:
        raise AssertionError(f"{path.name}: too small to be IDX images")

    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise AssertionError(f"{path.name}: bad magic {magic} (expected 2051)")

    expected = 16 + (n * rows * cols)
    if len(data) != expected:
        trailing = len(data) - expected
        raise AssertionError(
            f"{path.name}: decompressed size mismatch: got={len(data)} expected={expected} "
            f"(n={n}, rows={rows}, cols={cols}), trailing_bytes={trailing}"
        )


def _validate_idx_labels_gz(path: Path) -> None:
    """
    Ensure gzip contains exactly:
      8-byte header + n bytes, no trailing bytes.
    """
    with gzip.open(path, "rb") as f:
        data = f.read()

    if len(data) < 8:
        raise AssertionError(f"{path.name}: too small to be IDX labels")

    magic, n = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise AssertionError(f"{path.name}: bad magic {magic} (expected 2049)")

    expected = 8 + n
    if len(data) != expected:
        trailing = len(data) - expected
        raise AssertionError(
            f"{path.name}: decompressed size mismatch: got={len(data)} expected={expected} "
            f"(n={n}), trailing_bytes={trailing}"
        )


class TestMNISTInfrastructureDataset(TestCase):
    def test_download_and_format(self) -> None:
        from src.keydnn.infrastructure.datasets._mnist import MNIST

        train_n = 2
        test_n = 3

        def fake_urlretrieve(url: str, filename: str, reporthook=None, data=None):
            out = Path(filename)
            name = out.name
            if name.endswith(".part"):
                name = name[:-5]

            if name == "train-images-idx3-ubyte.gz":
                _write_gzip_bytes(out, _idx_images_payload(train_n))
                _validate_idx_images_gz(out)
            elif name == "train-labels-idx1-ubyte.gz":
                _write_gzip_bytes(out, _idx_labels_payload(train_n))
                _validate_idx_labels_gz(out)
            elif name == "t10k-images-idx3-ubyte.gz":
                _write_gzip_bytes(out, _idx_images_payload(test_n))
                _validate_idx_images_gz(out)
            elif name == "t10k-labels-idx1-ubyte.gz":
                _write_gzip_bytes(out, _idx_labels_payload(test_n))
                _validate_idx_labels_gz(out)
            else:
                raise AssertionError(f"Unexpected destination filename: {out.name}")

            return (filename, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fake_urlretrieve,
            ):
                ds_train = MNIST(
                    root=root,
                    train=True,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )
                ds_test = MNIST(
                    root=root,
                    train=False,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )

            raw_dir = root / "mnist" / "raw"
            self.assertTrue((raw_dir / "train-images-idx3-ubyte.gz").exists())
            self.assertTrue((raw_dir / "train-labels-idx1-ubyte.gz").exists())
            self.assertTrue((raw_dir / "t10k-images-idx3-ubyte.gz").exists())
            self.assertTrue((raw_dir / "t10k-labels-idx1-ubyte.gz").exists())

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)

            x0, y0 = ds_train[0]

            self.assertIsInstance(x0, np.ndarray)
            self.assertEqual(x0.shape, (1, 28, 28))
            self.assertEqual(x0.dtype, np.float32)
            self.assertIsInstance(y0, int)

            self.assertGreaterEqual(float(x0.min()), 0.0)
            self.assertLessEqual(float(x0.max()), 1.0)

            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y0, 9)

    def test_download_false_raises_if_missing(self) -> None:
        from src.keydnn.infrastructure.datasets._mnist import MNIST

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                _ = MNIST(root=root, train=True, download=False)


if __name__ == "__main__":
    unittest.main()
