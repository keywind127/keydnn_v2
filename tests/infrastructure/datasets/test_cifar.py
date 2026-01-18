# tests/infrastructure/datasets/test_cifar.py
import io
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.keydnn.infrastructure.datasets._base import _VerboseMixin

_VerboseMixin.set_verbose(False)


def _cifar_batch_dict(n: int, *, num_classes: int, kind: str) -> dict:
    """
    Build a CIFAR-like pickle dict.

    Parameters
    ----------
    n : int
        Number of samples.
    num_classes : int
        Label range [0, num_classes-1].
    kind : str
        "cifar10" or "cifar100" (controls label key naming).

    Returns
    -------
    dict
        Pickle payload matching CIFAR python batch format.
    """
    # CIFAR stores data as (N, 3072) uint8 in row-major, channel-first layout:
    # [R(1024), G(1024), B(1024)]
    count = n * 3072
    data = bytes((i % 256 for i in range(count)))
    data_arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 3072)

    labels = bytes((i % num_classes for i in range(n)))
    labels_arr = np.frombuffer(labels, dtype=np.uint8).reshape(n).tolist()

    if kind == "cifar10":
        return {b"data": data_arr, b"labels": labels_arr}
    if kind == "cifar100":
        # CIFAR-100 uses fine_labels (100 classes) and coarse_labels (20 superclasses)
        coarse = bytes((i % 20 for i in range(n)))
        coarse_arr = np.frombuffer(coarse, dtype=np.uint8).reshape(n).tolist()
        return {
            b"data": data_arr,
            b"fine_labels": labels_arr,
            b"coarse_labels": coarse_arr,
        }

    raise ValueError(f"Unknown kind: {kind}")


def _tar_gz_bytes(members: list[tuple[str, bytes]]) -> bytes:
    """
    Create an in-memory tar.gz containing the provided members.

    Parameters
    ----------
    members : list[(name, bytes)]
        Tar member name and its content.

    Returns
    -------
    bytes
        Complete tar.gz stream.
    """
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return bio.getvalue()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with path.open("wb") as f:
        f.write(payload)


def _make_cifar10_archive_bytes(*, train_n: int, test_n: int) -> bytes:
    """
    Create CIFAR-10 python-version tar.gz bytes with small deterministic batches.
    """
    folder = "cifar-10-batches-py"

    # CIFAR-10 expects data_batch_1..5 and test_batch.
    # We'll create 5 small training batches that sum to train_n.
    # Split train_n across 5 batches (some can be 0).
    sizes = [train_n // 5] * 5
    for i in range(train_n % 5):
        sizes[i] += 1

    members: list[tuple[str, bytes]] = []

    for i, n_i in enumerate(sizes, start=1):
        d = _cifar_batch_dict(n_i, num_classes=10, kind="cifar10")
        payload = pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)
        members.append((f"{folder}/data_batch_{i}", payload))

    d_test = _cifar_batch_dict(test_n, num_classes=10, kind="cifar10")
    payload_test = pickle.dumps(d_test, protocol=pickle.HIGHEST_PROTOCOL)
    members.append((f"{folder}/test_batch", payload_test))

    # Optional metadata files (not required by our loader)
    members.append(
        (
            f"{folder}/batches.meta",
            pickle.dumps({b"label_names": []}, protocol=pickle.HIGHEST_PROTOCOL),
        )
    )

    return _tar_gz_bytes(members)


def _make_cifar100_archive_bytes(*, train_n: int, test_n: int) -> bytes:
    """
    Create CIFAR-100 python-version tar.gz bytes with small deterministic batches.
    """
    folder = "cifar-100-python"

    d_train = _cifar_batch_dict(train_n, num_classes=100, kind="cifar100")
    d_test = _cifar_batch_dict(test_n, num_classes=100, kind="cifar100")

    members = [
        (f"{folder}/train", pickle.dumps(d_train, protocol=pickle.HIGHEST_PROTOCOL)),
        (f"{folder}/test", pickle.dumps(d_test, protocol=pickle.HIGHEST_PROTOCOL)),
        (
            f"{folder}/meta",
            pickle.dumps(
                {b"fine_label_names": [], b"coarse_label_names": []},
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        ),
    ]
    return _tar_gz_bytes(members)


class TestCIFARInfrastructureDataset(TestCase):
    def test_cifar10_download_and_format(self) -> None:
        """
        Verifies:
        - download=True triggers downloader + extraction
        - extracted folder exists under root/cifar10/raw
        - __len__ matches expected counts
        - __getitem__ returns x float32 (3,32,32) in [0,1] and y int in [0,9]
        """
        from src.keydnn.infrastructure.datasets._cifar import CIFAR10

        train_n = 7
        test_n = 4
        archive_bytes = _make_cifar10_archive_bytes(train_n=train_n, test_n=test_n)

        def fake_urlretrieve(url: str, filename: str, reporthook=None, data=None):
            out = Path(filename)
            name = out.name
            if name.endswith(".part"):
                name = name[:-5]

            if name == "cifar-10-python.tar.gz":
                _write_bytes(out, archive_bytes)
            else:
                raise AssertionError(f"Unexpected destination filename: {out.name}")

            return (filename, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fake_urlretrieve,
            ):
                ds_train = CIFAR10(
                    root=root,
                    train=True,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )
                ds_test = CIFAR10(
                    root=root,
                    train=False,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )

            raw_dir = root / "cifar10" / "raw"
            self.assertTrue((raw_dir / "cifar-10-python.tar.gz").exists())
            self.assertTrue((raw_dir / "cifar-10-batches-py").exists())
            self.assertTrue((raw_dir / "cifar-10-batches-py" / "test_batch").exists())

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)

            x0, y0 = ds_train[0]
            self.assertIsInstance(x0, np.ndarray)
            self.assertEqual(x0.shape, (3, 32, 32))
            self.assertEqual(x0.dtype, np.float32)
            self.assertIsInstance(y0, int)

            self.assertGreaterEqual(float(x0.min()), 0.0)
            self.assertLessEqual(float(x0.max()), 1.0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y0, 9)

    def test_cifar100_download_and_format(self) -> None:
        """
        Verifies:
        - download=True triggers downloader + extraction
        - extracted folder exists under root/cifar100/raw
        - __len__ matches expected counts
        - __getitem__ returns x float32 (3,32,32) in [0,1] and y int in [0,99]
        """
        from src.keydnn.infrastructure.datasets._cifar import CIFAR100

        train_n = 6
        test_n = 5
        archive_bytes = _make_cifar100_archive_bytes(train_n=train_n, test_n=test_n)

        def fake_urlretrieve(url: str, filename: str, reporthook=None, data=None):
            out = Path(filename)
            name = out.name
            if name.endswith(".part"):
                name = name[:-5]

            if name == "cifar-100-python.tar.gz":
                _write_bytes(out, archive_bytes)
            else:
                raise AssertionError(f"Unexpected destination filename: {out.name}")

            return (filename, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fake_urlretrieve,
            ):
                ds_train = CIFAR100(
                    root=root,
                    train=True,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )
                ds_test = CIFAR100(
                    root=root,
                    train=False,
                    download=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )

            raw_dir = root / "cifar100" / "raw"
            self.assertTrue((raw_dir / "cifar-100-python.tar.gz").exists())
            self.assertTrue((raw_dir / "cifar-100-python").exists())
            self.assertTrue((raw_dir / "cifar-100-python" / "train").exists())
            self.assertTrue((raw_dir / "cifar-100-python" / "test").exists())

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)

            x0, y0 = ds_train[0]
            self.assertIsInstance(x0, np.ndarray)
            self.assertEqual(x0.shape, (3, 32, 32))
            self.assertEqual(x0.dtype, np.float32)
            self.assertIsInstance(y0, int)

            self.assertGreaterEqual(float(x0.min()), 0.0)
            self.assertLessEqual(float(x0.max()), 1.0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y0, 99)

    def test_download_false_raises_if_missing(self) -> None:
        """
        Verifies:
        - download=False raises FileNotFoundError when raw files are missing.
        """
        from src.keydnn.infrastructure.datasets._cifar import CIFAR10, CIFAR100

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaises(FileNotFoundError):
                _ = CIFAR10(root=root, train=True, download=False)

            with self.assertRaises(FileNotFoundError):
                _ = CIFAR100(root=root, train=True, download=False)


if __name__ == "__main__":
    unittest.main()
