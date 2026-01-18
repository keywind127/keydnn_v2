import io
import gzip
import pickle
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np


# --------------------------------------------------------------------------------------
# MNIST helpers (match your infrastructure dataset tests)
# --------------------------------------------------------------------------------------


def _idx_images_payload(n: int, rows: int = 28, cols: int = 28) -> bytes:
    header = struct.pack(">IIII", 2051, n, rows, cols)
    count = n * rows * cols
    pixels = bytes((i % 256 for i in range(count)))
    return header + pixels


def _idx_labels_payload(n: int) -> bytes:
    header = struct.pack(">II", 2049, n)
    labels = bytes((i % 10 for i in range(n)))
    return header + labels


def _write_gzip_bytes(dst: Path, raw_payload: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    compressed = gzip.compress(raw_payload)
    with dst.open("wb") as f:
        f.write(compressed)


# --------------------------------------------------------------------------------------
# CIFAR helpers (match your infrastructure dataset tests)
# --------------------------------------------------------------------------------------


def _cifar_batch_dict(n: int, *, num_classes: int, kind: str) -> dict:
    # CIFAR stores data as (N, 3072) uint8
    count = n * 3072
    data = bytes((i % 256 for i in range(count)))
    data_arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 3072)

    labels = bytes((i % num_classes for i in range(n)))
    labels_arr = np.frombuffer(labels, dtype=np.uint8).reshape(n).tolist()

    if kind == "cifar10":
        return {b"data": data_arr, b"labels": labels_arr}
    if kind == "cifar100":
        coarse = bytes((i % 20 for i in range(n)))
        coarse_arr = np.frombuffer(coarse, dtype=np.uint8).reshape(n).tolist()
        return {
            b"data": data_arr,
            b"fine_labels": labels_arr,
            b"coarse_labels": coarse_arr,
        }
    raise ValueError(f"Unknown kind: {kind}")


def _tar_gz_bytes(members: list[tuple[str, bytes]]) -> bytes:
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
    folder = "cifar-10-batches-py"

    sizes = [train_n // 5] * 5
    for i in range(train_n % 5):
        sizes[i] += 1

    members: list[tuple[str, bytes]] = []

    for i, n_i in enumerate(sizes, start=1):
        d = _cifar_batch_dict(n_i, num_classes=10, kind="cifar10")
        members.append(
            (
                f"{folder}/data_batch_{i}",
                pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL),
            )
        )

    d_test = _cifar_batch_dict(test_n, num_classes=10, kind="cifar10")
    members.append(
        (f"{folder}/test_batch", pickle.dumps(d_test, protocol=pickle.HIGHEST_PROTOCOL))
    )

    # optional meta
    members.append(
        (
            f"{folder}/batches.meta",
            pickle.dumps({b"label_names": []}, protocol=pickle.HIGHEST_PROTOCOL),
        )
    )

    return _tar_gz_bytes(members)


def _make_cifar100_archive_bytes(*, train_n: int, test_n: int) -> bytes:
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


# --------------------------------------------------------------------------------------
# Presentation wrapper tests
# --------------------------------------------------------------------------------------


class TestDatasetLoadersMNIST(TestCase):
    def test_load_mnist_downloads_when_missing(self) -> None:
        from src.keydnn.presentation.apis.datasets.mnist import load_mnist

        train_n = 2
        test_n = 3

        def fake_urlretrieve(url: str, filename: str, reporthook=None, data=None):
            out = Path(filename)
            name = out.name
            if name.endswith(".part"):
                name = name[:-5]

            if name == "train-images-idx3-ubyte.gz":
                _write_gzip_bytes(out, _idx_images_payload(train_n))
            elif name == "train-labels-idx1-ubyte.gz":
                _write_gzip_bytes(out, _idx_labels_payload(train_n))
            elif name == "t10k-images-idx3-ubyte.gz":
                _write_gzip_bytes(out, _idx_images_payload(test_n))
            elif name == "t10k-labels-idx1-ubyte.gz":
                _write_gzip_bytes(out, _idx_labels_payload(test_n))
            else:
                raise AssertionError(f"Unexpected destination filename: {out.name}")

            return (filename, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fake_urlretrieve,
            ):
                ds_train = load_mnist(
                    root_path=root,
                    train=True,
                    normalize=False,
                    return_numpy=True,
                    dtype="float32",
                )
                ds_test = load_mnist(
                    root_path=root,
                    train=False,
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

    def test_load_mnist_does_not_download_when_present(self) -> None:
        from src.keydnn.presentation.apis.datasets.mnist import load_mnist

        train_n = 2
        test_n = 3

        def fail_urlretrieve(*args, **kwargs):
            raise AssertionError(
                "Network call attempted, but dataset should already exist."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "mnist" / "raw"

            # Pre-create exactly what MNIST expects so download=False is safe.
            _write_gzip_bytes(
                raw_dir / "train-images-idx3-ubyte.gz", _idx_images_payload(train_n)
            )
            _write_gzip_bytes(
                raw_dir / "train-labels-idx1-ubyte.gz", _idx_labels_payload(train_n)
            )
            _write_gzip_bytes(
                raw_dir / "t10k-images-idx3-ubyte.gz", _idx_images_payload(test_n)
            )
            _write_gzip_bytes(
                raw_dir / "t10k-labels-idx1-ubyte.gz", _idx_labels_payload(test_n)
            )

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fail_urlretrieve,
            ):
                ds_train = load_mnist(
                    root_path=root, train=True, normalize=False, return_numpy=True
                )
                ds_test = load_mnist(
                    root_path=root, train=False, normalize=False, return_numpy=True
                )

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)


class TestDatasetLoadersCIFAR(TestCase):
    def test_load_cifar10_downloads_when_missing(self) -> None:
        from src.keydnn.presentation.apis.datasets.cifar import load_cifar10

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
                ds_train = load_cifar10(
                    root_path=root, train=True, normalize=False, return_numpy=True
                )
                ds_test = load_cifar10(
                    root_path=root, train=False, normalize=False, return_numpy=True
                )

            raw_dir = root / "cifar10" / "raw"
            self.assertTrue((raw_dir / "cifar-10-python.tar.gz").exists())
            self.assertTrue((raw_dir / "cifar-10-batches-py").exists())

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)

            x0, y0 = ds_train[0]
            self.assertIsInstance(x0, np.ndarray)
            self.assertEqual(x0.shape, (3, 32, 32))
            self.assertEqual(x0.dtype, np.float32)
            self.assertIsInstance(y0, int)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y0, 9)

    def test_load_cifar10_does_not_download_when_present(self) -> None:
        from src.keydnn.presentation.apis.datasets.cifar import load_cifar10

        def fail_urlretrieve(*args, **kwargs):
            raise AssertionError(
                "Network call attempted, but dataset should already exist."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "cifar10" / "raw"
            base = raw_dir / "cifar-10-batches-py"
            base.mkdir(parents=True, exist_ok=True)

            # Minimal valid CIFAR-10 layout: data_batch_1..5 + test_batch
            sizes = [1, 1, 0, 0, 0]
            for i, n_i in enumerate(sizes, start=1):
                d = _cifar_batch_dict(n_i, num_classes=10, kind="cifar10")
                _write_bytes(
                    base / f"data_batch_{i}",
                    pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL),
                )

            d_test = _cifar_batch_dict(2, num_classes=10, kind="cifar10")
            _write_bytes(
                base / "test_batch",
                pickle.dumps(d_test, protocol=pickle.HIGHEST_PROTOCOL),
            )

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fail_urlretrieve,
            ):
                ds_train = load_cifar10(
                    root_path=root, train=True, normalize=False, return_numpy=True
                )
                ds_test = load_cifar10(
                    root_path=root, train=False, normalize=False, return_numpy=True
                )

            self.assertEqual(len(ds_train), sum(sizes))
            self.assertEqual(len(ds_test), 2)

    def test_load_cifar100_downloads_when_missing(self) -> None:
        from src.keydnn.presentation.apis.datasets.cifar import load_cifar100

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
                ds_train = load_cifar100(
                    root_path=root, train=True, normalize=False, return_numpy=True
                )
                ds_test = load_cifar100(
                    root_path=root, train=False, normalize=False, return_numpy=True
                )

            raw_dir = root / "cifar100" / "raw"
            self.assertTrue((raw_dir / "cifar-100-python.tar.gz").exists())
            self.assertTrue((raw_dir / "cifar-100-python").exists())

            self.assertEqual(len(ds_train), train_n)
            self.assertEqual(len(ds_test), test_n)

            x0, y0 = ds_train[0]
            self.assertIsInstance(x0, np.ndarray)
            self.assertEqual(x0.shape, (3, 32, 32))
            self.assertEqual(x0.dtype, np.float32)
            self.assertIsInstance(y0, int)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(y0, 99)

    def test_load_cifar100_does_not_download_when_present(self) -> None:
        from src.keydnn.presentation.apis.datasets.cifar import load_cifar100

        def fail_urlretrieve(*args, **kwargs):
            raise AssertionError(
                "Network call attempted, but dataset should already exist."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "cifar100" / "raw"
            base = raw_dir / "cifar-100-python"
            base.mkdir(parents=True, exist_ok=True)

            d_train = _cifar_batch_dict(3, num_classes=100, kind="cifar100")
            d_test = _cifar_batch_dict(2, num_classes=100, kind="cifar100")
            _write_bytes(
                base / "train", pickle.dumps(d_train, protocol=pickle.HIGHEST_PROTOCOL)
            )
            _write_bytes(
                base / "test", pickle.dumps(d_test, protocol=pickle.HIGHEST_PROTOCOL)
            )
            _write_bytes(
                base / "meta",
                pickle.dumps(
                    {b"fine_label_names": [], b"coarse_label_names": []},
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )

            with patch(
                "src.keydnn.infrastructure.datasets._download.urllib.request.urlretrieve",
                new=fail_urlretrieve,
            ):
                ds_train = load_cifar100(
                    root_path=root, train=True, normalize=False, return_numpy=True
                )
                ds_test = load_cifar100(
                    root_path=root, train=False, normalize=False, return_numpy=True
                )

            self.assertEqual(len(ds_train), 3)
            self.assertEqual(len(ds_test), 2)


if __name__ == "__main__":
    unittest.main()
