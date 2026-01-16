"""
CIFAR dataset loader (CIFAR-10 / CIFAR-100).

This module provides lightweight, self-contained implementations of the
CIFAR-10 and CIFAR-100 datasets, including automatic download, caching,
archive extraction, and basic preprocessing.

Design notes
------------
- Dataset archives are cached under:
  - `<root>/cifar10/raw` for CIFAR-10
  - `<root>/cifar100/raw` for CIFAR-100
- Network access is optional and explicitly controlled via `download=True`.
- Data is returned as NumPy arrays by default to avoid tight coupling with
  any specific Tensor implementation.
- Normalization and transformation hooks mirror common deep learning APIs.
- The CIFAR "python version" archive contains pickled batch files:
  - CIFAR-10: `cifar-10-batches-py/data_batch_1..5`, `test_batch`
  - CIFAR-100: `cifar-100-python/train`, `test`

References
----------
- Official CIFAR dataset page and archive structure:
  https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from ._download import download_url


# --------------------------------------------------------------------------------------
# URLs + archive metadata
# --------------------------------------------------------------------------------------

_CIFAR10_BASE_URLS: List[str] = [
    # Reliable mirrors first (often faster / less flaky)
    "https://ossci-datasets.s3.amazonaws.com/cifar10/",
    # Original host last
    "https://www.cs.toronto.edu/~kriz/",
]

_CIFAR100_BASE_URLS: List[str] = [
    "https://ossci-datasets.s3.amazonaws.com/cifar100/",
    "https://www.cs.toronto.edu/~kriz/",
]

_CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
_CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"

# Expected extracted folder names inside the tarball
_CIFAR10_FOLDER = "cifar-10-batches-py"
_CIFAR100_FOLDER = "cifar-100-python"

_CIFAR_ARCHIVE_SHA256 = {
    _CIFAR10_ARCHIVE: None,
    _CIFAR100_ARCHIVE: None,
}


def _expand_root(root: Union[str, Path]) -> Path:
    """
    Expand and normalize a dataset root path.

    Parameters
    ----------
    root : str or Path
        Root directory path.

    Returns
    -------
    Path
        Absolute, expanded path.
    """
    return Path(root).expanduser().resolve()


def _safe_pickle_load(path: Path) -> dict:
    """
    Load a CIFAR "python version" pickle payload.

    The CIFAR "python version" uses pickled dictionaries with byte-string keys.
    This helper centralizes that decoding behavior.

    Parameters
    ----------
    path : Path
        Path to a CIFAR batch file.

    Returns
    -------
    dict
        Pickle payload dictionary.
    """
    with path.open("rb") as f:
        # encoding='bytes' preserves byte keys as bytes (common for CIFAR batches)
        return pickle.load(f, encoding="bytes")  # type: ignore[arg-type]


def _load_cifar10_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a CIFAR-10 pickled batch file.

    Parameters
    ----------
    path : Path
        Path to a CIFAR-10 batch file (e.g., `data_batch_1`, `test_batch`).

    Returns
    -------
    (np.ndarray, np.ndarray)
        images : uint8 array of shape (N, 3, 32, 32) in NCHW order
        labels : int64 array of shape (N,)
    """
    d = _safe_pickle_load(path)

    data = np.asarray(d[b"data"], dtype=np.uint8)  # (N, 3072)
    labels = np.asarray(d[b"labels"], dtype=np.int64)  # (N,)

    images = data.reshape(-1, 3, 32, 32)  # NCHW
    return images, labels


def _extract_tar_gz(archive_path: Path, dst_dir: Path) -> None:
    """
    Extract a .tar.gz archive into dst_dir.

    Parameters
    ----------
    archive_path : Path
        Path to the .tar.gz archive.
    dst_dir : Path
        Output directory for extraction.

    Returns
    -------
    None
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)


def _ensure_extracted_folder(raw_dir: Path, *, folder_name: str) -> Path:
    """
    Return extracted dataset folder path and validate existence.

    Parameters
    ----------
    raw_dir : Path
        Raw directory where archive is extracted.
    folder_name : str
        Expected extracted top-level folder name.

    Returns
    -------
    Path
        Path to extracted folder.

    Raises
    ------
    FileNotFoundError
        If the extracted folder is missing.
    """
    folder = raw_dir / folder_name
    if not folder.exists():
        raise FileNotFoundError(f"Expected extracted folder missing: {folder}")
    return folder


# --------------------------------------------------------------------------------------
# CIFAR-10
# --------------------------------------------------------------------------------------


@dataclass
class CIFAR10:
    """
    CIFAR-10 dataset (32x32 RGB, 10 classes).

    Provides indexed access to CIFAR-10 with automatic download, caching,
    archive extraction, normalization, and transform hooks.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.
    train : bool, optional
        Whether to load the training split. If False, loads the test split.
    download : bool, optional
        Whether to download missing files automatically.
    transform : callable, optional
        Optional transform applied to each image.
    target_transform : callable, optional
        Optional transform applied to each label.
    normalize : bool, optional
        Whether to apply standard CIFAR-10 mean/std normalization in [0,1].
    return_numpy : bool, optional
        Whether to return NumPy arrays (default). Tensor conversion can be
        layered on top by the caller.
    dtype : str, optional
        Floating-point dtype used for image conversion.

    Notes
    -----
    - Images are returned as shape (3, 32, 32) by default (channel-first).
    - If `normalize=True`, normalization is applied per-channel.
    - The extracted folder `cifar-10-batches-py/` is preserved under the raw
      directory to match the official archive structure (and unit tests).
    """

    root: Union[str, Path]
    train: bool = True
    download: bool = False
    transform: Optional[Callable[[Any], Any]] = None
    target_transform: Optional[Callable[[Any], Any]] = None
    normalize: bool = False
    return_numpy: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """
        Resolve paths, ensure data availability, and load dataset into memory.
        """
        self.root = _expand_root(self.root)
        self.raw_dir = self.root / "cifar10" / "raw"

        self._ensure_data()

        base = _ensure_extracted_folder(self.raw_dir, folder_name=_CIFAR10_FOLDER)

        if self.train:
            batch_files = [base / f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = [base / "test_batch"]

        images_list = []
        labels_list = []

        for path in batch_files:
            x, y = _load_cifar10_batch(path)
            images_list.append(x)
            labels_list.append(y)

        self.images = np.concatenate(images_list, axis=0)  # (N,3,32,32) uint8
        self.labels = np.concatenate(labels_list, axis=0)  # (N,) int64

        # Normalize constants (standard CIFAR-10 in [0,1] space)
        self._mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        self._std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

    def _ensure_data(self) -> None:
        """
        Ensure CIFAR-10 archive is present and extracted locally.

        Downloads missing archive and extracts it when `download=True`.

        Raises
        ------
        FileNotFoundError
            If required files are missing and downloading is disabled.
        RuntimeError
            If a download fails from all configured mirrors.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        extracted = self.raw_dir / _CIFAR10_FOLDER

        # If already extracted (official layout), we are done.
        if extracted.exists():
            return

        if not self.download:
            raise FileNotFoundError(
                f"CIFAR-10 files missing in {self.raw_dir}. "
                f"Set download=True to fetch and extract them."
            )

        archive_path = self.raw_dir / _CIFAR10_ARCHIVE
        expected_sha256 = _CIFAR_ARCHIVE_SHA256[_CIFAR10_ARCHIVE]

        last_err: Exception | None = None
        for base_url in _CIFAR10_BASE_URLS:
            try:
                url = base_url + _CIFAR10_ARCHIVE
                download_url(url, archive_path, expected_sha256=expected_sha256)
                _extract_tar_gz(archive_path, self.raw_dir)
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise RuntimeError(
                f"Failed to download {_CIFAR10_ARCHIVE}: {last_err}"
            ) from last_err

        # Validate extracted folder exists (matches unit tests).
        _ = _ensure_extracted_folder(self.raw_dir, folder_name=_CIFAR10_FOLDER)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieve a single (image, label) pair.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        (Any, Any)
            Image and label pair. By default, the image is a NumPy array of
            shape (3, 32, 32) and dtype `self.dtype`, and the label is an int.
        """
        x = self.images[idx].astype(self.dtype) / 255.0
        y = int(self.labels[idx])

        if self.normalize:
            x = (x - self._mean[:, None, None]) / self._std[:, None, None]

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


def download_cifar10(root: str | Path) -> Path:
    """
    Convenience helper to download and extract CIFAR-10.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.

    Returns
    -------
    Path
        Path to the CIFAR-10 raw data directory.
    """
    ds = CIFAR10(root=root, train=True, download=True)
    _ = len(ds)
    return Path(root).expanduser().resolve() / "cifar10" / "raw"


# --------------------------------------------------------------------------------------
# CIFAR-100
# --------------------------------------------------------------------------------------


@dataclass
class CIFAR100:
    """
    CIFAR-100 dataset (32x32 RGB, 100 classes).

    This implementation loads the "fine" labels by default.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.
    train : bool, optional
        Whether to load the training split. If False, loads the test split.
    download : bool, optional
        Whether to download missing files automatically.
    transform : callable, optional
        Optional transform applied to each image.
    target_transform : callable, optional
        Optional transform applied to each label.
    normalize : bool, optional
        Whether to apply standard CIFAR-100 mean/std normalization in [0,1].
    return_numpy : bool, optional
        Whether to return NumPy arrays (default).
    dtype : str, optional
        Floating-point dtype used for image conversion.
    """

    root: Union[str, Path]
    train: bool = True
    download: bool = False
    transform: Optional[Callable[[Any], Any]] = None
    target_transform: Optional[Callable[[Any], Any]] = None
    normalize: bool = False
    return_numpy: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """
        Resolve paths, ensure data availability, and load dataset into memory.
        """
        self.root = _expand_root(self.root)
        self.raw_dir = self.root / "cifar100" / "raw"

        self._ensure_data()

        base = _ensure_extracted_folder(self.raw_dir, folder_name=_CIFAR100_FOLDER)
        batch_file = base / ("train" if self.train else "test")

        d = _safe_pickle_load(batch_file)
        x = np.asarray(d[b"data"], dtype=np.uint8)  # (N, 3072)
        # fine labels are the actual 100 classes
        y = np.asarray(d[b"fine_labels"], dtype=np.uint8)  # (N,)

        self.images = x.reshape(-1, 3, 32, 32)
        self.labels = y.reshape(-1)

        # CIFAR-100 mean/std (in [0,1]) commonly used
        self._mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
        self._std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

    def _ensure_data(self) -> None:
        """
        Ensure CIFAR-100 archive is present and extracted locally.

        Downloads missing archive and extracts it when `download=True`.

        Raises
        ------
        FileNotFoundError
            If required files are missing and downloading is disabled.
        RuntimeError
            If a download fails from all configured mirrors.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        extracted = self.raw_dir / _CIFAR100_FOLDER

        if extracted.exists():
            return

        if not self.download:
            raise FileNotFoundError(
                f"CIFAR-100 files missing in {self.raw_dir}. "
                f"Set download=True to fetch and extract them."
            )

        archive_path = self.raw_dir / _CIFAR100_ARCHIVE
        expected_sha256 = _CIFAR_ARCHIVE_SHA256[_CIFAR100_ARCHIVE]

        last_err: Exception | None = None
        for base_url in _CIFAR100_BASE_URLS:
            try:
                url = base_url + _CIFAR100_ARCHIVE
                download_url(url, archive_path, expected_sha256=expected_sha256)
                _extract_tar_gz(archive_path, self.raw_dir)
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise RuntimeError(
                f"Failed to download {_CIFAR100_ARCHIVE}: {last_err}"
            ) from last_err

        _ = _ensure_extracted_folder(self.raw_dir, folder_name=_CIFAR100_FOLDER)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieve a single (image, label) pair.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        (Any, Any)
            Image and label pair. By default, the image is a NumPy array of
            shape (3, 32, 32) and dtype `self.dtype`, and the label is an int.
        """
        x = self.images[idx]  # uint8 (3,32,32)
        y = int(self.labels[idx])

        x_f = x.astype(self.dtype) / 255.0

        if self.normalize:
            x_f = (x_f - self._mean.reshape(3, 1, 1)) / self._std.reshape(3, 1, 1)

        if self.transform is not None:
            x_f = self.transform(x_f)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x_f, y


def download_cifar100(root: str | Path) -> Path:
    """
    Convenience helper to download and extract CIFAR-100.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.

    Returns
    -------
    Path
        Path to the CIFAR-100 raw data directory.
    """
    ds = CIFAR100(root=root, train=True, download=True)
    _ = len(ds)
    return Path(root).expanduser().resolve() / "cifar100" / "raw"
