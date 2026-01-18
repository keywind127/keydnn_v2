"""
MNIST dataset loader.

This module provides a lightweight, self-contained implementation of the
MNIST dataset, including automatic download, caching, IDX parsing, and
basic preprocessing.

Design notes
------------
- Dataset files are cached under `<root>/mnist/raw`.
- Network access is optional and explicitly controlled via `download=True`.
- Data is returned as NumPy arrays by default to avoid tight coupling with
  any specific Tensor implementation.
- Normalization and transformation hooks mirror common deep learning APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, List

import numpy as np

from ._download import download_url
from ._base import _VerboseMixin
from ._idx import load_idx_images_gz, load_idx_labels_gz


_MNIST_BASE_URLS: List[str] = [
    # Reliable mirrors first
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    # Keep the original host last (can be flaky)
    "http://yann.lecun.com/exdb/mnist/",
]


_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": None,
    "train-labels-idx1-ubyte.gz": None,
    "t10k-images-idx3-ubyte.gz": None,
    "t10k-labels-idx1-ubyte.gz": None,
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


@dataclass
class MNIST(_VerboseMixin):
    """
    MNIST handwritten digit dataset.

    This class provides indexed access to the MNIST dataset, supporting
    automatic download, caching, normalization, and user-defined transforms.

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
        Whether to apply standard MNIST mean/std normalization.
    return_numpy : bool, optional
        Whether to return NumPy arrays (default). Tensor conversion can be
        layered on top by the caller.
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
        self.raw_dir = self.root / "mnist" / "raw"

        self._ensure_data()

        if self.train:
            img_name = "train-images-idx3-ubyte.gz"
            lbl_name = "train-labels-idx1-ubyte.gz"
        else:
            img_name = "t10k-images-idx3-ubyte.gz"
            lbl_name = "t10k-labels-idx1-ubyte.gz"

        self.images = load_idx_images_gz(self.raw_dir / img_name)
        self.labels = load_idx_labels_gz(self.raw_dir / lbl_name)

        # MNIST mean/std in [0,1] space
        self._mean = 0.1307
        self._std = 0.3081

    def _ensure_data(self) -> None:
        """
        Ensure all required MNIST files are present locally.

        Downloads missing files when `download=True`.

        Raises
        ------
        FileNotFoundError
            If files are missing and downloading is disabled.
        RuntimeError
            If a download fails from all configured mirrors.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        missing = [
            fname for fname in _MNIST_FILES if not (self.raw_dir / fname).exists()
        ]

        if not missing:
            return

        if not self.download:
            raise FileNotFoundError(
                f"MNIST files missing in {self.raw_dir}. "
                f"Set download=True to fetch them."
            )

        for fname in missing:
            expected_sha256 = _MNIST_FILES[fname]
            last_err = None
            for base in _MNIST_BASE_URLS:
                try:
                    download_url(
                        base + fname,
                        self.raw_dir / fname,
                        expected_sha256=expected_sha256,
                        verbose=self._verbose,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e

            if last_err is not None:
                raise RuntimeError(
                    f"Failed to download {fname}: {last_err}"
                ) from last_err

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
            shape (1, 28, 28) and dtype `self.dtype`, and the label is an int.
        """
        x = self.images[idx]
        y = int(self.labels[idx])

        x_f = x.astype(self.dtype) / 255.0

        if self.normalize:
            x_f = (x_f - self._mean) / self._std

        x_f = np.expand_dims(x_f, axis=0)

        if self.transform is not None:
            x_f = self.transform(x_f)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x_f, y


def download_mnist(root: str | Path) -> Path:
    """
    Convenience helper to download the MNIST dataset.

    Parameters
    ----------
    root : str or Path
        Root directory for dataset storage.

    Returns
    -------
    Path
        Path to the MNIST raw data directory.
    """
    ds = MNIST(root=root, train=True, download=True)
    _ = len(ds)
    return Path(root).expanduser().resolve() / "mnist" / "raw"
