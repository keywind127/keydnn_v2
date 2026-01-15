# src/keydnn/infrastructure/datasets/_mnist.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any, List

import numpy as np

from ._download import download_url
from ._idx import load_idx_images_gz, load_idx_labels_gz


_MNIST_BASE_URLS: List[str] = [
    "http://yann.lecun.com/exdb/mnist/",
    # Optional: add mirrors here
]

# You should fill these with real SHA256 values for each .gz file.
# (Compute once and pin them; it’s worth it.)
_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": None,
    "train-labels-idx1-ubyte.gz": None,
    "t10k-images-idx3-ubyte.gz": None,
    "t10k-labels-idx1-ubyte.gz": None,
}


def _expand_root(root: Union[str, Path]) -> Path:
    return Path(root).expanduser().resolve()


@dataclass
class MNIST:
    root: Union[str, Path]
    train: bool = True
    download: bool = False
    transform: Optional[Callable[[Any], Any]] = None
    target_transform: Optional[Callable[[Any], Any]] = None
    normalize: bool = False
    return_numpy: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.root = _expand_root(self.root)
        self.raw_dir = self.root / "mnist" / "raw"

        self._ensure_data()

        if self.train:
            img_name = "train-images-idx3-ubyte.gz"
            lbl_name = "train-labels-idx1-ubyte.gz"
        else:
            img_name = "t10k-images-idx3-ubyte.gz"
            lbl_name = "t10k-labels-idx1-ubyte.gz"

        images_u8 = load_idx_images_gz(self.raw_dir / img_name)
        labels_u8 = load_idx_labels_gz(self.raw_dir / lbl_name)

        # Store canonical forms
        self.images = images_u8  # uint8 (N,28,28)
        self.labels = labels_u8  # uint8 (N,)

        # Precompute normalization constants if requested
        # MNIST common mean/std (in [0,1]) ~ 0.1307 / 0.3081 (optional)
        self._mean = 0.1307
        self._std = 0.3081

    def _ensure_data(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        missing = []
        for fname in _MNIST_FILES.keys():
            if not (self.raw_dir / fname).exists():
                missing.append(fname)

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
                    url = base + fname
                    download_url(
                        url, self.raw_dir / fname, expected_sha256=expected_sha256
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
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.images[idx]  # uint8 (28,28)
        y = int(self.labels[idx])

        # Convert to float in [0,1]
        x_f = x.astype(self.dtype) / 255.0

        if self.normalize:
            x_f = (x_f - self._mean) / self._std

        # Add channel dim (1,28,28) if you want CNN-friendly default
        x_f = np.expand_dims(x_f, axis=0)

        if self.transform is not None:
            x_f = self.transform(x_f)
        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.return_numpy:
            return x_f, y

        # Optional: convert to your Tensor type here
        # from ..tensor import Tensor
        # return Tensor.from_numpy(x_f), y
        return x_f, y


def download_mnist(root: str | Path) -> Path:
    ds = MNIST(root=root, train=True, download=True)
    _ = len(ds)
    return Path(root).expanduser().resolve() / "mnist" / "raw"
