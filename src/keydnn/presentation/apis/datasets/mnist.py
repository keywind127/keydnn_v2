"""
MNIST dataset convenience loaders.

This module provides a high-level, user-friendly interface for working with
the MNIST dataset in KeyDNN. It wraps the lower-level :class:`MNIST` dataset
implementation and exposes a simplified API with sensible defaults for
common use cases.

Design goals
------------
- Hide dataset root and download mechanics from end users.
- Provide a stable, presentation-layer entry point for loading MNIST.
- Default to safe, predictable behavior (automatic download when missing).
- Remain thin and explicit, delegating all dataset logic to the core
  infrastructure implementation.

Intended usage
--------------
This module is intended to be imported by user code, examples, and tutorials:

>>> from keydnn.datasets import load_mnist
>>> train_ds = load_mnist(train=True, normalize=True)
>>> test_ds = load_mnist(train=False)

Advanced users may still import and instantiate :class:`MNIST` directly
from the infrastructure layer when finer-grained control is required.
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Union
from pathlib import Path


from ....infrastructure.datasets._mnist import (
    download_mnist,
    download_url,
    MNIST,
)


def load_mnist(
    *,
    train: bool = True,
    transform: Optional[Callable[[Any], Any]] = None,
    target_transform: Optional[Callable[[Any], Any]] = None,
    normalize: bool = True,
    return_numpy: bool = False,
    dtype: str = "float32",
    root_path: Union[str, Path] = "data",
) -> MNIST:
    """
    Load the MNIST dataset with sensible defaults.

    This is a thin convenience wrapper around :class:`MNIST` that:
    - Uses a default dataset root (`data/`)
    - Automatically downloads the dataset if it is not already present
    - Exposes only the commonly tuned parameters in the public API

    Parameters
    ----------
    train : bool, default=True
        Whether to load the training split. If False, loads the test split.
    transform : callable, optional
        Optional transform applied to each image.
    target_transform : callable, optional
        Optional transform applied to each label.
    normalize : bool, default=True
        Whether to apply standard MNIST mean/std normalization.
    return_numpy : bool, default=False
        Whether to return NumPy arrays. If False, callers may wrap outputs
        into tensors downstream.
    dtype : str, default="float32"
        Floating-point dtype used for image conversion.
    root_path : str or Path, default="data"
        Base directory for dataset storage. The MNIST files are stored under
        `<root_path>/mnist/raw`.

    Returns
    -------
    MNIST
        An initialized MNIST dataset instance.
    """
    root_path = Path(root_path).expanduser().resolve()
    raw_dir = root_path / "mnist" / "raw"

    download = not raw_dir.exists()

    return MNIST(
        root=root_path,
        train=train,
        download=download,
        transform=transform,
        target_transform=target_transform,
        normalize=normalize,
        return_numpy=return_numpy,
        dtype=dtype,
    )


__all__ = [
    "download_mnist",
    "download_url",
    "MNIST",
    "load_mnist",
]
