"""
CIFAR dataset convenience loaders.

This module provides high-level, user-facing helpers for loading the
CIFAR-10 and CIFAR-100 vision datasets with sensible defaults. It wraps
the lower-level infrastructure implementations and exposes a simplified
API suitable for examples, tutorials, and quick experiments.

Design goals
------------
- Hide dataset root and download mechanics from end users.
- Automatically download datasets when missing.
- Provide a stable presentation-layer API with minimal parameters.
- Delegate all dataset parsing, caching, and preprocessing to the
  infrastructure dataset implementations.

Datasets
--------
- CIFAR-10: 32x32 RGB images, 10 classes
- CIFAR-100: 32x32 RGB images, 100 classes

Advanced users may instantiate :class:`CIFAR10` or :class:`CIFAR100`
directly from the infrastructure layer when finer-grained control is
required.
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Union
from pathlib import Path


from ....infrastructure.datasets._cifar import (
    CIFAR10,
    CIFAR100,
    download_cifar10,
    download_cifar100,
)


def load_cifar10(
    *,
    train: bool = True,
    transform: Optional[Callable[[Any], Any]] = None,
    target_transform: Optional[Callable[[Any], Any]] = None,
    normalize: bool = True,
    return_numpy: bool = False,
    dtype: str = "float32",
    root_path: Union[str, Path] = "data",
) -> CIFAR10:
    """
    Load the CIFAR-10 dataset with sensible defaults.

    This is a thin convenience wrapper around :class:`CIFAR10` that:
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
        Whether to apply standard CIFAR-10 mean/std normalization.
    return_numpy : bool, default=False
        Whether to return NumPy arrays. If False, callers may wrap outputs
        into tensors downstream.
    dtype : str, default="float32"
        Floating-point dtype used for image conversion.
    root_path : str or Path, default="data"
        Base directory for dataset storage. The CIFAR-10 files are stored
        under `<root_path>/cifar10/raw`.

    Returns
    -------
    CIFAR10
        An initialized CIFAR-10 dataset instance.
    """
    root_path = Path(root_path).expanduser().resolve()
    raw_dir = root_path / "cifar10" / "raw"

    download = not raw_dir.exists()

    return CIFAR10(
        root=root_path,
        train=train,
        download=download,
        transform=transform,
        target_transform=target_transform,
        normalize=normalize,
        return_numpy=return_numpy,
        dtype=dtype,
    )


def load_cifar100(
    *,
    train: bool = True,
    transform: Optional[Callable[[Any], Any]] = None,
    target_transform: Optional[Callable[[Any], Any]] = None,
    normalize: bool = True,
    return_numpy: bool = False,
    dtype: str = "float32",
    root_path: Union[str, Path] = "data",
) -> CIFAR100:
    """
    Load the CIFAR-100 dataset with sensible defaults.

    This is a thin convenience wrapper around :class:`CIFAR100` that:
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
        Whether to apply standard CIFAR-100 mean/std normalization.
    return_numpy : bool, default=False
        Whether to return NumPy arrays. If False, callers may wrap outputs
        into tensors downstream.
    dtype : str, default="float32"
        Floating-point dtype used for image conversion.
    root_path : str or Path, default="data"
        Base directory for dataset storage. The CIFAR-100 files are stored
        under `<root_path>/cifar100/raw`.

    Returns
    -------
    CIFAR100
        An initialized CIFAR-100 dataset instance.
    """
    root_path = Path(root_path).expanduser().resolve()
    raw_dir = root_path / "cifar100" / "raw"

    download = not raw_dir.exists()

    return CIFAR100(
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
    "CIFAR10",
    "CIFAR100",
    "download_cifar10",
    "download_cifar100",
    "load_cifar10",
    "load_cifar100",
]
