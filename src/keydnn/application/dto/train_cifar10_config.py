"""
DTOs for application-layer training examples.

This module defines configuration objects used by application-layer use cases
(e.g., runnable training demos invoked from a CLI). These objects form part of
the application boundary and should remain stable even if the presentation
layer (CLI flags, argument names) changes.

Design notes
------------
- DTOs should be simple, serializable, and free of infrastructure dependencies.
- They serve as a contract between presentation (argument parsing) and
  application (use case orchestration).
- Defaults are chosen to provide a fast, reasonable demo experience.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainCifar10Config:
    """
    Configuration for the CIFAR-10 CNN training example use case.

    This DTO is part of the application layer boundary and should remain stable
    even if the CLI presentation changes. It is intentionally lightweight and
    should not import or depend on infrastructure modules.

    Parameters
    ----------
    root : str
        Dataset cache root directory. The CIFAR-10 dataset will be stored under
        `<root>/cifar10/raw` by the infrastructure dataset loader.
    device : str
        Device selector string, e.g. "cpu" or "cuda:0".
    epochs : int
        Number of training epochs to run.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for the optimizer.
    seed : int
        Random seed used for shuffling / reproducible batching.
    shuffle : bool
        Whether to shuffle the training set at the start of each epoch.
    limit_train : int
        Optional cap on the number of training examples used (0 means all).
        Useful for quick smoke tests.
    limit_test : int
        Optional cap on the number of test examples used (0 means all).
        Useful for quick smoke tests.
    normalize : bool
        Whether to apply dataset normalization. If True, the use case may apply
        common CIFAR-10 mean/std normalization in [0,1] space.
    verbose : int
        Verbosity level (0 for quiet). The use case may print training progress
        and evaluation metrics when verbose is non-zero.
    """

    root: str = "~/.cache/keydnn"
    device: str = "cpu"
    epochs: int = 2
    batch_size: int = 128
    lr: float = 0.1
    seed: int = 0
    shuffle: bool = True
    limit_train: int = 0
    limit_test: int = 0
    normalize: bool = False
    verbose: int = 1
