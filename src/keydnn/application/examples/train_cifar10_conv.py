"""
CIFAR-10 CNN training example (application use case).

This module implements an application-layer use case that trains a small
convolutional network on CIFAR-10 using KeyDNN infrastructure components
(dataset loader, Tensor, modules, and optimizers). It is intended to be callable
from the presentation layer (e.g., `python -m keydnn test --train_cifar10_example`)
as a runnable demo or smoke test.

Design notes
------------
- This is orchestration logic: it composes infrastructure pieces but does not
  implement kernels or low-level tensor operations.
- The default loss used here is MSE on one-hot labels to minimize required
  dependencies on specialized loss implementations (e.g., softmax-cross-entropy).
- The implementation prioritizes clarity and portability over maximal training
  performance. It materializes the dataset into NumPy arrays for simplicity.
- Exit codes follow conventional CLI semantics (0 for success; exceptions for
  invalid configuration or missing runtime support).
"""

from __future__ import annotations

import time
from typing import Iterable, Tuple

import numpy as np

from ..dto.train_cifar10_config import TrainCifar10Config


def _cuda_available() -> bool:
    """
    Return True if KeyDNN CUDA native wrappers appear loadable.

    This is a lightweight runtime capability check used to provide a clear,
    early error message when the user requests a CUDA device but CUDA support
    is not present in the current environment.

    Returns
    -------
    bool
        True if CUDA wrappers can be imported and initialized; otherwise False.
    """
    try:
        from ...infrastructure.native_cuda.python._native_loader import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _device_from_string(device_str: str):
    """
    Create a KeyDNN Device from a device selector string.

    Parameters
    ----------
    device_str : str
        Device selector (e.g., "cpu", "cuda:0").

    Returns
    -------
    Device
        A KeyDNN Device instance corresponding to the requested device.
    """
    from ...domain.device._device import Device

    return Device(device_str)


def _tensor_from_numpy(arr: np.ndarray, *, device):
    """
    Create a KeyDNN Tensor from a NumPy array using public APIs.

    The resulting tensor is allocated on the requested device and populated
    via `copy_from_numpy`, which provides a controlled host-to-device boundary
    when using CUDA.

    Parameters
    ----------
    arr : np.ndarray
        Input array to copy into a KeyDNN Tensor.
    device : Device
        Target device.

    Returns
    -------
    Tensor
        KeyDNN Tensor containing the copied data.
    """
    from ...infrastructure.tensor._tensor import Tensor

    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device)
    t.copy_from_numpy(a)
    return t


def _as_float(x) -> float:
    """
    Extract a Python float from numbers / numpy / Tensor-like objects.

    Accepts:
    - Python scalars (int/float)
    - NumPy scalars or arrays containing a single value
    - Tensor-like objects exposing `to_numpy()`

    Parameters
    ----------
    x : Any
        Scalar-like value.

    Returns
    -------
    float
        Extracted scalar as a Python float.
    """
    if isinstance(x, (int, float)):
        return float(x)

    if hasattr(x, "to_numpy"):
        v = np.asarray(x.to_numpy())
        return float(v.reshape(-1)[0])

    v = np.asarray(x)
    return float(v.reshape(-1)[0])


def _mse_loss(pred, target):
    """
    Mean squared error loss: mean((pred - target)^2).

    This helper is compatible with KeyDNN Tensor math as long as the involved
    Tensor type supports subtraction, multiplication, and a reduction method
    (`mean()` preferred, otherwise `sum()`).

    Parameters
    ----------
    pred : Tensor-like
        Predicted output (typically shape (N, C)).
    target : Tensor-like
        Target output (typically shape (N, C), one-hot encoded).

    Returns
    -------
    Tensor-like
        Scalar loss.

    Raises
    ------
    AttributeError
        If the Tensor type does not implement `mean()` or `sum()`.
    """
    diff = pred - target
    sq = diff * diff
    if hasattr(sq, "mean"):
        return sq.mean()
    if hasattr(sq, "sum"):
        return sq.sum() * (1.0 / target.shape[0])
    raise AttributeError("Tensor must implement mean() or sum()")


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer class labels to a one-hot float32 matrix.

    Parameters
    ----------
    labels : np.ndarray
        Integer labels of shape (N,).
    num_classes : int, optional
        Number of classes. Defaults to 10 for CIFAR-10.

    Returns
    -------
    np.ndarray
        One-hot matrix of shape (N, num_classes) with dtype float32.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _accuracy_from_logits_np(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Compute classification accuracy from logits using argmax.

    Parameters
    ----------
    y_true : np.ndarray
        Integer labels of shape (N,).
    logits : np.ndarray
        Model outputs of shape (N, C). Values need not be normalized.

    Returns
    -------
    float
        Fraction of correct predictions in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.argmax(np.asarray(logits), axis=1).astype(np.int64)
    return float((pred == y_true).mean())


def _iter_minibatches(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield mini-batches from NumPy arrays.

    Parameters
    ----------
    x_np : np.ndarray
        Input features of shape (N, C, H, W) or (N, D).
    y_np : np.ndarray
        Targets of shape (N, ...) aligned with x_np.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle before iterating.
    seed : int
        PRNG seed used when shuffling.

    Yields
    ------
    (np.ndarray, np.ndarray)
        A pair (x_batch, y_batch).
    """
    n = x_np.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield x_np[j], y_np[j]


def _maybe_normalize_cifar10(x: np.ndarray) -> np.ndarray:
    """
    Optionally apply common CIFAR-10 normalization in [0,1] space.

    CIFAR-10 mean/std (per channel) often used:
      mean = (0.4914, 0.4822, 0.4465)
      std  = (0.2470, 0.2435, 0.2616)

    Parameters
    ----------
    x : np.ndarray
        Input images (N, 3, 32, 32) float32 in [0,1].

    Returns
    -------
    np.ndarray
        Normalized images.
    """
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def _build_cnn(device):
    """
    Build a small CNN for CIFAR-10.

    Architecture (simple, demo-friendly):
    - Conv(3->16, 3x3, pad=1) -> ReLU -> MaxPool(2)
    - Conv(16->32, 3x3, pad=1) -> ReLU -> MaxPool(2)
    - Flatten
    - Linear(32*8*8 -> 128) -> ReLU
    - Linear(128 -> 10)

    Parameters
    ----------
    device : Device
        Target device used for device-aware module construction on CUDA.

    Returns
    -------
    Module
        A KeyDNN Sequential model instance.
    """
    from ...infrastructure.models._sequential import Sequential

    # Layers
    from ...infrastructure.convolution._conv2d_module import Conv2d  # type: ignore
    from ...infrastructure.pooling._pooling_module import MaxPool2d  # type: ignore
    from ...infrastructure.flatten._flatten_module import Flatten  # type: ignore
    from ...infrastructure.fully_connected._linear import Linear  # type: ignore

    # Prefer ReLU; fallback to Sigmoid if not available.
    try:
        from ...infrastructure.activations._modules import ReLU  # type: ignore

        act = ReLU()
        act2 = ReLU()
    except Exception:
        from ...infrastructure.activations._modules import Sigmoid  # type: ignore

        act = Sigmoid()
        act2 = Sigmoid()

    use_cuda = str(device).startswith("cuda")

    def _conv(in_ch, out_ch):
        if use_cuda:
            return Conv2d(
                in_ch, out_ch, kernel_size=3, stride=1, padding=1, device=device
            )
        return Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def _linear(in_f, out_f):
        if use_cuda:
            return Linear(in_f, out_f, device=device)
        return Linear(in_f, out_f)

    return Sequential(
        _conv(3, 16),
        act,
        MaxPool2d(kernel_size=2, stride=2),
        _conv(16, 32),
        act2,
        MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        _linear(32 * 8 * 8, 128),
        act2,
        _linear(128, 10),
    )


def run_train_cifar10_conv(cfg: TrainCifar10Config) -> int:
    """
    Train a small CNN on CIFAR-10 (application use case).

    This function orchestrates:
    - dataset download/loading (infrastructure)
    - model construction (infrastructure)
    - optimization loop using `train_on_batch` (infrastructure)
    - periodic evaluation on the test split

    Parameters
    ----------
    cfg : TrainCifar10Config
        Use case configuration (application boundary DTO).

    Returns
    -------
    int
        Process exit code (0 for success).

    Raises
    ------
    RuntimeError
        If CUDA is requested but unavailable in the current environment.
    """
    if cfg.device.startswith("cuda") and not _cuda_available():
        raise RuntimeError(
            f"Requested device={cfg.device}, but CUDA wrappers are not available."
        )

    device = _device_from_string(cfg.device)

    from ...infrastructure.datasets._cifar import CIFAR10
    from ...infrastructure.optimizers._sgd import SGD

    # ------------------------------------------------------------------
    # Dataset (materialized NumPy arrays, MNIST-style)
    # ------------------------------------------------------------------
    ds_train = CIFAR10(
        root=cfg.root,
        train=True,
        download=True,
        normalize=False,  # handled explicitly here
        return_numpy=True,
        dtype="float32",
    )
    ds_test = CIFAR10(
        root=cfg.root,
        train=False,
        download=True,
        normalize=False,
        return_numpy=True,
        dtype="float32",
    )

    # Direct access (NO Python loops)
    x_train = ds_train.images.astype(np.float32) / 255.0  # (N,3,32,32)
    y_train = ds_train.labels.astype(np.int64)

    x_test = ds_test.images.astype(np.float32) / 255.0
    y_test = ds_test.labels.astype(np.int64)

    if cfg.limit_train > 0:
        x_train = x_train[: cfg.limit_train]
        y_train = y_train[: cfg.limit_train]
    if cfg.limit_test > 0:
        x_test = x_test[: cfg.limit_test]
        y_test = y_test[: cfg.limit_test]

    if cfg.normalize:
        x_train = _maybe_normalize_cifar10(x_train)
        x_test = _maybe_normalize_cifar10(x_test)

    # One-hot targets for MSE
    y_train_oh = _one_hot(y_train, 10)

    # ------------------------------------------------------------------
    # Model / Optimizer
    # ------------------------------------------------------------------
    model = _build_cnn(device)
    model.build(_tensor_from_numpy(x_train[:1], device=device))
    opt = SGD(model.parameters(), lr=float(cfg.lr))

    def acc_metric(y_true_batch, y_pred_batch):
        """
        Batch accuracy metric compatible with `train_on_batch`.

        Parameters
        ----------
        y_true_batch : Tensor-like
            One-hot targets (N, 10).
        y_pred_batch : Tensor-like
            Model outputs (N, 10).

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        yp = np.asarray(y_pred_batch.to_numpy(), dtype=np.float32)
        yt = np.asarray(y_true_batch.to_numpy(), dtype=np.float32)
        y_int = np.argmax(yt, axis=1).astype(np.int64)
        return _accuracy_from_logits_np(y_int, yp)

    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")
        print(
            f"CNN CIFAR-10 | lr={cfg.lr} | batch={cfg.batch_size} | "
            f"epochs={cfg.epochs} | normalize={cfg.normalize}"
        )
        print("Loss: MSE(one-hot) | Metric: acc(argmax logits)")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        losses = []
        accs = []

        for xb_np, yb_np in _iter_minibatches(
            x_train,
            y_train_oh,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            seed=cfg.seed + epoch,
        ):
            xb = _tensor_from_numpy(xb_np, device=device)
            yb = _tensor_from_numpy(yb_np, device=device)

            logs = model.train_on_batch(
                xb,
                yb,
                loss=_mse_loss,
                optimizer=opt,
                metrics=[acc_metric],
                metric_names=["acc"],
            )

            losses.append(_as_float(logs["loss"]))
            accs.append(_as_float(logs["acc"]))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = float(np.mean(accs)) if accs else float("nan")

        xb = _tensor_from_numpy(x_test, device=device)
        logits = model(xb)
        logits_np = np.asarray(logits.to_numpy(), dtype=np.float32)
        test_acc = _accuracy_from_logits_np(y_test, logits_np)

        dt = time.time() - t0
        if cfg.verbose:
            print(
                f"Epoch {epoch:02d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_acc={test_acc:.4f} | {dt:.2f}s"
            )

    return 0
