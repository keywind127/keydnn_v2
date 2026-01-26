# src/keydnn/application/examples/train_mnist_mlp.py
"""
MNIST MLP training example (application use case).

This module implements an application-layer use case that trains a simple MLP
on MNIST using KeyDNN infrastructure components (dataset loader, Tensor,
modules, and optimizers). It is intended to be callable from the presentation
layer (e.g., `python -m keydnn test --train_mnist_example`) as a runnable demo
or smoke test.

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

from ..dto.train_mnist_config import TrainMnistConfig

from ...infrastructure.random import _seed as random_seed
from ...infrastructure.random import _determinism as determinism


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
        Input array to copy into a KeyDNN Tensor. It will be converted to
        float32 if needed.
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
        Number of classes. Defaults to 10 for MNIST.

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
        Input features of shape (N, D).
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


def _build_mlp(device, *, hidden_dim: int):
    """
    Build a simple MLP model for MNIST.

    The model maps a flattened MNIST input vector (784) to 10 output logits
    using a single hidden layer.

    Parameters
    ----------
    device : Device
        Target device used for device-aware module construction on CUDA.
    hidden_dim : int
        Hidden layer width.

    Returns
    -------
    Module
        A KeyDNN Sequential model instance.
    """
    from ...infrastructure.models._sequential import Sequential
    from ...infrastructure.fully_connected._linear import Linear

    # Prefer ReLU; fallback to Sigmoid if not available.
    try:
        from ...infrastructure.activations._modules import ReLU  # type: ignore

        act = ReLU()
    except Exception:
        from ...infrastructure.activations._modules import Sigmoid  # type: ignore

        act = Sigmoid()

    if str(device).startswith("cuda"):
        return Sequential(
            Linear(784, hidden_dim, device=device),
            act,
            Linear(hidden_dim, 10, device=device),
        )

    return Sequential(
        Linear(784, hidden_dim),
        act,
        Linear(hidden_dim, 10),
    )


def run_train_mnist_mlp(cfg: TrainMnistConfig) -> int:
    """
    Train a simple MLP on MNIST (application use case).

    This function orchestrates:
    - dataset download/loading (infrastructure)
    - model construction (infrastructure)
    - optimization loop using `train_on_batch` (infrastructure)
    - periodic evaluation on the test split

    Parameters
    ----------
    cfg : TrainMnistConfig
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

    random_seed.seed(int(cfg.seed))
    determinism.set_deterministic(True, cpu_threads=4)

    device = _device_from_string(cfg.device)

    from ...infrastructure.datasets._mnist import MNIST
    from ...infrastructure.optimizers._sgd import SGD

    # Load MNIST (NumPy by default)
    ds_train = MNIST(
        root=cfg.root, train=True, download=True, normalize=False, return_numpy=True
    )
    ds_test = MNIST(
        root=cfg.root, train=False, download=True, normalize=False, return_numpy=True
    )

    # Materialize for simplicity
    x_train = np.stack(
        [ds_train[i][0] for i in range(len(ds_train))], axis=0
    )  # (N,1,28,28)
    y_train = np.array([ds_train[i][1] for i in range(len(ds_train))], dtype=np.int64)

    x_test = np.stack([ds_test[i][0] for i in range(len(ds_test))], axis=0)
    y_test = np.array([ds_test[i][1] for i in range(len(ds_test))], dtype=np.int64)

    if cfg.limit_train > 0:
        x_train = x_train[: cfg.limit_train]
        y_train = y_train[: cfg.limit_train]
    if cfg.limit_test > 0:
        x_test = x_test[: cfg.limit_test]
        y_test = y_test[: cfg.limit_test]

    # Flatten to (N,784)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    # One-hot for MSE
    y_train_oh = _one_hot(y_train, 10)
    y_test_oh = _one_hot(y_test, 10)

    model = _build_mlp(device, hidden_dim=cfg.hidden_dim)
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
            f"MLP: 784 -> {cfg.hidden_dim} -> 10 | lr={cfg.lr} | "
            f"batch={cfg.batch_size} | epochs={cfg.epochs}"
        )
        print("Loss: MSE(one-hot) | Metric: acc(argmax logits)")

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
