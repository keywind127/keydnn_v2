"""
MNIST CNN smoke-train script (Conv2d + BatchNorm2d + ReLU + Pool2d + GlobalAvgPool2d
+ Linear + Dropout + Softmax).

This is a *demo / integration test* script intended to:
- exercise a realistic CNN block stack
- verify forward/backward wiring across common layers
- run quickly on a limited MNIST subset (configurable)

Usage
-----
# CPU
python scripts/train_mnist_cnn_bn.py --device cpu --epochs 3

# CUDA (if available)
python scripts/train_mnist_cnn_bn.py --device cuda:0 --epochs 3

# Fast smoke test
python scripts/train_mnist_cnn_bn.py --device cuda:0 --epochs 1 --limit-train 2048 --limit-test 1024
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from keydnn.tensors import Tensor


# --------------------------------------------------------------------------------------
# Utilities (device + tensor conversion)
# --------------------------------------------------------------------------------------


def _cuda_available() -> bool:
    """
    Return True if KeyDNN CUDA native wrappers appear loadable.
    """
    try:
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _device_from_string(device_str: str):
    """
    Create a KeyDNN Device from a string like "cpu" or "cuda:0".
    """
    from keydnn.domain.device._device import Device

    return Device(device_str)


def _tensor_from_numpy(arr: np.ndarray, *, device, requires_grad: bool = False):
    """
    Create a KeyDNN Tensor from a NumPy array using public APIs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    device : Device
        Target device.
    requires_grad : bool
        Whether the created tensor should track gradients.

    Returns
    -------
    Tensor
        A KeyDNN Tensor populated from NumPy.
    """
    from keydnn.presentation.apis.tensors import Tensor  # preferred stable import

    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device, requires_grad=requires_grad)
    t.copy_from_numpy(a)
    return t


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels (N,) to one-hot float32 matrix (N, C).
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _accuracy_from_probs_np(y_true_int: np.ndarray, probs: np.ndarray) -> float:
    """
    Accuracy by argmax(probs) vs integer labels.
    """
    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    pred = np.argmax(np.asarray(probs), axis=1).astype(np.int64)
    return float((pred == y_true_int).mean())


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def _build_mnist_cnn(device):
    """
    Build a demo-friendly CNN for MNIST using the requested layer stack.

    Stack (one reasonable ordering):
    - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    - GlobalAvgPool2d
    - Dropout
    - Linear
    - Softmax

    Notes
    -----
    - MNIST input is (N,1,28,28).
    - After two 2x2 pools: 28 -> 14 -> 7.
    - GlobalAvgPool2d reduces spatial dims to (N, C) or (N, C, 1, 1)
      We add Flatten to be safe if it returns (N, C, 1, 1).
    """
    from keydnn.presentation.apis.models import Sequential

    # Layers (prefer presentation APIs; fall back to infrastructure if needed)
    try:
        from keydnn.presentation.apis.layers import (
            Conv2d,
            MaxPool2d,
            GlobalAvgPool2d,
            Linear,
            Flatten,
            BatchNorm2d,
            Dropout,
        )
    except Exception:
        from keydnn.infrastructure.convolution._conv2d_module import Conv2d  # type: ignore
        from keydnn.infrastructure.pooling._pooling_module import MaxPool2d, GlobalAvgPool2d  # type: ignore
        from keydnn.infrastructure.fully_connected._linear import Linear  # type: ignore
        from keydnn.infrastructure.flatten._flatten_module import Flatten  # type: ignore
        from keydnn.infrastructure.layers._batchnorm import BatchNorm2d  # type: ignore
        from keydnn.infrastructure.layers._dropout import Dropout  # type: ignore

    try:
        from keydnn.presentation.apis.activations import ReLU, Softmax
    except Exception:
        from keydnn.infrastructure.activations._modules import ReLU, Softmax  # type: ignore

    # ------------------------------------------------------------------
    # Helper: try to pass device, otherwise fall back
    # ------------------------------------------------------------------

    def _maybe_device(ctor, *args, **kwargs):
        try:
            return ctor(*args, device=device, **kwargs)
        except TypeError:
            return ctor(*args, **kwargs)

    return Sequential(
        # --------------------------------------------------------------
        # Block 1
        # --------------------------------------------------------------
        _maybe_device(Conv2d, 1, 16, kernel_size=3, stride=1, padding=1),
        _maybe_device(BatchNorm2d, 16),
        _maybe_device(ReLU),
        _maybe_device(MaxPool2d, kernel_size=2, stride=2),
        # --------------------------------------------------------------
        # Block 2
        # --------------------------------------------------------------
        _maybe_device(Conv2d, 16, 32, kernel_size=3, stride=1, padding=1),
        _maybe_device(BatchNorm2d, 32),
        _maybe_device(ReLU),
        _maybe_device(MaxPool2d, kernel_size=2, stride=2),
        # --------------------------------------------------------------
        # Head
        # --------------------------------------------------------------
        # _maybe_device(GlobalAvgPool2d),
        _maybe_device(Flatten),
        _maybe_device(Dropout, p=0.2),
        _maybe_device(Linear, 1568, 10),
        _maybe_device(Softmax, axis=1),
    )


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """
    CLI configuration for this script.
    """

    root: str = "~/.cache/keydnn"
    device: str = "cpu"
    epochs: int = 3
    batch_size: int = 128
    lr: float = 0.1
    seed: int = 0
    shuffle: bool = True
    limit_train: int = 0
    limit_test: int = 0
    verbose: int = 1


# --------------------------------------------------------------------------------------
# Main training
# --------------------------------------------------------------------------------------


def main() -> int:
    """
    Script entrypoint.
    """
    ap = argparse.ArgumentParser(prog="train_mnist_cnn_bn", description=__doc__)
    ap.add_argument("--root", type=str, default="~/.cache/keydnn_v2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = Config(
        root=args.root,
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        shuffle=not bool(args.no_shuffle),
        limit_train=int(args.limit_train),
        limit_test=int(args.limit_test),
        verbose=int(args.verbose),
    )

    if cfg.device.startswith("cuda") and not _cuda_available():
        raise RuntimeError(
            f"Requested device={cfg.device}, but CUDA wrappers are not available."
        )

    device = _device_from_string(cfg.device)

    try:
        from keydnn.presentation.apis.datasets.mnist import MNIST  # type: ignore
    except Exception:
        from keydnn.infrastructure.datasets._mnist import MNIST  # type: ignore

    from keydnn.presentation.apis.optimizers import SGD  # preferred stable import

    ds_train = MNIST(
        root=cfg.root,
        train=True,
        download=True,
        normalize=False,
        return_numpy=True,
        dtype="float32",
    )
    ds_test = MNIST(
        root=cfg.root,
        train=False,
        download=True,
        normalize=False,
        return_numpy=True,
        dtype="float32",
    )

    # Materialize into NumPy (simple + deterministic; OK for demo)
    n_train = len(ds_train)
    n_test = len(ds_test)
    if cfg.limit_train > 0:
        n_train = min(n_train, cfg.limit_train)
    if cfg.limit_test > 0:
        n_test = min(n_test, cfg.limit_test)

    x_train = np.stack([ds_train[i][0] for i in range(n_train)], axis=0)  # (N,1,28,28)
    y_train_int = np.array([ds_train[i][1] for i in range(n_train)], dtype=np.int64)

    x_test = np.stack([ds_test[i][0] for i in range(n_test)], axis=0)
    y_test_int = np.array([ds_test[i][1] for i in range(n_test)], dtype=np.int64)

    # One-hot targets for MSE demo loss
    y_train_oh = _one_hot(y_train_int, 10)
    y_test_oh = _one_hot(y_test_int, 10)

    model = _build_mnist_cnn(device)
    opt = SGD(model.parameters(), lr=float(cfg.lr))

    def acc_metric(y_true_batch, y_pred_batch):
        """
        Batch accuracy metric compatible with `train_on_batch`.

        This computes argmax over predicted probabilities/logits and compares
        with integer labels derived from one-hot targets.
        """
        yp = np.asarray(y_pred_batch.to_numpy(), dtype=np.float32)
        yt = np.asarray(y_true_batch.to_numpy(), dtype=np.float32)
        y_int = np.argmax(yt, axis=1).astype(np.int64)
        return _accuracy_from_probs_np(y_int, yp)

    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Train: {x_train.shape} | Test: {x_test.shape}")
        print(
            f"epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} shuffle={cfg.shuffle}"
        )
        print("Loss: CCE(one-hot) | Metric: acc(argmax softmax)")

    # Train using Model.fit (so Dropout/BN training-mode semantics are exercised)
    x_t = _tensor_from_numpy(x_train, device=device, requires_grad=False)
    y_t = _tensor_from_numpy(y_train_oh, device=device, requires_grad=False)

    x_val_t = _tensor_from_numpy(x_test, device=device, requires_grad=False)
    y_val_t = _tensor_from_numpy(y_test_oh, device=device, requires_grad=False)

    model.build(x_t[:1])  # ensure built before training

    history = model.fit(
        x_t,
        y_t,
        loss="cce",
        optimizer=opt,
        metrics=[acc_metric],
        metric_names=["acc"],
        batch_size=int(cfg.batch_size),
        epochs=int(cfg.epochs),
        shuffle=bool(cfg.shuffle),
        verbose=int(cfg.verbose),
        validation_data=(x_val_t, y_val_t),
    )

    # Quick eval (forward only)
    x_eval = _tensor_from_numpy(x_test, device=device, requires_grad=False)
    probs = model(x_eval)
    probs_np = np.asarray(probs.to_numpy(), dtype=np.float32)
    test_acc = _accuracy_from_probs_np(y_test_int, probs_np)

    print(f"Final test_acc={test_acc:.4f}")

    # from keydnn.infrastructure.tensor._cuda_memory_pool import GLOBAL_CUDA_MEMORY_POOL

    # print(GLOBAL_CUDA_MEMORY_POOL.stats()[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
