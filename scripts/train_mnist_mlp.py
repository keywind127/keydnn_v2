from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _cuda_available() -> bool:
    """
    Return True if KeyDNN CUDA native wrappers appear loadable.
    """
    from keydnn.presentation.apis.backend.ops.cuda import cuda_available

    return cuda_available()


def _device_from_string(device_str: str):
    """
    Create a KeyDNN Device from a string like "cpu" or "cuda:0".
    """
    from keydnn.domain.device._device import Device

    return Device(device_str)


def _tensor_from_numpy(arr: np.ndarray, *, device):
    """
    Create a KeyDNN Tensor from a NumPy array using only public APIs.
    """
    from keydnn.infrastructure.tensor._tensor import Tensor

    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device)
    t.copy_from_numpy(a)
    return t


def _as_float(x) -> float:
    """
    KeyDNN-friendly scalar extraction.

    Accepts:
    - python number
    - numpy scalar / array(1,)
    - Tensor-like exposing to_numpy()
    """
    if isinstance(x, (int, float)):
        return float(x)

    if hasattr(x, "to_numpy"):
        v = x.to_numpy()
        v = np.asarray(v)
        return float(v.reshape(-1)[0])

    v = np.asarray(x)
    return float(v.reshape(-1)[0])


def _mse_loss(pred, target):
    """
    Mean-squared error loss: mean((pred - target)^2).

    Works with KeyDNN Tensor ops (sub, mul, mean or sum).
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
    Convert integer labels (N,) to one-hot (N, C).
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _accuracy_from_logits_np(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Accuracy by argmax(logits) vs y_true.
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
    Yield (x_batch, y_batch) from NumPy arrays.
    """
    n = x_np.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield x_np[j], y_np[j]


def _build_mlp(device, *, hidden_dim: int = 256):
    """
    Build a simple MLP: 784 -> hidden -> 10.

    Note:
    - If you do not have ReLU in your activations module, swap to Sigmoid().
    """
    from keydnn.infrastructure.models._sequential import Sequential
    from keydnn.infrastructure.fully_connected._linear import Linear

    try:
        from keydnn.infrastructure.activations._modules import ReLU  # type: ignore

        act = ReLU()
    except Exception:
        from keydnn.infrastructure.activations._modules import Sigmoid  # type: ignore

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a simple MLP on MNIST using KeyDNN."
    )
    parser.add_argument(
        "--root", type=str, default=str(Path("~/.cache/keydnn").expanduser())
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help='e.g. "cpu" or "cuda:0"'
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--limit-train", type=int, default=0, help="0 means use all training samples"
    )
    parser.add_argument(
        "--limit-test", type=int, default=0, help="0 means use all test samples"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    # Device selection guardrails
    if args.device.startswith("cuda") and not _cuda_available():
        raise RuntimeError(
            f"Requested device={args.device}, but KeyDNN CUDA native wrappers are not available."
        )

    device = _device_from_string(args.device)

    # Load dataset (NumPy)
    from keydnn.infrastructure.datasets._mnist import MNIST

    ds_train = MNIST(
        root=args.root, train=True, download=True, normalize=False, return_numpy=True
    )
    ds_test = MNIST(
        root=args.root, train=False, download=True, normalize=False, return_numpy=True
    )

    # Materialize to NumPy arrays for simpler batching
    x_train = np.stack(
        [ds_train[i][0] for i in range(len(ds_train))], axis=0
    )  # (N,1,28,28)
    y_train = np.array([ds_train[i][1] for i in range(len(ds_train))], dtype=np.int64)

    x_test = np.stack([ds_test[i][0] for i in range(len(ds_test))], axis=0)
    y_test = np.array([ds_test[i][1] for i in range(len(ds_test))], dtype=np.int64)

    if args.limit_train and args.limit_train > 0:
        x_train = x_train[: args.limit_train]
        y_train = y_train[: args.limit_train]

    if args.limit_test and args.limit_test > 0:
        x_test = x_test[: args.limit_test]
        y_test = y_test[: args.limit_test]

    # Flatten images: (N,1,28,28) -> (N,784)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    # One-hot targets for MSE: (N,) -> (N,10)
    y_train_oh = _one_hot(y_train, 10)
    y_test_oh = _one_hot(y_test, 10)

    # Build model + optimizer
    model = _build_mlp(device, hidden_dim=args.hidden_dim)

    from keydnn.infrastructure.optimizers._sgd import SGD

    opt = SGD(model.parameters(), lr=float(args.lr))

    def acc_metric(y_true_batch, y_pred_batch):
        yp = np.asarray(y_pred_batch.to_numpy(), dtype=np.float32)
        yt = np.asarray(y_true_batch.to_numpy(), dtype=np.float32)
        y_int = np.argmax(yt, axis=1).astype(np.int64)
        return _accuracy_from_logits_np(y_int, yp)

    # Training loop
    print(f"Device: {device}")
    print(f"Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")
    print(
        f"MLP: 784 -> {args.hidden_dim} -> 10 | lr={args.lr} | batch={args.batch_size} | epochs={args.epochs}"
    )
    print("Loss: MSE(one-hot) | Metric: acc(argmax logits)")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        shuffle = not args.no_shuffle

        # Train epoch
        losses = []
        accs = []

        for xb_np, yb_np in _iter_minibatches(
            x_train,
            y_train_oh,
            batch_size=args.batch_size,
            shuffle=shuffle,
            seed=args.seed + epoch,
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
        yb = _tensor_from_numpy(y_test_oh, device=device)
        logits = model(xb)
        logits_np = np.asarray(logits.to_numpy(), dtype=np.float32)
        test_acc = _accuracy_from_logits_np(y_test, logits_np)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| test_acc={test_acc:.4f} | {dt:.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
