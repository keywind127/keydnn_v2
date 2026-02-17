"""
Integration Test: Keras -> KeyDNN MNIST CNN Parity

This script:
1) Trains a small CNN on MNIST in Keras (NHWC).
2) Converts the trained model into KeyDNN using `from_keras`.
3) Runs inference in both frameworks and compares:
   - accuracy on the same test subset
   - probability parity (max/mean absolute difference)

Model Architecture (Keras)
-------------------------
Input (28,28,1)
  -> Conv2D -> BatchNorm -> ReLU -> MaxPool
  -> Conv2D -> BatchNorm -> ReLU -> MaxPool
  -> GlobalAveragePooling2D
  -> Dense -> ReLU
  -> Dropout
  -> Dense(10) -> Softmax

Usage
-----
# CPU
python scripts/train_mnist_cnn_keras_to_keydnn.py --device cpu --epochs 3

# CUDA (KeyDNN inference on cuda, if your KeyDNN build supports it)
python scripts/train_mnist_cnn_keras_to_keydnn.py --device cuda:0 --epochs 3

# Fast smoke test
python scripts/train_mnist_cnn_keras_to_keydnn.py --epochs 1 --limit-train 4096 --limit-test 1024
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, optimizers

from keydnn.presentation.interops.keras import from_keras
import keydnn as kd


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    device: str = "cpu"  # KeyDNN device for inference after conversion
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0
    limit_train: int = 0  # 0 => full MNIST train
    limit_test: int = 0  # 0 => full MNIST test
    verbose: int = 1


# --------------------------------------------------------------------------------------
# Keras model
# --------------------------------------------------------------------------------------


def build_keras_mnist_cnn() -> tf.keras.Sequential:
    """
    Build a small MNIST CNN in Keras using Sequential (NCHW),
    compatible with KeyDNN Phase-1 from_keras (Sequential + channels_first).
    """
    return tf.keras.Sequential(
        [
            layers.Input(shape=(1, 28, 28), name="input"),  # NCHW (C,H,W)
            # Block 1
            layers.Conv2D(
                16,
                kernel_size=3,
                padding="same",
                use_bias=True,
                data_format="channels_first",
                name="conv1",
            ),
            layers.BatchNormalization(axis=1, name="bn1"),  # channel axis = 1 in NCHW
            layers.ReLU(name="relu1"),
            layers.MaxPool2D(
                pool_size=2, strides=2, data_format="channels_first", name="pool1"
            ),
            # Block 2
            layers.Conv2D(
                32,
                kernel_size=3,
                padding="same",
                use_bias=True,
                data_format="channels_first",
                name="conv2",
            ),
            layers.BatchNormalization(axis=1, name="bn2"),
            layers.ReLU(name="relu2"),
            layers.MaxPool2D(
                pool_size=2, strides=2, data_format="channels_first", name="pool2"
            ),
            # Head
            layers.GlobalAveragePooling2D(data_format="channels_first", name="gap"),
            layers.Flatten(),
            layers.Dense(64, activation="linear", name="fc1"),
            layers.ReLU(name="relu3"),
            layers.Dropout(0.2, name="dropout"),
            layers.Dense(10, activation="linear", name="fc2"),
            layers.Softmax(axis=-1, name="softmax"),
        ],
        name="mnist_cnn_bn_seq_nchw",
    )


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_mnist(*, limit_train: int, limit_test: int, seed: int):
    """
    Load MNIST as:
      - Keras input:  (N, 28, 28, 1) float32 in [0,1]
      - labels int64: (N,)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # add channel dim then transpose to NCHW for Keras channels_first
    x_train = x_train[:, None, :, :]  # (N,1,28,28)
    x_test = x_test[:, None, :, :]

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    if limit_train > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x_train))[:limit_train]
        x_train, y_train = x_train[idx], y_train[idx]

    if limit_test > 0:
        rng = np.random.default_rng(seed + 1)
        idx = rng.permutation(len(x_test))[:limit_test]
        x_test, y_test = x_test[idx], y_test[idx]

    return x_train, y_train, x_test, y_test


def accuracy_from_probs(y_true_int: np.ndarray, probs: np.ndarray) -> float:
    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    pred = np.argmax(np.asarray(probs), axis=-1).astype(np.int64)
    return float((pred == y_true_int).mean())


# --------------------------------------------------------------------------------------
# KeyDNN helpers
# --------------------------------------------------------------------------------------


def to_keydnn_input_nchw(x_nhwc: np.ndarray) -> np.ndarray:
    """
    Keras uses NHWC: (N,H,W,C)
    KeyDNN in your codebase typically uses NCHW: (N,C,H,W)
    """
    x = np.asarray(x_nhwc, dtype=np.float32)
    return np.transpose(x, (0, 3, 1, 2)).copy()


def set_keydnn_eval_mode(model: kd.Sequential) -> None:
    """
    Best-effort: put KeyDNN modules into eval mode so BN/Dropout behave like inference.
    """
    for layer in model.layers():
        if not hasattr(layer, "training"):
            continue
        if callable(layer.training):
            layer.training(False)
        else:
            layer.training = False


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="train_mnist_cnn_keras_to_keydnn", description=__doc__
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = Config(
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        limit_train=int(args.limit_train),
        limit_test=int(args.limit_test),
        verbose=int(args.verbose),
    )

    # Repro-ish
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    # Load data
    x_train, y_train, x_test, y_test = load_mnist(
        limit_train=cfg.limit_train,
        limit_test=cfg.limit_test,
        seed=cfg.seed,
    )

    if cfg.verbose:
        print(f"[DATA] train={x_train.shape} test={x_test.shape}")
        print(
            f"[CFG ] epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} seed={cfg.seed}"
        )
        print(f"[KD  ] device={cfg.device}")

    # Build + train Keras
    keras_model = build_keras_mnist_cnn()
    keras_model.summary()

    keras_model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    keras_model.fit(
        x_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        verbose=cfg.verbose,
        validation_data=(x_test, y_test),
    )

    kr_loss, kr_acc = keras_model.evaluate(x_test, y_test, verbose=0)
    if cfg.verbose:
        print(f"[KR ] test_loss={kr_loss:.6f} test_acc={kr_acc:.4f}")

    kr_probs = keras_model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
    if cfg.verbose:
        print(f"[KR ] probs: max={kr_probs.max():.6f} min={kr_probs.min():.6f}")

    # Convert -> KeyDNN
    keydnn_model = from_keras(
        keras_model,
        device="cpu",  # load weights on CPU first for safety; then move
        allow_non_linear_activation=True,
    )

    # Move to requested device
    keydnn_model.to_(cfg.device)
    set_keydnn_eval_mode(keydnn_model)

    if cfg.verbose:
        print("[KD ] model summary:")
        try:
            print(keydnn_model.summary())
        except Exception:
            pass

    x_test_nchw = np.asarray(x_test, dtype=np.float32)

    x_kd = kd.numpy_to_tensor(x_test_nchw)
    kd_probs_t = keydnn_model.predict(x_kd)  # expected probs after softmax
    kd_probs = kd_probs_t.to_numpy()

    kd_acc = accuracy_from_probs(y_test, kd_probs)
    if cfg.verbose:
        print(f"[KD ] probs: max={kd_probs.max():.6f} min={kd_probs.min():.6f}")
        print(f"[KD ] test_acc={kd_acc:.4f}")

    # Parity stats (probability-level)
    kr_probs_2d = np.asarray(kr_probs, dtype=np.float32).reshape(len(x_test), -1)
    kd_probs_2d = np.asarray(kd_probs, dtype=np.float32).reshape(len(x_test), -1)

    if kr_probs_2d.shape != kd_probs_2d.shape:
        print(f"[WARN] shape mismatch: KR={kr_probs_2d.shape} KD={kd_probs_2d.shape}")
    else:
        abs_diff = np.abs(kr_probs_2d - kd_probs_2d)
        print(
            "[PAR] abs_diff: "
            f"max={abs_diff.max():.8f} mean={abs_diff.mean():.8f} p99={np.quantile(abs_diff, 0.99):.8f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
