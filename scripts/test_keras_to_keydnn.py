#!/usr/bin/env python3
"""
CLI-only end-to-end test: Keras (.h5) -> KeyDNN via `keydnn convert`.

This script's SOLE purpose is to test the console feature:
  python -m keydnn convert --src ./model.h5 --dst ./model.json

What this script verifies
------------------------
1) A small MNIST CNN can be trained in Keras.
2) The trained model can be saved to .h5 (channels_first for current interop constraints).
3) The KeyDNN CLI (`keydnn convert`) produces a *checkpoint-style* JSON artifact
   that includes BOTH architecture AND weights/state (the same schema as Sequential.save_json).
4) The converted KeyDNN model can be loaded and produces numerically-close outputs
   to the Keras reference model.

Important note (why your previous loader failed)
------------------------------------------------
You recently updated `keydnn convert` to write a checkpoint JSON using `Sequential.save_json(...)`.
That file does NOT have a top-level "modules" list. It uses:
  - payload["format"] == "keydnn.json.ckpt.v1"
  - payload["arch"]        (architecture)
  - payload["state"]       (weights, base64, dtype/shape metadata)

So the test script must load using `Sequential.load_json(dst_json)`, not a custom "modules" loader.

Usage
-----
# Basic CPU run
python scripts/test_keras_to_keydnn.py

# Change limits / tolerances
python scripts/test_keras_to_keydnn.py --epochs 1 --limit-train 4096 --limit-test 256 --rtol 1e-3 --atol 1e-3

# If you have KeyDNN CUDA available (optional)
python scripts/test_keras_to_keydnn.py --device cuda:0

Troubleshooting
---------------
- If TensorFlow emits oneDNN notes and you want stricter determinism on CPU:
    set TF_ENABLE_ONEDNN_OPTS=0
- If numerical mismatch occurs, rerun with:
    --debug --dump-max-diff
to print norms and the worst offending element.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# TensorFlow / Keras helpers
# --------------------------------------------------------------------------------------


def _require_tensorflow() -> Any:
    """
    Import TensorFlow lazily so this script can exist in repos where TF is optional.
    """
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:
        raise ImportError(
            "This script requires TensorFlow/Keras. Install with: pip install tensorflow"
        ) from e


def _prep_mnist(
    tf: Any, *, limit_train: int, limit_test: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST and return (x_train, y_train, x_test, y_test).

    - x is float32 in [0,1]
    - x is NHWC: (N,28,28,1)
    - y is int64: (N,)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]

    rng = np.random.default_rng(int(seed))

    if limit_train > 0:
        idx = rng.permutation(x_train.shape[0])[: int(limit_train)]
        x_train = x_train[idx]
        y_train = y_train[idx]

    if limit_test > 0:
        idx = rng.permutation(x_test.shape[0])[: int(limit_test)]
        x_test = x_test[idx]
        y_test = y_test[idx]

    return (
        x_train,
        np.asarray(y_train, dtype=np.int64),
        x_test,
        np.asarray(y_test, dtype=np.int64),
    )


def _nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    """
    Convert NHWC (N,H,W,C) to NCHW (N,C,H,W).
    """
    return np.asarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def _build_keras_mnist_cnn_channels_last(tf: Any, *, num_classes: int = 10) -> Any:
    """
    CPU-friendly training uses channels_last (NHWC).

    NOTE: Use activation="linear" in parameterized layers and keep explicit
    activation modules (ReLU/Softmax) as dedicated layers to match KeyDNN graphs.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                padding="same",
                activation="linear",
                data_format="channels_last",
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), data_format="channels_last"
            ),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding="same",
                activation="linear",
                data_format="channels_last",
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last"),
            tf.keras.layers.Flatten(),  # no-op if already rank-2, but keeps stacks aligned
            tf.keras.layers.Dense(num_classes, activation="linear"),
            tf.keras.layers.Softmax(axis=-1),
        ]
    )


def _build_keras_mnist_cnn_channels_first(tf: Any, *, num_classes: int = 10) -> Any:
    """
    Phase-1 KeyDNN interop expects channels_first stacks (NCHW).

    We keep the same layer ordering/semantics as the channels_last model so
    we can copy weights layerwise.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1, 28, 28)),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                padding="same",
                activation="linear",
                data_format="channels_first",
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), data_format="channels_first"
            ),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding="same",
                activation="linear",
                data_format="channels_first",
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first"),
            tf.keras.layers.Flatten(),  # ensures Dense sees (N,C)
            tf.keras.layers.Dense(num_classes, activation="linear"),
            tf.keras.layers.Softmax(axis=-1),
        ]
    )


def _copy_weights_layerwise(model_src: Any, model_dst: Any) -> None:
    """
    Copy weights layer-by-layer assuming identical topology.

    Notes
    -----
    We keep layer stacks identical across channels_last vs channels_first
    by using the same layer types and ordering, so weight lists match.
    """
    src_layers = list(model_src.layers)
    dst_layers = list(model_dst.layers)
    if len(src_layers) != len(dst_layers):
        raise ValueError(
            f"Keras layer count mismatch for weight copy: src={len(src_layers)} dst={len(dst_layers)}"
        )
    for a, b in zip(src_layers, dst_layers):
        w = a.get_weights()
        if w:
            b.set_weights(w)


# --------------------------------------------------------------------------------------
# KeyDNN runtime helpers
# --------------------------------------------------------------------------------------


def _device_from_string(device_str: str):
    """
    Create a KeyDNN Device from a string like "cpu" or "cuda:0".
    """
    from keydnn.domain.device._device import Device  # type: ignore

    return Device(device_str)


def _tensor_from_numpy(arr: np.ndarray, *, device):
    """
    Create a KeyDNN Tensor from NumPy using stable APIs if available.
    """
    a = np.asarray(arr, dtype=np.float32)

    try:
        from keydnn.presentation.apis.tensors import Tensor  # type: ignore

        t = Tensor(shape=a.shape, device=device, requires_grad=False)
        t.copy_from_numpy(a)
        return t
    except Exception:
        from keydnn.infrastructure.tensor._tensor import Tensor  # type: ignore

        try:
            return Tensor(data=a, device=device)
        except TypeError:
            t = Tensor(shape=a.shape, device=device, requires_grad=False, ctx=None)
            if hasattr(t, "copy_from_numpy"):
                t.copy_from_numpy(a)
            else:
                t.copy_from(a)
            return t


def _forward_keydnn(model: Any, x_np: np.ndarray, *, device) -> np.ndarray:
    """
    Run a forward pass through a KeyDNN model and return numpy output.
    """
    x = _tensor_from_numpy(x_np, device=device)
    y = model(x)
    return np.asarray(y.to_numpy(), dtype=np.float32)


def _debug_param_norms_sequential(model: Any) -> None:
    """
    Print L2 norms for common parameter attrs to confirm weights are non-default.
    """

    def _l2(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.sum(x * x)))

    attrs = ("weight", "bias", "gamma", "beta", "running_mean", "running_var")
    mods = list(getattr(model, "modules", [])) if hasattr(model, "modules") else []
    if not mods and hasattr(model, "__len__"):
        try:
            mods = [model[i] for i in range(len(model))]
        except Exception:
            mods = []

    print("[debug] KeyDNN parameter norms:")
    for i, m in enumerate(mods):
        for a in attrs:
            p = getattr(m, a, None)
            if p is None:
                continue
            if hasattr(p, "to_numpy"):
                arr = np.asarray(p.to_numpy(), dtype=np.float32)
            elif hasattr(p, "data") and isinstance(getattr(p, "data"), np.ndarray):
                arr = np.asarray(p.data, dtype=np.float32)
            else:
                continue
            print(
                f"  m{i}.{type(m).__name__}.{a}: shape={tuple(arr.shape)} l2={_l2(arr):.6f}"
            )


# --------------------------------------------------------------------------------------
# CLI invocation
# --------------------------------------------------------------------------------------


def _run_keydnn_convert(*, src_h5: Path, dst_json: Path) -> None:
    """
    Invoke the CLI feature under test.

    Uses: python -m keydnn convert --src ... --dst ...
    Raises RuntimeError on non-zero exit.
    """
    cmd = [
        sys.executable,
        "-m",
        "keydnn",
        "convert",
        "--src",
        str(src_h5),
        "--dst",
        str(dst_json),
    ]
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.stdout.strip():
        print("convert stdout:\n", p.stdout)
    if p.stderr.strip():
        print("convert stderr:\n", p.stderr)

    if p.returncode != 0:
        raise RuntimeError(f"keydnn convert failed with exit code {p.returncode}")


# --------------------------------------------------------------------------------------
# Artifact loading (NEW: ckpt-aware)
# --------------------------------------------------------------------------------------


def _load_keydnn_model_from_cli_artifact(dst_json: Path) -> Any:
    """
    Load the KeyDNN model artifact produced by `keydnn convert`.

    Expected formats
    ----------------
    - Preferred (default after your convert.py update):
        format == "keydnn.json.ckpt.v1"
        Load with: Sequential.load_json(path)

    - Legacy config-only:
        format == "keydnn.model.v1"
        This contains modules/config only; it is NOT suitable for parity checks.

    Returns
    -------
    Any
        A KeyDNN Sequential model.

    Raises
    ------
    ValueError
        If the artifact is config-only or an unknown schema.
    """
    payload = json.loads(dst_json.read_text(encoding="utf-8"))
    fmt = str(payload.get("format", ""))

    if fmt == "keydnn.json.ckpt.v1":
        try:
            # Primary location per your unit tests
            from keydnn.infrastructure.models._sequential import Sequential  # type: ignore
        except Exception:
            # Fallback if project structure differs
            from keydnn.infrastructure.module.containers import Sequential  # type: ignore

        return Sequential.load_json(dst_json)

    if fmt == "keydnn.model.v1":
        raise ValueError(
            "CLI produced a config-only artifact (keydnn.model.v1) without weights/state. "
            "Parity checks are meaningless. Ensure `keydnn convert` is using Sequential.save_json(...) "
            "and re-run."
        )

    # If someone changes the CLI later, fail loudly with context.
    raise ValueError(
        f"Unknown/unsupported KeyDNN artifact schema: format={fmt!r}. "
        f"Update _load_keydnn_model_from_cli_artifact(...) to match."
    )


# --------------------------------------------------------------------------------------
# Debug helpers
# --------------------------------------------------------------------------------------


def _dump_max_diff(a: np.ndarray, b: np.ndarray) -> None:
    """
    Print worst offending element and a few summary stats.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    idx = int(np.argmax(diff))
    max_abs = float(diff.flat[idx])
    denom = float(np.abs(b.flat[idx])) + 1e-12
    max_rel = float(max_abs / denom)
    print(f"[diff] max_abs={max_abs:.6e} max_rel={max_rel:.6e} at flat_index={idx}")
    print(f"[diff] a={float(a.flat[idx]):.6e} b={float(b.flat[idx]):.6e}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    epochs: int = 1
    batch: int = 64
    lr: float = 1e-3
    seed: int = 0
    limit_train: int = 4096
    limit_test: int = 256
    src_h5: str = "./mnist_cnn_ch_first.h5"
    dst_json: str = "./mnist_cnn_keydnn.json"
    device: str = "cpu"
    rtol: float = 1e-3
    atol: float = 1e-3
    debug: bool = True
    dump_max_diff: bool = False


def main() -> int:
    ap = argparse.ArgumentParser(prog="test_keras_to_keydnn", description=__doc__)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=4096)
    ap.add_argument("--limit-test", type=int, default=256)
    ap.add_argument("--src-h5", type=str, default="./mnist_cnn_ch_first.h5")
    ap.add_argument("--dst-json", type=str, default="./mnist_cnn_keydnn.json")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--dump-max-diff", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        seed=int(args.seed),
        limit_train=int(args.limit_train),
        limit_test=int(args.limit_test),
        src_h5=str(args.src_h5),
        dst_json=str(args.dst_json),
        device=str(args.device),
        rtol=float(args.rtol),
        atol=float(args.atol),
        debug=not bool(args.no_debug),
        dump_max_diff=bool(args.dump_max_diff),
    )

    tf = _require_tensorflow()
    tf.random.set_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    print("[1/6] Load MNIST")
    x_train, y_train, x_test, _ = _prep_mnist(
        tf, limit_train=cfg.limit_train, limit_test=cfg.limit_test, seed=cfg.seed
    )

    print("[2/6] Train Keras CNN (channels_last)")
    model_cl = _build_keras_mnist_cnn_channels_last(tf)
    model_cl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg.lr)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model_cl.fit(
        x_train,
        y_train,
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch),
        verbose=1,
    )

    print("[3/6] Create channels_first Keras model, copy weights, save .h5")
    model_cf = _build_keras_mnist_cnn_channels_first(tf)
    _ = model_cf(_nhwc_to_nchw(x_train[:2]))  # build graph/weights
    _copy_weights_layerwise(model_cl, model_cf)

    src_h5 = Path(cfg.src_h5)
    src_h5.parent.mkdir(parents=True, exist_ok=True)
    model_cf.save(str(src_h5), include_optimizer=False)

    print("[4/6] Run CLI: python -m keydnn convert --src ... --dst ...")
    dst_json = Path(cfg.dst_json)
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    _run_keydnn_convert(src_h5=src_h5, dst_json=dst_json)

    print("[5/6] Load KeyDNN artifact produced by CLI")
    kd_model = _load_keydnn_model_from_cli_artifact(dst_json)

    if cfg.debug:
        try:
            _debug_param_norms_sequential(kd_model)
        except Exception as e:
            print(f"[debug] (skipped param norm print) reason: {e}")

    print("[6/6] Compare inference (Keras channels_first vs KeyDNN loaded)")
    device = _device_from_string(cfg.device)

    n = min(int(cfg.batch), x_test.shape[0])
    x_batch_nchw = _nhwc_to_nchw(x_test[:n])

    # Keras reference
    y_keras = model_cf.predict(x_batch_nchw, batch_size=n, verbose=0).astype(np.float32)

    # KeyDNN
    y_kd = _forward_keydnn(kd_model, x_batch_nchw, device=device)

    try:
        np.testing.assert_allclose(
            y_kd, y_keras, rtol=float(cfg.rtol), atol=float(cfg.atol)
        )
    except AssertionError:
        if cfg.dump_max_diff:
            _dump_max_diff(y_kd, y_keras)
        raise

    print(
        f"OK: KeyDNN outputs match Keras within rtol={cfg.rtol} atol={cfg.atol} "
        f"(shape={y_kd.shape})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
