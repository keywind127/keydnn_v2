#!/usr/bin/env python3
"""
Keras Sequential -> KeyDNN conversion smoke script (forward numerical sanity).

This script builds a small `tf.keras.Sequential` model, converts it to KeyDNN via
`from_keras(...)`, then compares forward outputs on the same input.

Usage
-----
# CPU, default pipeline
python scripts/keras_to_keydnn_smoke.py --device cpu

# Choose a pipeline
python scripts/keras_to_keydnn_smoke.py --case conv2d_bn_relu_pool_flatten_dense_dropout

# Tighten/loosen tolerances
python scripts/keras_to_keydnn_smoke.py --rtol 1e-4 --atol 1e-4

Notes
-----
- TensorFlow is required for this script.
- Pipelines use channels_first where relevant (Phase 1 constraint).
- Dropout/BatchNorm are executed in inference mode:
  - Keras: training=False
  - KeyDNN: best-effort eval/training=False
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# TensorFlow / Keras
# --------------------------------------------------------------------------------------


def _require_tensorflow() -> Any:
    """
    Import TensorFlow and raise a friendly error if missing.
    """
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:
        raise ImportError(
            "This script requires TensorFlow. Install with: pip install keydnn[keras]"
        ) from e


# --------------------------------------------------------------------------------------
# KeyDNN helpers
# --------------------------------------------------------------------------------------


def _device_from_string(device_str: str):
    """
    Create a KeyDNN Device from a string like "cpu" or "cuda:0".
    """
    from keydnn.domain.device._device import Device

    return Device(device_str)


def _to_module_list(kd_out: Any) -> List[Any]:
    """
    Normalize importer output to an ordered list of modules.
    """
    if isinstance(kd_out, list):
        return list(kd_out)

    for attr in ("modules", "layers"):
        if hasattr(kd_out, attr):
            v = getattr(kd_out, attr)
            if isinstance(v, (list, tuple)):
                return list(v)

    return [kd_out]


def _set_eval_mode(kd_out: Any) -> None:
    """
    Best-effort set inference mode for KeyDNN containers/modules.
    """
    if hasattr(kd_out, "eval") and callable(getattr(kd_out, "eval")):
        kd_out.eval()
        return

    for m in _to_module_list(kd_out):
        if hasattr(m, "training"):
            try:
                setattr(m, "training", False)
            except Exception:
                pass


def _tensor_from_numpy(x: np.ndarray, *, device: Any):
    """
    Create a KeyDNN Tensor from numpy using public API when possible.
    """
    x = np.asarray(x, dtype=np.float32)

    # Preferred stable import
    try:
        from keydnn.presentation.apis.tensors import Tensor  # type: ignore
    except Exception:
        from keydnn.infrastructure.tensor._tensor import Tensor  # type: ignore

    try:
        t = Tensor(shape=tuple(x.shape), device=device, requires_grad=False, ctx=None)
    except TypeError:
        # Older constructor variants
        t = Tensor(tuple(x.shape), device)  # type: ignore

    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        t.copy_from_numpy(x)
        return t

    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        t.from_numpy(x)  # type: ignore
        return t

    raise RuntimeError("KeyDNN Tensor does not support numpy loading.")


def _tensor_to_numpy(y: Any) -> np.ndarray:
    """
    Convert KeyDNN Tensor-like output to numpy.
    """
    if hasattr(y, "to_numpy") and callable(getattr(y, "to_numpy")):
        return np.asarray(y.to_numpy())
    if hasattr(y, "data") and isinstance(getattr(y, "data"), np.ndarray):
        return np.asarray(y.data)
    raise RuntimeError("Failed to convert KeyDNN output to numpy.")


def _forward_keydnn(kd_out: Any, x_np: np.ndarray, *, device: Any) -> np.ndarray:
    """
    Run a forward pass through a KeyDNN container or module list.
    """
    x = _tensor_from_numpy(x_np, device=device)

    if callable(kd_out):
        y = kd_out(x)
        return _tensor_to_numpy(y)

    y_any = x
    for m in _to_module_list(kd_out):
        if not callable(m):
            raise RuntimeError(f"Non-callable module in pipeline: {type(m).__name__}")
        y_any = m(y_any)
    return _tensor_to_numpy(y_any)


# --------------------------------------------------------------------------------------
# Keras model factory
# --------------------------------------------------------------------------------------


def _build_keras_case(tf: Any, case: str, *, seed: int) -> Tuple[Any, np.ndarray]:
    """
    Build a Keras Sequential model and a matching input batch.

    Returns
    -------
    (model, x_np)
    """
    tf.random.set_seed(int(seed))
    np.random.seed(int(seed))

    if case == "conv2d_relu_flatten_dense":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 8, 8)),
                tf.keras.layers.Conv2D(
                    filters=4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(5, use_bias=True, activation="linear"),
            ]
        )
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        return model, x

    if case == "conv2d_bn_relu_pool_flatten_dense_dropout":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 16, 16)),
                tf.keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=False,
                    activation="linear",
                ),
                tf.keras.layers.BatchNormalization(axis=1, center=True, scale=True),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, use_bias=True, activation="linear"),
                tf.keras.layers.Dropout(0.25),
            ]
        )
        x = np.random.randn(2, 3, 16, 16).astype(np.float32)
        return model, x

    if case == "conv2d_avgpool_gap_flatten_dense_softmax":
        # Flatten after GAP keeps Dense happy even if KeyDNN GAP returns (N,C,1,1).
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, 12, 12)),
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                ),
                tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4, use_bias=True, activation="linear"),
                tf.keras.layers.Softmax(),
            ]
        )
        x = np.random.randn(2, 3, 12, 12).astype(np.float32)
        return model, x

    if case == "flatten_dense_leakyrelu_dense_sigmoid":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(2, 3, 4)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, use_bias=True, activation="linear"),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(3, use_bias=True, activation="linear"),
                tf.keras.layers.Activation("sigmoid"),
            ]
        )
        x = np.random.randn(2, 2, 3, 4).astype(np.float32)
        return model, x

    if case == "layernorm_dense":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(12,)),
                tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True),
                tf.keras.layers.Dense(6, use_bias=True, activation="linear"),
            ]
        )
        x = np.random.randn(2, 12).astype(np.float32)
        return model, x

    if case == "conv2d_transpose_relu":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4, 6, 6)),
                tf.keras.layers.Conv2DTranspose(
                    filters=3,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="valid",
                    data_format="channels_first",
                    use_bias=True,
                    activation="linear",
                ),
                tf.keras.layers.ReLU(),
            ]
        )
        x = np.random.randn(2, 4, 6, 6).astype(np.float32)
        return model, x

    raise ValueError(f"Unknown --case '{case}'.")


def _available_cases() -> List[str]:
    return [
        "conv2d_relu_flatten_dense",
        "conv2d_bn_relu_pool_flatten_dense_dropout",
        "conv2d_avgpool_gap_flatten_dense_softmax",
        "flatten_dense_leakyrelu_dense_sigmoid",
        "layernorm_dense",
        "conv2d_transpose_relu",
    ]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    case: str
    device: str
    dtype: str
    seed: int
    rtol: float
    atol: float
    strict: bool
    allow_non_linear_activation: bool
    verbose: int


def _parse_args() -> Config:
    ap = argparse.ArgumentParser(
        prog="keras_to_keydnn_smoke",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--case", type=str, default="conv2d_relu_flatten_dense")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--allow-non-linear-activation", action="store_true", default=False)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.case not in _available_cases():
        raise SystemExit(
            f"Unknown --case '{args.case}'. Valid cases: {', '.join(_available_cases())}"
        )

    return Config(
        case=str(args.case),
        device=str(args.device),
        dtype=str(args.dtype),
        seed=int(args.seed),
        rtol=float(args.rtol),
        atol=float(args.atol),
        strict=bool(args.strict),
        allow_non_linear_activation=bool(args.allow_non_linear_activation),
        verbose=int(args.verbose),
    )


def main() -> int:
    cfg = _parse_args()
    tf = _require_tensorflow()

    device = _device_from_string(cfg.device)

    if cfg.dtype.lower() != "float32":
        raise SystemExit("This smoke script currently supports --dtype float32 only.")

    if cfg.verbose:
        print(f"[keras_to_keydnn_smoke] case={cfg.case} device={cfg.device}")
        print(
            f"[keras_to_keydnn_smoke] rtol={cfg.rtol} atol={cfg.atol} seed={cfg.seed}"
        )

    model, x = _build_keras_case(tf, cfg.case, seed=cfg.seed)

    # Keras reference (inference mode)
    y_ref = model(x, training=False).numpy()

    # Convert to KeyDNN
    from keydnn.presentation.interops.keras.importer import from_keras  # type: ignore

    kd = from_keras(
        model,
        device=device,
        dtype=np.float32,
        strict=cfg.strict,
        allow_non_linear_activation=cfg.allow_non_linear_activation,
    )
    _set_eval_mode(kd)

    y_kd = _forward_keydnn(kd, x, device=device)

    if cfg.verbose:
        print(f"[keras_to_keydnn_smoke] keras out:  {tuple(y_ref.shape)}")
        print(f"[keras_to_keydnn_smoke] keydnn out: {tuple(y_kd.shape)}")

    np.testing.assert_allclose(y_kd, y_ref, rtol=cfg.rtol, atol=cfg.atol)

    if cfg.verbose:
        print("[keras_to_keydnn_smoke] PASS: outputs match within tolerance")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
