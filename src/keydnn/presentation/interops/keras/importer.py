# src/keydnn/presentation/interops/keras/importer.py
"""
Keras model importer for KeyDNN.

This module provides entrypoints for converting Keras models (or saved model
artifacts) into equivalent KeyDNN modules using registered converters.

Phase 1 scope
-------------
- Supports `tf.keras.Sequential` models.
- Supports a minimal set of layers via `layer_registry.py`.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Union

import numpy as np

from .context import KerasImportContext
from .layer_registry import build_registry
from .converters._base import KerasInteropError


def _require_tensorflow() -> Any:
    """
    Import TensorFlow lazily. Raises a friendly error if missing.
    """
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:
        raise ImportError(
            "Keras interop requires TensorFlow. Install with: pip install keydnn[keras]"
        ) from e


def _resolve_device(device: Any):
    """
    Normalize the device argument to a KeyDNN Device object when needed.
    """
    if device is None:
        from ....domain.device._device import Device

        return Device("cpu")

    # If passed as string like "cpu" / "cuda:0"
    if isinstance(device, str):
        from ....domain.device._device import Device

        return Device(device)

    return device


def _try_make_sequential(mods: Sequence[Any]) -> Any:
    """
    Try to build a KeyDNN Sequential container. If KeyDNN does not define one,
    return the list of modules.
    """
    # Try common locations; keep import local to avoid side effects.
    try:
        from ....infrastructure.models import Sequential

        return Sequential(list(mods))
    except Exception:
        pass

    try:
        from ....presentation.apis.models import Sequential

        return Sequential(list(mods))
    except Exception:
        pass

    # Fallback: return list; callers can wrap it later.
    return list(mods)


def from_keras(
    model_or_path: Union[Any, str],
    *,
    device: Any = "cpu",
    dtype: Any = np.float32,
    strict: bool = True,
    allow_non_linear_activation: bool = False,
) -> Any:
    """
    Convert a Keras model (or saved artifact path) into a KeyDNN module.

    Parameters
    ----------
    model_or_path : Any or str
        Either a `tf.keras.Model` instance or a filesystem path to a saved model
        (e.g., `.keras` or `.h5`).
    device : Any, optional
        KeyDNN device or device string. Defaults to "cpu".
    dtype : Any, optional
        Target dtype. Defaults to `np.float32`.
    strict : bool, optional
        If True, unsupported layers raise an error. Defaults to True.
    allow_non_linear_activation : bool, optional
        If True, certain converters may accept non-linear activations.
        Defaults to False.

    Returns
    -------
    Any
        A KeyDNN module (typically a Sequential container) or a list of KeyDNN
        modules if no Sequential container is available.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    KerasInteropError
        If model type or layers are unsupported in the current phase.
    """
    tf = _require_tensorflow()

    dev = _resolve_device(device)
    ctx = KerasImportContext(
        device=dev,
        dtype=dtype,
        strict=bool(strict),
        allow_non_linear_activation=bool(allow_non_linear_activation),
    )

    # Load model if path is provided
    if isinstance(model_or_path, str):
        model = tf.keras.models.load_model(model_or_path, compile=False)
    else:
        model = model_or_path

    # Phase 1: Sequential only
    if not isinstance(model, tf.keras.Sequential):
        raise KerasInteropError(
            f"Only tf.keras.Sequential is supported in Phase 1, got {type(model).__name__}."
        )

    registry = build_registry(tf, ctx=ctx)

    kd_layers: List[Any] = []
    for k_layer in list(getattr(model, "layers", [])):
        conv = registry.require(k_layer, ctx=ctx)
        kd = conv.build(k_layer, ctx)
        conv.load_weights(kd, k_layer, ctx)
        kd_layers.append(kd)

    return _try_make_sequential(kd_layers)
