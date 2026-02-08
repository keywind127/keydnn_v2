# src/keydnn/presentation/interops/keras/converters/layernorm.py
"""
Keras LayerNormalization layer converter.

This module implements conversion from `tf.keras.layers.LayerNormalization`
to KeyDNN's CPU `LayerNorm` module.

Mapping summary
---------------
Keras LayerNormalization stores:
- axis                    : int or list[int] specifying normalized axes
- epsilon                 : float
- center                  : bool (beta enabled)
- scale                   : bool (gamma enabled)
- gamma (scale)           : shape matches the reduced axes (broadcastable)
- beta  (offset)          : shape matches the reduced axes (broadcastable)

KeyDNN LayerNorm stores:
- normalized_shape        : tuple[int, ...] for the *trailing* K dimensions
- eps                     : float
- affine                  : bool (gamma+beta parameters when True)
- gamma                   : Parameter of shape normalized_shape if affine=True
- beta                    : Parameter of shape normalized_shape if affine=True

Converter behavior
------------------
- Phase 1 supports only trailing-axis LayerNormalization:
  - axis must normalize over the last K dimensions of the input tensor
  - Supported forms:
    - axis = -1
    - axis = [-1]
    - axis = [-K, ..., -1] for K >= 1
- `normalized_shape` is inferred from the Keras layer weights when available
  (preferred), otherwise inferred from the built input shape + axis.
- `affine` is enabled only when both `center=True` and `scale=True` in Keras.
  Partial affine configurations are represented as non-affine KeyDNN LayerNorm.
- Epsilon is mapped directly.
- KeyDNN LayerNorm is CPU-only. Conversion rejects non-CPU targets.

Notes
-----
- Keras LayerNormalization supports more general `axis` configurations (including
  non-trailing axes). Those configurations are rejected in Phase 1 because
  KeyDNN LayerNorm is defined over trailing dimensions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from ._base import BaseConverter, KerasInteropError
from ._utils import copy_param_from_numpy
from .....domain.device._device import Device


def _require_cpu_device(ctx: Any) -> Device:
    """
    Require CPU device for LayerNorm conversion.

    Parameters
    ----------
    ctx : Any
        Conversion context providing `device`.

    Returns
    -------
    Device
        Resolved KeyDNN device.

    Raises
    ------
    KerasInteropError
        If the resolved device is not CPU.
    """
    dev = getattr(ctx, "device", None)
    if dev is None:
        dev = Device("cpu")
    if isinstance(dev, str):
        dev = Device(dev)
    if str(dev) != "cpu":
        raise KerasInteropError(
            f"LayerNorm conversion supports CPU only, got device={dev}."
        )
    return dev


def _axis_to_tuple(axis: Any) -> Tuple[int, ...]:
    """
    Normalize a Keras LayerNormalization axis specification to a tuple of ints.

    Parameters
    ----------
    axis : Any
        Keras axis specification (int or iterable of ints).

    Returns
    -------
    Tuple[int, ...]
        Normalized axis tuple.

    Raises
    ------
    KerasInteropError
        If axis cannot be interpreted as an integer or sequence of integers.
    """
    if isinstance(axis, (int, np.integer)):
        return (int(axis),)
    if isinstance(axis, (tuple, list)):
        if len(axis) == 0:
            raise KerasInteropError("LayerNormalization axis must be non-empty.")
        try:
            return tuple(int(a) for a in axis)
        except Exception as e:  # pragma: no cover
            raise KerasInteropError(f"Invalid LayerNormalization axis={axis!r}.") from e
    raise KerasInteropError(
        f"LayerNormalization axis must be int or sequence of ints, got {axis!r}."
    )


def _infer_input_shape(k_layer: Any) -> Optional[Tuple[Optional[int], ...]]:
    """
    Best-effort inference of the built input shape for a Keras layer.

    Parameters
    ----------
    k_layer : Any
        Keras LayerNormalization layer.

    Returns
    -------
    Optional[Tuple[Optional[int], ...]]
        Input shape tuple when available, otherwise None.

    Notes
    -----
    This helper checks common public attributes without relying on private APIs.
    """
    candidates = [
        getattr(k_layer, "input_shape", None),
        getattr(k_layer, "_batch_input_shape", None),
        getattr(k_layer, "batch_input_shape", None),
        getattr(getattr(k_layer, "input", None), "shape", None),
    ]

    for shp in candidates:
        if shp is None:
            continue
        try:
            return tuple(shp)
        except Exception:
            try:
                return tuple(getattr(shp, "as_list")())
            except Exception:
                continue
    return None


def _require_trailing_axes(axis: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    """
    Require that axis corresponds to the trailing K dimensions of the input.

    Parameters
    ----------
    axis : Tuple[int, ...]
        Normalized axis tuple.
    rank : int
        Input rank (including batch dimension).

    Returns
    -------
    Tuple[int, ...]
        Canonical trailing axes as negative indices (e.g., (-2, -1)).

    Raises
    ------
    KerasInteropError
        If axis does not represent the trailing K dimensions.
    """
    if rank <= 0:
        raise KerasInteropError(f"Invalid input rank={rank} for LayerNormalization.")

    # Canonicalize axis to negative indices in range [-rank, -1]
    canon: Tuple[int, ...] = tuple((a - rank) if a >= 0 else a for a in axis)

    # Validate range
    for a in canon:
        if a < -rank or a > -1:
            raise KerasInteropError(
                f"LayerNormalization axis contains out-of-range index {a} for rank={rank}."
            )

    # Remove duplicates while preserving order
    seen = set()
    canon_unique = []
    for a in canon:
        if a in seen:
            raise KerasInteropError(
                "LayerNormalization axis with repeated entries is not supported in Phase 1."
            )
        seen.add(a)
        canon_unique.append(a)
    canon_u = tuple(canon_unique)

    # Phase 1: axis must equal the last K dimensions
    k = len(canon_u)
    trailing = tuple(range(-k, 0))  # (-k, ..., -1)
    if canon_u != trailing:
        raise KerasInteropError(
            "LayerNormalization axis is not supported in Phase 1. "
            f"Expected trailing axes {trailing} for rank={rank}, got axis={canon_u}."
        )
    return canon_u


def _extract_ln_weights(
    k_layer: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract LayerNormalization weights from a Keras layer.

    Parameters
    ----------
    k_layer : Any
        Keras LayerNormalization layer instance.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (gamma, beta) where each may be None depending on scale/center.

    Raises
    ------
    KerasInteropError
        If the layer is unbuilt but declares weights, or if an unexpected weight
        count is encountered.
    """
    w = list(k_layer.get_weights())
    center = bool(getattr(k_layer, "center", True))
    scale = bool(getattr(k_layer, "scale", True))

    expected = (1 if scale else 0) + (1 if center else 0)
    if len(w) == 0:
        # Valid when both center=False and scale=False, or when layer is built
        # but has no weights. In Phase 1, normalized_shape must still be inferred
        # via input shape.
        if expected != 0:
            raise KerasInteropError(
                "Keras LayerNormalization has no weights. Make sure the Keras model is built "
                "(e.g., call it once or load from file)."
            )
        return None, None

    if len(w) != expected:
        raise KerasInteropError(
            f"Unexpected LayerNormalization weight count: {len(w)} (expected {expected})."
        )

    idx = 0
    gamma = None
    beta = None
    if scale:
        gamma = np.asarray(w[idx])
        idx += 1
    if center:
        beta = np.asarray(w[idx]) if idx < len(w) else None

    if gamma is not None and gamma.ndim < 1:
        raise KerasInteropError("LayerNormalization gamma must be at least 1D.")
    if beta is not None and beta.ndim < 1:
        raise KerasInteropError("LayerNormalization beta must be at least 1D.")

    return gamma, beta


def _infer_normalized_shape(
    k_layer: Any,
    *,
    axis: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Infer KeyDNN LayerNorm `normalized_shape` from a built Keras LayerNormalization.

    Parameters
    ----------
    k_layer : Any
        Built Keras LayerNormalization layer.
    axis : Tuple[int, ...]
        Normalized axis tuple (int tuple).

    Returns
    -------
    Tuple[int, ...]
        Normalized shape (trailing dimensions).

    Raises
    ------
    KerasInteropError
        If normalized shape cannot be inferred.
    """
    gamma, beta = _extract_ln_weights(k_layer)

    # Preferred: infer from affine weight shape (gamma/beta)
    for p in (gamma, beta):
        if p is None:
            continue
        shp = tuple(int(d) for d in p.shape)
        if any(d <= 0 for d in shp):
            raise KerasInteropError(
                f"LayerNormalization parameter shape must be positive, got {shp}."
            )
        return shp

    # Fallback: infer from input shape + axis
    input_shape = _infer_input_shape(k_layer)
    if input_shape is None:
        raise KerasInteropError(
            "LayerNormalization normalized_shape could not be inferred: missing weights and input shape."
        )

    rank = len(input_shape)
    _ = _require_trailing_axes(_axis_to_tuple(axis), rank=rank)  # validate semantics

    k = len(axis)
    trailing = input_shape[-k:]
    if any(d is None for d in trailing):
        raise KerasInteropError(
            f"LayerNormalization normalized_shape requires concrete trailing dims, got input_shape={input_shape}."
        )
    norm = tuple(int(d) for d in trailing)
    if any(d <= 0 for d in norm):
        raise KerasInteropError(
            f"LayerNormalization normalized_shape must be positive, got {norm}."
        )
    return norm


@dataclass(frozen=True)
class LayerNormalizationConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.LayerNormalization` -> KeyDNN `LayerNorm`.

    Phase 1 behavior
    ----------------
    - CPU-only conversion.
    - Supports trailing-axis LayerNormalization only.
    - Enables affine only when both center=True and scale=True in Keras.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN LayerNorm module corresponding to a Keras LayerNormalization layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras LayerNormalization layer.
        ctx : Any
            Conversion context providing `device`.

        Returns
        -------
        Any
            Constructed KeyDNN LayerNorm module.

        Raises
        ------
        KerasInteropError
            If axis semantics are unsupported, device is not CPU, or shape cannot be inferred.
        """
        device = _require_cpu_device(ctx)

        axis_raw = getattr(k_layer, "axis", -1)
        axis = _axis_to_tuple(axis_raw)

        input_shape = _infer_input_shape(k_layer)
        if input_shape is None:
            # If input shape is unavailable, attempt weight-based shape inference first.
            # Axis validation still requires a rank; infer rank as len(axis)+1 (batch + normalized dims).
            # This allows basic axis checks for canonical negative trailing indices.
            rank = 1 + len(axis)
        else:
            rank = len(input_shape)

        trailing_axis = _require_trailing_axes(axis, rank=rank)
        _ = trailing_axis  # keep canonicalization explicit in Phase 1

        normalized_shape = _infer_normalized_shape(k_layer, axis=axis)

        eps = float(getattr(k_layer, "epsilon", 1e-3))
        center = bool(getattr(k_layer, "center", True))
        scale = bool(getattr(k_layer, "scale", True))
        affine = bool(center and scale)

        from .....infrastructure.layers._layernorm import LayerNorm

        return LayerNorm(
            normalized_shape=normalized_shape,
            device=device,
            eps=float(eps),
            affine=affine,
        )

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Copy Keras LayerNormalization weights into an existing KeyDNN LayerNorm module.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN LayerNorm module created by `build`.
        k_layer : Any
            Source Keras LayerNormalization layer.
        ctx : Any
            Conversion context (reserved for future use).

        Raises
        ------
        KerasInteropError
            If affine configurations are incompatible or required parameters are missing.
        """
        gamma, beta = _extract_ln_weights(k_layer)

        kd_affine = bool(getattr(kd_layer, "affine", False))

        if kd_affine:
            if gamma is None or beta is None:
                raise KerasInteropError(
                    "KeyDNN LayerNorm is affine but Keras LayerNormalization is missing gamma/beta."
                )
            if (
                getattr(kd_layer, "gamma", None) is None
                or getattr(kd_layer, "beta", None) is None
            ):
                raise KerasInteropError(
                    "KeyDNN LayerNorm affine parameters are missing."
                )

            copy_param_from_numpy(kd_layer.gamma, gamma.astype(np.float32, copy=False))
            copy_param_from_numpy(kd_layer.beta, beta.astype(np.float32, copy=False))
            return

        # Non-affine KeyDNN LayerNorm
        if gamma is not None or beta is not None:
            raise KerasInteropError(
                "Keras LayerNormalization has gamma/beta but KeyDNN LayerNorm was constructed with affine=False."
            )
