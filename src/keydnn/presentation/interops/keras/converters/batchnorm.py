# src/keydnn/presentation/interops/keras/converters/batchnorm.py
"""
Keras BatchNormalization layer converter.

This module implements conversion from `tf.keras.layers.BatchNormalization`
to KeyDNN's CPU BatchNorm modules:

- `BatchNorm1d` for 2D inputs shaped (N, C)
- `BatchNorm2d` for 4D inputs shaped (N, C, H, W) with NCHW layout

Mapping summary
---------------
Keras BatchNormalization stores:
- gamma (scale)            : (C,) if `scale=True`, otherwise absent
- beta  (offset)           : (C,) if `center=True`, otherwise absent
- moving_mean              : (C,) non-trainable
- moving_variance          : (C,) non-trainable

KeyDNN BatchNorm modules store:
- running_mean             : (C,) non-trainable
- running_var              : (C,) non-trainable
- gamma                    : (C,) Parameter if affine=True
- beta                     : (C,) Parameter if affine=True

Converter behavior
------------------
- A single converter (`BatchNormalizationConverter`) selects `BatchNorm1d` or
  `BatchNorm2d` based on inferred input rank (2D vs 4D).
- `affine` is enabled only when both `center=True` and `scale=True` in Keras.
  Partial affine configurations are represented as non-affine KeyDNN BatchNorm.
- Running statistics are copied from Keras moving statistics into KeyDNN
  running buffers.
- Epsilon and momentum are mapped directly.

Phase 1 constraints
-------------------
- KeyDNN BatchNorm implementations are CPU-only. Conversion rejects non-CPU
  targets.
- Keras `axis` must represent channel-first semantics:
  - Supported: axis=1
  - Unsupported: axis=-1 (channels_last) and multi-axis normalization
- The converter supports only rank-2 and rank-4 BatchNormalization use cases.
  When input rank cannot be inferred from the Keras layer, conversion defaults
  to `BatchNorm2d` and relies on runtime shape validation in the module.

Notes
-----
- The converter materializes affine parameters only when Keras provides both
  gamma and beta. When either is disabled, a non-affine KeyDNN BatchNorm is
  created and any present gamma/beta weights are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ._base import BaseConverter, KerasInteropError
from ._utils import copy_param_from_numpy
from .....domain.device._device import Device


def _require_cpu_device(ctx: Any) -> Device:
    """
    Require CPU device for BatchNorm conversion.

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
            f"BatchNorm conversion supports CPU only, got device={dev}."
        )
    return dev


def _extract_bn_weights(
    k_layer: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract BatchNormalization weights from a Keras layer.

    Parameters
    ----------
    k_layer : Any
        Keras BatchNormalization layer instance.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]
        (gamma, beta, moving_mean, moving_variance), where gamma/beta may be None.

    Raises
    ------
    KerasInteropError
        If weights are missing (unbuilt layer) or unexpected count is encountered.
    """
    w = list(k_layer.get_weights())
    if len(w) == 0:
        raise KerasInteropError(
            "Keras BatchNormalization has no weights. Make sure the Keras model is built "
            "(e.g., call it once or load from file)."
        )

    center = bool(getattr(k_layer, "center", True))
    scale = bool(getattr(k_layer, "scale", True))

    # Keras typical order: [gamma, beta, moving_mean, moving_variance]
    # If scale=False, gamma is omitted; if center=False, beta is omitted.
    expected = (1 if scale else 0) + (1 if center else 0) + 2
    if len(w) != expected:
        raise KerasInteropError(
            f"Unexpected BatchNormalization weight count: {len(w)} (expected {expected})."
        )

    idx = 0
    gamma = None
    beta = None

    if scale:
        gamma = np.asarray(w[idx])
        idx += 1
    if center:
        beta = np.asarray(w[idx])
        idx += 1

    moving_mean = np.asarray(w[idx])
    moving_var = np.asarray(w[idx + 1])

    if moving_mean.ndim != 1 or moving_var.ndim != 1:
        raise KerasInteropError("BatchNormalization moving statistics must be 1D.")
    if gamma is not None and gamma.ndim != 1:
        raise KerasInteropError("BatchNormalization gamma must be 1D.")
    if beta is not None and beta.ndim != 1:
        raise KerasInteropError("BatchNormalization beta must be 1D.")

    return gamma, beta, moving_mean, moving_var


def _infer_num_features(k_layer: Any) -> int:
    """
    Infer the channel count (num_features) from a built Keras BatchNormalization layer.

    Parameters
    ----------
    k_layer : Any
        Keras BatchNormalization layer.

    Returns
    -------
    int
        Number of channels/features.

    Raises
    ------
    KerasInteropError
        If feature count cannot be inferred.
    """
    _, _, moving_mean, _ = _extract_bn_weights(k_layer)
    return int(moving_mean.shape[0])


def _require_axis_channel_first(k_layer: Any) -> None:
    """
    Require channel-first axis semantics for BatchNormalization conversion.

    Parameters
    ----------
    k_layer : Any
        Keras BatchNormalization layer.

    Raises
    ------
    KerasInteropError
        If `axis` is not 1 (channels-first convention for 2D/4D in Phase 1).
    """
    axis = getattr(k_layer, "axis", 1)
    if isinstance(axis, (tuple, list)):
        raise KerasInteropError(
            "BatchNormalization with multiple axes is not supported in Phase 1."
        )
    if int(axis) != 1:
        raise KerasInteropError(
            f"BatchNormalization axis={axis} not supported in Phase 1 "
            "(expected axis=1 for channels_first)."
        )


def _infer_input_rank(k_layer: Any) -> Optional[int]:
    """
    Best-effort inference of the input rank for a Keras layer.

    Parameters
    ----------
    k_layer : Any
        Keras BatchNormalization layer.

    Returns
    -------
    Optional[int]
        Inferred rank (2 or 4) when available, otherwise None.

    Notes
    -----
    Keras exposes shape metadata differently across versions and execution modes.
    This helper checks common attributes without relying on private APIs.
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
        # TensorShape -> tuple
        try:
            shp_t = tuple(shp)
        except Exception:
            try:
                shp_t = tuple(getattr(shp, "as_list")())
            except Exception:
                continue

        # drop possible None batch dimension
        if len(shp_t) == 0:
            continue
        # Some sources include batch dim; converter cares about rank including batch
        r = len(shp_t)
        if r in (2, 4):
            return r

    return None


@dataclass(frozen=True)
class BatchNormalizationConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.BatchNormalization` -> KeyDNN BatchNorm modules.

    The converter selects:
    - KeyDNN `BatchNorm1d` when input rank is inferred as 2
    - KeyDNN `BatchNorm2d` when input rank is inferred as 4 or is unavailable
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN BatchNorm module corresponding to a Keras BatchNormalization layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras BatchNormalization layer.
        ctx : Any
            Conversion context providing `device`.

        Returns
        -------
        Any
            Constructed KeyDNN BatchNorm module (`BatchNorm1d` or `BatchNorm2d`).

        Raises
        ------
        KerasInteropError
            If axis semantics are unsupported, device is not CPU, or weights are missing.
        """
        _require_axis_channel_first(k_layer)
        device = _require_cpu_device(ctx)

        num_features = _infer_num_features(k_layer)

        eps = float(getattr(k_layer, "epsilon", 1e-3))
        momentum = float(getattr(k_layer, "momentum", 0.99))

        center = bool(getattr(k_layer, "center", True))
        scale = bool(getattr(k_layer, "scale", True))
        affine = bool(center and scale)

        rank = _infer_input_rank(k_layer)

        from .....infrastructure.layers._batchnorm import BatchNorm1d, BatchNorm2d

        if rank == 2:
            return BatchNorm1d(
                num_features=int(num_features),
                device=device,
                eps=float(eps),
                momentum=float(momentum),
                affine=affine,
            )

        if rank == 4 or rank is None:
            return BatchNorm2d(
                num_features=int(num_features),
                device=device,
                eps=float(eps),
                momentum=float(momentum),
                affine=affine,
            )

        raise KerasInteropError(
            f"BatchNormalization input rank={rank} not supported in Phase 1 "
            "(supported ranks: 2 and 4)."
        )

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Copy Keras BatchNormalization weights into an existing KeyDNN BatchNorm module.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN BatchNorm module created by `build`.
        k_layer : Any
            Source Keras BatchNormalization layer.
        ctx : Any
            Conversion context (reserved for future use).

        Raises
        ------
        KerasInteropError
            If affine configurations are incompatible or required parameters are missing.
        """
        gamma, beta, moving_mean, moving_var = _extract_bn_weights(k_layer)

        # running stats
        copy_param_from_numpy(
            kd_layer.running_mean, moving_mean.astype(np.float32, copy=False)
        )
        copy_param_from_numpy(
            kd_layer.running_var, moving_var.astype(np.float32, copy=False)
        )

        if bool(getattr(kd_layer, "affine", False)):
            if gamma is None or beta is None:
                raise KerasInteropError(
                    "KeyDNN BatchNorm is affine but Keras BatchNormalization is missing gamma/beta."
                )
            if (
                getattr(kd_layer, "gamma", None) is None
                or getattr(kd_layer, "beta", None) is None
            ):
                raise KerasInteropError(
                    "KeyDNN BatchNorm affine parameters are missing."
                )
            copy_param_from_numpy(kd_layer.gamma, gamma.astype(np.float32, copy=False))
            copy_param_from_numpy(kd_layer.beta, beta.astype(np.float32, copy=False))
        else:
            if gamma is not None or beta is not None:
                raise KerasInteropError(
                    "Keras BatchNormalization has gamma/beta but KeyDNN BatchNorm was constructed with affine=False."
                )
