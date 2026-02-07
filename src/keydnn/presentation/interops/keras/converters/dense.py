# dense.py
"""
Keras Dense layer conversion.

This module implements conversion from `tf.keras.layers.Dense` to KeyDNN's
lazy `Dense` implementation.

Mapping summary
---------------
Keras Dense stores:
- kernel: (in_features, out_features)
- bias:   (out_features,) if enabled

KeyDNN Dense stores:
- weight: (out_features, in_features)
- bias:   (out_features,) if enabled

Therefore the kernel must be transposed when copied into KeyDNN:
    keydnn.weight = keras.kernel.T
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ._base import BaseConverter, KerasInteropError
from ._utils import (
    copy_param_from_numpy,
    is_linear_activation,
    resolve_device,
    resolve_dtype,
)


def extract_dense_weights(k_layer: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract Dense weights from a Keras layer.

    Parameters
    ----------
    k_layer : Any
        A Keras Dense layer instance.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (kernel, bias) where:
        - kernel has shape (in_features, out_features)
        - bias has shape (out_features,) or is None

    Raises
    ------
    KerasInteropError
        If the Keras layer has no weights (not built), unexpected weight count,
        or invalid tensor ranks.
    """
    w = k_layer.get_weights()
    if len(w) == 0:
        raise KerasInteropError(
            "Keras Dense has no weights. Make sure the Keras model is built "
            "(e.g., call it once or load from file)."
        )
    if len(w) == 1:
        kernel = np.asarray(w[0])
        bias = None
    elif len(w) == 2:
        kernel = np.asarray(w[0])
        bias = np.asarray(w[1])
    else:
        raise KerasInteropError(
            f"Unexpected Dense weight count: {len(w)} (expected 1 or 2)."
        )

    if kernel.ndim != 2:
        raise KerasInteropError(f"Dense kernel must be 2D, got shape={kernel.shape}.")
    if bias is not None and bias.ndim != 1:
        raise KerasInteropError(f"Dense bias must be 1D, got shape={bias.shape}.")

    return kernel, bias


@dataclass(frozen=True)
class DenseConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.Dense` to KeyDNN `Dense`.

    Parameters
    ----------
    allow_non_linear_activation : bool, optional
        If False, conversion rejects non-linear activations and requires
        activations to be represented explicitly as separate KeyDNN modules.
        Defaults to False.
    """

    allow_non_linear_activation: bool = False

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN Dense module corresponding to a Keras Dense layer.

        This method materializes parameters eagerly based on the extracted
        Keras kernel shape so that `weight`/`bias` exist before weight loading.

        Parameters
        ----------
        k_layer : Any
            Source Keras Dense layer.
        ctx : Any
            Conversion context providing `device` and `dtype`.

        Returns
        -------
        Any
            Constructed KeyDNN Dense module.

        Raises
        ------
        KerasInteropError
            If activation is non-linear and not allowed, or if Keras weights are
            missing/invalid.
        """
        kernel, bias = extract_dense_weights(k_layer)
        in_features, out_features = kernel.shape

        act = getattr(k_layer, "activation", None)
        if not is_linear_activation(act) and not self.allow_non_linear_activation:
            act_name = (
                getattr(act, "__name__", None) or getattr(act, "name", None) or str(act)
            )
            raise KerasInteropError(
                f"Keras Dense activation '{act_name}' not supported in Phase 1."
            )

        device = resolve_device(ctx)
        dtype = resolve_dtype(ctx)

        from .....infrastructure.fully_connected import Dense

        use_bias = bool(getattr(k_layer, "use_bias", bias is not None))
        kd = Dense(
            out_features=int(out_features), bias=use_bias, device=device, dtype=dtype
        )

        kd._materialize(int(in_features), device=device)
        return kd

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Copy Keras Dense parameters into an existing KeyDNN Dense module.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN Dense module created by `build`.
        k_layer : Any
            Source Keras Dense layer.
        ctx : Any
            Conversion context (reserved for future use).

        Raises
        ------
        KerasInteropError
            If the KeyDNN layer is not built or bias presence is incompatible.
        """
        kernel, bias = extract_dense_weights(k_layer)
        in_features, out_features = kernel.shape

        if getattr(kd_layer, "weight", None) is None:
            raise KerasInteropError(
                "KeyDNN Dense is not built; expected weight to exist."
            )

        w_np = kernel.T.astype(np.float32, copy=False)
        copy_param_from_numpy(kd_layer.weight, w_np)

        if bias is not None:
            if getattr(kd_layer, "bias", None) is None:
                raise KerasInteropError(
                    "Keras Dense has bias but KeyDNN Dense.bias is None."
                )
            copy_param_from_numpy(kd_layer.bias, bias.astype(np.float32, copy=False))
