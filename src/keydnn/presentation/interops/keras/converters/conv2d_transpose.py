# src/keydnn/presentation/interops/keras/converters/conv2d_transpose.py
"""
Keras Conv2DTranspose layer conversion.

This module implements conversion from `tf.keras.layers.Conv2DTranspose` to
KeyDNN's `Conv2dTranspose` implementation.

Mapping summary
---------------
Keras Conv2DTranspose stores:
- kernel: (k_h, k_w, out_channels, in_channels)
- bias:   (out_channels,) if enabled
- strides: int or (s_h, s_w)
- padding: {"valid", "same"} (string)
- output_padding: None or int or (op_h, op_w)
- data_format: {"channels_last", "channels_first"} (often defaults to channels_last)
- dilation_rate: int or (d_h, d_w)
- groups: int (TF/Keras versions may expose this depending on backend)

KeyDNN Conv2dTranspose stores:
- weight: (in_channels, out_channels, k_h, k_w)  [IOHW]
- bias:   (out_channels,) if enabled
- stride: (s_h, s_w)
- padding: (p_h, p_w)  [static padding]
- output_padding: (op_h, op_w)
- assumes NCHW tensor layout
- does not support dilation or groups

Therefore the kernel must be transposed when copied into KeyDNN:
    keydnn.weight = keras.kernel.transpose(3, 2, 0, 1)

Phase 1 limitations
-------------------
- Only `data_format="channels_first"` is supported (KeyDNN assumes NCHW).
- `groups` must be 1.
- `dilation_rate` must be (1, 1).
- `padding="same"` is supported only when:
  - strides == (1, 1)
  - output_padding is None or (0, 0)
  - kernel size is odd in both dimensions
  In this case, padding is mapped to (k_h//2, k_w//2).
- `padding="valid"` maps to (0, 0).

Notes
-----
- Keras Conv2DTranspose activation is not fused into KeyDNN Conv2dTranspose
  in Phase 1. Activation should be represented as a separate layer and
  converted via activation converters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ._base import BaseConverter, KerasInteropError
from ._utils import copy_param_from_numpy, resolve_device, resolve_dtype


def _pair_int(v: Any, *, name: str) -> Tuple[int, int]:
    """
    Normalize an integer-or-pair value into a pair of integers.

    Parameters
    ----------
    v : Any
        Value that is either an integer or a length-2 tuple/list of integers.
    name : str
        Attribute name for error reporting.

    Returns
    -------
    Tuple[int, int]
        Normalized (h, w) pair.

    Raises
    ------
    KerasInteropError
        If the value cannot be normalized to a pair of integers.
    """
    if v is None:
        raise KerasInteropError(f"Expected {name} to be int or pair, got None.")
    if isinstance(v, (int, np.integer)):
        return int(v), int(v)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise KerasInteropError(f"Expected {name} to be int or pair, got {v!r}.")


def _activation_name(act: Any) -> Optional[str]:
    """
    Normalize a Keras activation specification to a short name when possible.

    Parameters
    ----------
    act : Any
        Keras activation spec (string/callable/object with `.name`).

    Returns
    -------
    Optional[str]
        Lowercased activation name if detected, otherwise None.
    """
    if act is None:
        return None
    if isinstance(act, str):
        return act.lower()
    name = getattr(act, "__name__", None) or getattr(act, "name", None)
    if name is None:
        return None
    return str(name).lower()


def _require_linear_activation(k_layer: Any) -> None:
    """
    Reject fused, non-linear activations on Keras Conv2DTranspose in Phase 1.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2DTranspose layer.

    Raises
    ------
    KerasInteropError
        If the activation is not linear.
    """
    act = getattr(k_layer, "activation", None)
    name = _activation_name(act) or "linear"
    if name != "linear":
        raise KerasInteropError(
            f"Keras Conv2DTranspose activation '{name}' not supported in Phase 1. "
            "Represent activation explicitly as a separate layer."
        )


def _require_channels_first(k_layer: Any) -> None:
    """
    Require NCHW semantics for conversion.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2DTranspose layer.

    Raises
    ------
    KerasInteropError
        If `data_format` is not 'channels_first'.
    """
    data_format = getattr(k_layer, "data_format", None) or "channels_last"
    if str(data_format).lower() != "channels_first":
        raise KerasInteropError(
            "Keras Conv2DTranspose with data_format='channels_last' is not supported in Phase 1. "
            "KeyDNN Conv2dTranspose assumes NCHW (channels_first)."
        )


def _require_no_groups_no_dilation(k_layer: Any) -> None:
    """
    Reject groups and dilation configurations not supported by KeyDNN Conv2dTranspose.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2DTranspose layer.

    Raises
    ------
    KerasInteropError
        If groups != 1 or dilation_rate != (1, 1).
    """
    groups = getattr(k_layer, "groups", 1)
    try:
        groups_i = int(groups)
    except Exception as e:  # pragma: no cover
        raise KerasInteropError(
            f"Invalid Conv2DTranspose groups value: {groups!r}."
        ) from e
    if groups_i != 1:
        raise KerasInteropError(
            f"Keras Conv2DTranspose groups={groups_i} not supported in KeyDNN Conv2dTranspose."
        )

    dilation = getattr(k_layer, "dilation_rate", 1)
    d_h, d_w = _pair_int(dilation, name="dilation_rate")
    if (d_h, d_w) != (1, 1):
        raise KerasInteropError(
            f"Keras Conv2DTranspose dilation_rate={(d_h, d_w)} not supported in KeyDNN Conv2dTranspose."
        )


def _resolve_output_padding(k_layer: Any) -> Tuple[int, int]:
    """
    Resolve Keras Conv2DTranspose output_padding into a static (op_h, op_w) pair.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2DTranspose layer.

    Returns
    -------
    Tuple[int, int]
        Output padding pair (op_h, op_w).
    """
    op = getattr(k_layer, "output_padding", None)
    if op is None:
        return 0, 0
    return _pair_int(op, name="output_padding")


def _resolve_padding(
    k_layer: Any,
    *,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_padding: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Resolve Keras Conv2DTranspose padding into a static (p_h, p_w) pair.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2DTranspose layer.
    kernel_size : Tuple[int, int]
        Kernel size (k_h, k_w).
    stride : Tuple[int, int]
        Stride (s_h, s_w).
    output_padding : Tuple[int, int]
        Output padding (op_h, op_w).

    Returns
    -------
    Tuple[int, int]
        Padding pair (p_h, p_w).

    Raises
    ------
    KerasInteropError
        If padding configuration cannot be mapped to KeyDNN's static padding.
    """
    padding = getattr(k_layer, "padding", "valid")

    if not isinstance(padding, str):
        raise KerasInteropError(
            f"Unsupported Keras Conv2DTranspose padding type: {type(padding)!r}."
        )

    p = padding.lower()
    if p == "valid":
        return 0, 0

    if p == "same":
        # Static SAME mapping is constrained in Phase 1.
        if stride != (1, 1):
            raise KerasInteropError(
                "Keras Conv2DTranspose padding='same' with stride != (1, 1) is not supported "
                "in Phase 1 because KeyDNN Conv2dTranspose uses static padding."
            )
        if output_padding != (0, 0):
            raise KerasInteropError(
                "Keras Conv2DTranspose padding='same' with non-zero output_padding is not supported "
                "in Phase 1 because KeyDNN Conv2dTranspose uses static padding."
            )
        k_h, k_w = kernel_size
        if (k_h % 2) != 1 or (k_w % 2) != 1:
            raise KerasInteropError(
                "Keras Conv2DTranspose padding='same' requires odd kernel sizes in Phase 1."
            )
        return k_h // 2, k_w // 2

    raise KerasInteropError(f"Unsupported Keras Conv2DTranspose padding='{padding}'.")


def extract_conv2d_transpose_weights(
    k_layer: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract Conv2DTranspose weights from a Keras layer.

    Parameters
    ----------
    k_layer : Any
        A Keras Conv2DTranspose layer instance.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (kernel, bias) where:
        - kernel has shape (k_h, k_w, out_channels, in_channels)
        - bias has shape (out_channels,) or is None

    Raises
    ------
    KerasInteropError
        If the Keras layer has no weights (not built), unexpected weight count,
        or invalid tensor ranks.
    """
    w = k_layer.get_weights()
    if len(w) == 0:
        raise KerasInteropError(
            "Keras Conv2DTranspose has no weights. Make sure the Keras model is built "
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
            f"Unexpected Conv2DTranspose weight count: {len(w)} (expected 1 or 2)."
        )

    if kernel.ndim != 4:
        raise KerasInteropError(
            f"Conv2DTranspose kernel must be 4D, got shape={kernel.shape}."
        )
    if bias is not None and bias.ndim != 1:
        raise KerasInteropError(
            f"Conv2DTranspose bias must be 1D, got shape={bias.shape}."
        )

    return kernel, bias


@dataclass(frozen=True)
class Conv2DTransposeConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.Conv2DTranspose` to KeyDNN `Conv2dTranspose`.

    This converter constructs a KeyDNN `Conv2dTranspose` layer and copies kernel and
    bias parameters with correct layout transposition.

    Phase 1 behavior
    ----------------
    - Requires NCHW (`data_format="channels_first"`).
    - Rejects dilation and groups.
    - Supports `padding` in {"valid", "same"} under constraints.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN Conv2dTranspose module corresponding to a Keras Conv2DTranspose layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras Conv2DTranspose layer.
        ctx : Any
            Conversion context providing `device` and `dtype`.

        Returns
        -------
        Any
            Constructed KeyDNN Conv2dTranspose module.

        Raises
        ------
        KerasInteropError
            If unsupported attributes are present or weights/config are invalid.
        """
        _require_linear_activation(k_layer)
        _require_channels_first(k_layer)
        _require_no_groups_no_dilation(k_layer)

        kernel, bias = extract_conv2d_transpose_weights(k_layer)
        k_h, k_w, out_c, in_c = kernel.shape

        stride = _pair_int(getattr(k_layer, "strides", 1), name="strides")
        output_padding = _resolve_output_padding(k_layer)
        padding = _resolve_padding(
            k_layer,
            kernel_size=(int(k_h), int(k_w)),
            stride=stride,
            output_padding=output_padding,
        )

        device = resolve_device(ctx)
        dtype = resolve_dtype(ctx)

        from .....infrastructure.convolution.transpose._conv2d_transpose_module import (
            Conv2dTranspose,
        )

        use_bias = bool(getattr(k_layer, "use_bias", bias is not None))
        kd = Conv2dTranspose(
            in_channels=int(in_c),
            out_channels=int(out_c),
            kernel_size=(int(k_h), int(k_w)),
            stride=(int(stride[0]), int(stride[1])),
            padding=(int(padding[0]), int(padding[1])),
            output_padding=(int(output_padding[0]), int(output_padding[1])),
            bias=use_bias,
            device=device,
            dtype=dtype,
        )
        return kd

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Copy Keras Conv2DTranspose parameters into an existing KeyDNN Conv2dTranspose module.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN Conv2dTranspose module created by `build`.
        k_layer : Any
            Source Keras Conv2DTranspose layer.
        ctx : Any
            Conversion context (reserved for future use).

        Raises
        ------
        KerasInteropError
            If the KeyDNN layer is missing parameters or bias presence is incompatible.
        """
        kernel, bias = extract_conv2d_transpose_weights(k_layer)
        k_h, k_w, out_c, in_c = kernel.shape

        if getattr(kd_layer, "weight", None) is None:
            raise KerasInteropError(
                "KeyDNN Conv2dTranspose is missing weight parameter."
            )

        # Keras: (k_h, k_w, out_c, in_c) -> KeyDNN: (in_c, out_c, k_h, k_w) [IOHW]
        w_np = kernel.transpose(3, 2, 0, 1).astype(np.float32, copy=False)
        copy_param_from_numpy(kd_layer.weight, w_np)

        if bias is not None:
            if getattr(kd_layer, "bias", None) is None:
                raise KerasInteropError(
                    "Keras Conv2DTranspose has bias but KeyDNN Conv2dTranspose.bias is None."
                )
            if bias.shape[0] != out_c:
                raise KerasInteropError(
                    f"Conv2DTranspose bias shape mismatch: expected ({out_c},), got {bias.shape}."
                )
            copy_param_from_numpy(kd_layer.bias, bias.astype(np.float32, copy=False))
