# src/keydnn/presentation/interops/keras/converters/conv2d.py
"""
Keras Conv2D layer conversion.

This module implements conversion from `tf.keras.layers.Conv2D` to KeyDNN's
`Conv2d` implementation.

Mapping summary
---------------
Keras Conv2D stores:
- kernel: (k_h, k_w, in_channels, out_channels)
- bias:   (out_channels,) if enabled
- strides: int or (s_h, s_w)
- padding: {"valid", "same"} or (p_h, p_w) depending on configuration
- data_format: {"channels_last", "channels_first"} (often defaults to channels_last)
- dilation_rate: int or (d_h, d_w)
- groups: int

KeyDNN Conv2d stores:
- weight: (out_channels, in_channels, k_h, k_w)
- bias:   (out_channels,) if enabled
- stride: (s_h, s_w)
- padding: (p_h, p_w)
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
  - kernel size is odd in both dimensions
  In this case, padding is mapped to (k_h//2, k_w//2).

Notes
-----
- Keras Conv2D activation is not fused into KeyDNN Conv2d in Phase 1.
  Activation should be represented as a separate layer and converted via
  activation converters.
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
    Reject fused, non-linear activations on Keras Conv2D in Phase 1.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2D layer.

    Raises
    ------
    KerasInteropError
        If the activation is not linear.
    """
    act = getattr(k_layer, "activation", None)
    name = _activation_name(act) or "linear"

    # Keras linear may appear as "linear" or as a callable with name "linear".
    if name != "linear":
        raise KerasInteropError(
            f"Keras Conv2D activation '{name}' not supported in Phase 1. "
            "Represent activation explicitly as a separate layer."
        )


def _require_channels_first(k_layer: Any) -> None:
    """
    Require NCHW semantics for conversion.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2D layer.

    Raises
    ------
    KerasInteropError
        If `data_format` is not 'channels_first'.
    """
    data_format = getattr(k_layer, "data_format", None) or "channels_last"
    if str(data_format).lower() != "channels_first":
        raise KerasInteropError(
            "Keras Conv2D with data_format='channels_last' is not supported in Phase 1. "
            "KeyDNN Conv2d assumes NCHW (channels_first)."
        )


def _require_no_groups_no_dilation(k_layer: Any) -> None:
    """
    Reject groups and dilation configurations not supported by KeyDNN Conv2d.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2D layer.

    Raises
    ------
    KerasInteropError
        If groups != 1 or dilation_rate != (1, 1).
    """
    groups = getattr(k_layer, "groups", 1)
    try:
        groups_i = int(groups)
    except Exception as e:  # pragma: no cover
        raise KerasInteropError(f"Invalid Conv2D groups value: {groups!r}.") from e
    if groups_i != 1:
        raise KerasInteropError(
            f"Keras Conv2D groups={groups_i} not supported in KeyDNN Conv2d."
        )

    dilation = getattr(k_layer, "dilation_rate", 1)
    d_h, d_w = _pair_int(dilation, name="dilation_rate")
    if (d_h, d_w) != (1, 1):
        raise KerasInteropError(
            f"Keras Conv2D dilation_rate={(d_h, d_w)} not supported in KeyDNN Conv2d."
        )


def _resolve_padding(
    k_layer: Any, *, kernel_size: Tuple[int, int], stride: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Resolve Keras Conv2D padding into a static (p_h, p_w) pair.

    Parameters
    ----------
    k_layer : Any
        Keras Conv2D layer.
    kernel_size : Tuple[int, int]
        Kernel size (k_h, k_w).
    stride : Tuple[int, int]
        Stride (s_h, s_w).

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

    if isinstance(padding, str):
        p = padding.lower()
        if p == "valid":
            return 0, 0
        if p == "same":
            # Static SAME padding is only safe when stride=1 and odd kernels.
            if stride != (1, 1):
                raise KerasInteropError(
                    "Keras Conv2D padding='same' with stride != (1, 1) is not supported "
                    "in Phase 1 because KeyDNN Conv2d uses static padding."
                )
            k_h, k_w = kernel_size
            if (k_h % 2) != 1 or (k_w % 2) != 1:
                raise KerasInteropError(
                    "Keras Conv2D padding='same' requires odd kernel sizes in Phase 1."
                )
            return k_h // 2, k_w // 2

        raise KerasInteropError(f"Unsupported Keras Conv2D padding='{padding}'.")

    return _pair_int(padding, name="padding")


def extract_conv2d_weights(k_layer: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract Conv2D weights from a Keras layer.

    Parameters
    ----------
    k_layer : Any
        A Keras Conv2D layer instance.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (kernel, bias) where:
        - kernel has shape (k_h, k_w, in_channels, out_channels)
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
            "Keras Conv2D has no weights. Make sure the Keras model is built "
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
            f"Unexpected Conv2D weight count: {len(w)} (expected 1 or 2)."
        )

    if kernel.ndim != 4:
        raise KerasInteropError(f"Conv2D kernel must be 4D, got shape={kernel.shape}.")
    if bias is not None and bias.ndim != 1:
        raise KerasInteropError(f"Conv2D bias must be 1D, got shape={bias.shape}.")

    return kernel, bias


@dataclass(frozen=True)
class Conv2DConverter(BaseConverter[Any]):
    """
    Converter for `tf.keras.layers.Conv2D` to KeyDNN `Conv2d`.

    This converter constructs a KeyDNN `Conv2d` layer and copies kernel and
    bias parameters with correct layout transposition.

    Phase 1 behavior
    ----------------
    - Requires NCHW (`data_format="channels_first"`).
    - Rejects dilation and groups.
    - Supports `padding` in {"valid", "same"} under constraints or explicit pairs.
    """

    def build(self, k_layer: Any, ctx: Any) -> Any:
        """
        Construct a KeyDNN Conv2d module corresponding to a Keras Conv2D layer.

        Parameters
        ----------
        k_layer : Any
            Source Keras Conv2D layer.
        ctx : Any
            Conversion context providing `device` and `dtype`.

        Returns
        -------
        Any
            Constructed KeyDNN Conv2d module.

        Raises
        ------
        KerasInteropError
            If unsupported attributes are present or weights/config are invalid.
        """
        _require_linear_activation(k_layer)
        _require_channels_first(k_layer)
        _require_no_groups_no_dilation(k_layer)

        kernel, bias = extract_conv2d_weights(k_layer)
        k_h, k_w, in_c, out_c = kernel.shape

        stride = _pair_int(getattr(k_layer, "strides", 1), name="strides")
        padding = _resolve_padding(
            k_layer, kernel_size=(int(k_h), int(k_w)), stride=stride
        )

        device = resolve_device(ctx)
        dtype = resolve_dtype(ctx)

        from .....infrastructure.convolution._conv2d_module import Conv2d

        use_bias = bool(getattr(k_layer, "use_bias", bias is not None))
        kd = Conv2d(
            in_channels=int(in_c),
            out_channels=int(out_c),
            kernel_size=(int(k_h), int(k_w)),
            stride=(int(stride[0]), int(stride[1])),
            padding=(int(padding[0]), int(padding[1])),
            bias=use_bias,
            device=device,
            dtype=dtype,
        )
        return kd

    def load_weights(self, kd_layer: Any, k_layer: Any, ctx: Any) -> None:
        """
        Copy Keras Conv2D parameters into an existing KeyDNN Conv2d module.

        Parameters
        ----------
        kd_layer : Any
            Destination KeyDNN Conv2d module created by `build`.
        k_layer : Any
            Source Keras Conv2D layer.
        ctx : Any
            Conversion context (reserved for future use).

        Raises
        ------
        KerasInteropError
            If the KeyDNN layer is missing parameters or bias presence is incompatible.
        """
        kernel, bias = extract_conv2d_weights(k_layer)
        k_h, k_w, in_c, out_c = kernel.shape

        if getattr(kd_layer, "weight", None) is None:
            raise KerasInteropError("KeyDNN Conv2d is missing weight parameter.")

        # Keras: (k_h, k_w, in_c, out_c) -> KeyDNN: (out_c, in_c, k_h, k_w)
        w_np = kernel.transpose(3, 2, 0, 1).astype(np.float32, copy=False)
        copy_param_from_numpy(kd_layer.weight, w_np)

        if bias is not None:
            if getattr(kd_layer, "bias", None) is None:
                raise KerasInteropError(
                    "Keras Conv2D has bias but KeyDNN Conv2d.bias is None."
                )
            if bias.shape[0] != out_c:
                raise KerasInteropError(
                    f"Conv2D bias shape mismatch: expected ({out_c},), got {bias.shape}."
                )
            copy_param_from_numpy(kd_layer.bias, bias.astype(np.float32, copy=False))
