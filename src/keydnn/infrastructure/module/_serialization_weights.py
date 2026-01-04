"""
Weight (parameter) serialization helpers.

This module implements a lightweight, JSON-friendly checkpoint format for model
parameters by converting arrays to/from an encoded payload representation
(typically base64 + dtype/shape metadata), using helpers in `infrastructure.encoding._b64`.

Design goals
------------
- **Device-agnostic**: parameters may live on CPU or CUDA as long as they expose
  a host-facing conversion API.
- **Duck-typed**: no hard dependency on a specific `Tensor`/`Parameter` class.
  The functions operate on "parameter-like" objects discovered via
  `model.named_parameters()`.
- **Stable keys**: parameters are stored in a dict keyed by the string parameter
  name returned by `named_parameters()`.

Expected interfaces
-------------------
Parameter / tensor-like objects are expected to support one of the following:

Read (extract)
- Preferred: `p.to_numpy() -> np.ndarray` (may perform D2H if needed)
- Fallback: `p.data` is already an `np.ndarray`
- Legacy fallback: `p.data.to_numpy() -> np.ndarray`

Write (load)
- Preferred: `p.copy_from_numpy(arr) -> None` (may allocate and/or perform H2D)
- Fallback: `p.data` is an `np.ndarray` that can be written in-place
- Legacy fallback: `p.data.copy_from_numpy(arr) -> None`

Shape checking
--------------
When loading, the logical shape is obtained from `p.shape` (preferred), falling
back to `p.data.shape` if necessary. A mismatch between model and checkpoint
shapes raises `ValueError`.

Payload format
--------------
This module does not define the binary encoding itself. It delegates to:
- `ndarray_to_payload(arr) -> dict[str, Any]`
- `payload_to_ndarray(payload) -> np.ndarray`

Those helpers are responsible for ensuring the payload is JSON-serializable.

Notes
-----
- `load_state_payload_()` currently coerces incoming arrays to `float32`. This
  matches common training defaults and avoids dtype mismatches, but it assumes
  the model expects float32 weights.
- Functions are intentionally minimal and avoid optimizer state, buffers, or
  non-parameter metadata. Extend at a higher layer if needed.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..encoding._b64 import ndarray_to_payload, payload_to_ndarray


def _param_to_numpy(p: Any) -> np.ndarray:
    """
    Convert a parameter/tensor-like object to a host NumPy array.

    This helper centralizes the supported read paths for checkpoint extraction.
    It prefers converting through the object API rather than reaching into
    storage, allowing CUDA-backed parameters to perform a D2H transfer.

    Contract
    --------
    Supported conversion paths, in priority order:

    1) Preferred: `p.to_numpy()` if present and callable.
       - Expected to return an array-like object convertible to `np.ndarray`.
    2) Fallback: if `p.data` is already a `np.ndarray`, return it as an array.
    3) Legacy fallback: if `p.data` is tensor-like and has `to_numpy()`, call it.

    Parameters
    ----------
    p:
        Parameter-like object to convert.

    Returns
    -------
    np.ndarray
        A host-resident NumPy array view/copy of the parameter data.

    Raises
    ------
    TypeError
        If no supported conversion path exists.
    """
    # 1) Preferred: serialize via the object API, not storage.
    to_np = getattr(p, "to_numpy", None)
    if callable(to_np):
        arr = to_np()
        return np.asarray(arr)

    # 2) New CPU storage convention: p.data is np.ndarray
    t = getattr(p, "data", None)
    if isinstance(t, np.ndarray):
        return np.asarray(t)

    # 3) Legacy: p.data is tensor-like
    if t is not None:
        to_np2 = getattr(t, "to_numpy", None)
        if callable(to_np2):
            arr = to_np2()
            return np.asarray(arr)

    raise TypeError(
        f"Unsupported parameter storage for serialization: type(p)={type(p)!r}, "
        f"type(p.data)={type(getattr(p, 'data', None))!r}. "
        "Expected p.to_numpy() or np.ndarray storage."
    )


def _param_shape(p: Any) -> tuple[int, ...] | None:
    """
    Determine the logical shape of a parameter/tensor-like object.

    This is used during load to validate checkpoint compatibility before writing.

    Resolution order
    ----------------
    1) Preferred: `p.shape` (works even if `p.data` is a device pointer / handle)
    2) Fallback: if `p.data` is `np.ndarray`, use `p.data.shape`
    3) Legacy: if `p.data` is tensor-like and exposes `.shape`, use that

    Parameters
    ----------
    p:
        Parameter-like object whose shape should be inspected.

    Returns
    -------
    tuple[int, ...] | None
        The shape as a tuple, or `None` if no supported shape source exists.
    """
    sh = getattr(p, "shape", None)
    if sh is not None:
        try:
            return tuple(sh)
        except TypeError:
            pass

    t = getattr(p, "data", None)
    if isinstance(t, np.ndarray):
        return tuple(t.shape)

    # Legacy: p.data tensor-like has shape
    if t is not None and hasattr(t, "shape"):
        try:
            return tuple(getattr(t, "shape"))
        except TypeError:
            pass

    return None


def _param_copy_from_numpy(p: Any, arr: np.ndarray) -> None:
    """
    Write a host NumPy array into a parameter/tensor-like object.

    This helper centralizes the supported write paths for checkpoint loading.
    It prefers writing through the object API so CUDA-backed parameters can
    allocate and/or perform an H2D transfer.

    Contract
    --------
    Supported assignment paths, in priority order:

    1) Preferred: `p.copy_from_numpy(arr)` if present and callable.
       - Expected to handle CPU copy or H2D upload (and allocation if needed).
    2) Fallback: if `p.data` is a `np.ndarray`, write in-place via `t[...] = arr`.
    3) Legacy fallback: if `p.data` is tensor-like and supports `copy_from_numpy`,
       call `p.data.copy_from_numpy(arr)`.

    Parameters
    ----------
    p:
        Parameter-like object to receive the data.
    arr:
        Host-resident NumPy array containing the parameter values.

    Raises
    ------
    TypeError
        If no supported assignment path exists.
    """
    cfn = getattr(p, "copy_from_numpy", None)
    if callable(cfn):
        cfn(arr)
        return

    t = getattr(p, "data", None)
    if isinstance(t, np.ndarray):
        t[...] = arr
        return

    if t is not None:
        cfn2 = getattr(t, "copy_from_numpy", None)
        if callable(cfn2):
            cfn2(arr)
            return

    raise TypeError(
        f"Unsupported parameter target for load: type(p)={type(p)!r}, "
        f"type(p.data)={type(getattr(p, 'data', None))!r}. "
        "Expected copy_from_numpy() or np.ndarray storage."
    )


def extract_state_payload(model: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract model parameters into JSON-serializable payloads.

    This function iterates `model.named_parameters()` and produces a dict mapping
    each parameter name to an encoded payload (as returned by `ndarray_to_payload`).

    Device behavior
    ---------------
    Device-agnostic as long as parameters implement `to_numpy()` for conversion
    to a host NumPy array (e.g., CUDA parameters perform D2H internally).

    Parameters
    ----------
    model:
        Module/model-like object implementing `named_parameters()` yielding
        `(name, param)` pairs.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from parameter name to a JSON-serializable ndarray payload.

    Raises
    ------
    AttributeError
        If `model` does not implement a callable `named_parameters()`.
    TypeError
        If a parameter cannot be converted to a NumPy array via supported paths.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    out: dict[str, dict[str, Any]] = {}

    for name, p in named_params():
        arr = _param_to_numpy(p)
        out[str(name)] = ndarray_to_payload(np.asarray(arr))

    return out


def load_state_payload_(model: Any, payloads: Dict[str, Dict[str, Any]]) -> None:
    """
    Load parameters in-place from JSON-serializable payloads.

    This function iterates `model.named_parameters()` and, for each parameter:
    - finds the matching payload by name,
    - decodes it to a NumPy array via `payload_to_ndarray`,
    - validates shape compatibility against the target parameter,
    - writes values into the target via `_param_copy_from_numpy()`.

    Device behavior
    ---------------
    Device-agnostic as long as the target parameters implement `copy_from_numpy()`
    (e.g., CUDA parameters perform H2D internally and allocate if needed).

    Dtype behavior
    --------------
    Decoded arrays are coerced to `float32` via `astype(np.float32, copy=False)`.
    This assumes the model weights are stored/expected as float32.

    Parameters
    ----------
    model:
        Module/model-like object implementing `named_parameters()` yielding
        `(name, param)` pairs.
    payloads:
        Mapping from parameter name to an ndarray payload.

    Raises
    ------
    AttributeError
        If `model` does not implement a callable `named_parameters()`.
    KeyError
        If a parameter name from the model is missing in `payloads`.
    TypeError
        If a target parameter does not expose a discoverable shape or cannot
        accept assignment via supported paths.
    ValueError
        If the checkpoint array shape does not match the model parameter shape.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    for name, p in named_params():
        key = str(name)
        if key not in payloads:
            raise KeyError(f"Missing parameter in checkpoint: '{key}'")

        arr = payload_to_ndarray(payloads[key]).astype(np.float32, copy=False)

        target_shape = _param_shape(p)
        if target_shape is None:
            raise TypeError(
                f"Unsupported parameter target for '{key}': type={type(p)!r} "
                "(missing 'shape')."
            )

        if tuple(arr.shape) != tuple(target_shape):
            raise ValueError(
                f"Shape mismatch for '{key}': model {target_shape} vs ckpt {arr.shape}"
            )

        _param_copy_from_numpy(p, arr)
