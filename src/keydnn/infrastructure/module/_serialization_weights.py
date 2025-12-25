from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..encoding._b64 import ndarray_to_payload, payload_to_ndarray


def extract_state_payload(model: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameters into JSON payloads keyed by parameter name.

    Backward compatibility
    ----------------------
    This function supports both legacy and current storage conventions:

    - Legacy: Parameter/Tensor `.data` was itself a Tensor-like object that
      implements `to_numpy()`.
    - Current: Parameter/Tensor `.data` returns the underlying storage:
        - CPU: `np.ndarray`
        - CUDA: device pointer handle (int). CUDA parameters are not serializable
          via JSON payloads unless an explicit device->host copy is implemented.

    Notes
    -----
    - Serialization is defined only for CPU-resident parameters.
    - If a CUDA parameter is encountered, this function raises a clear error.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    out: dict[str, dict[str, Any]] = {}

    for name, p in named_params():
        # `p` is typically a Parameter, but we keep this flexible.
        t = getattr(p, "data", p)

        # 1) If it's already a NumPy array (new CPU behavior), use it directly.
        if isinstance(t, np.ndarray):
            arr = t

        # 2) If it's Tensor/Parameter-like (legacy behavior), call to_numpy().
        elif hasattr(t, "to_numpy") and callable(getattr(t, "to_numpy")):
            arr = t.to_numpy()

        # 3) If `p` itself is Tensor/Parameter-like and `data` was missing/odd.
        elif hasattr(p, "to_numpy") and callable(getattr(p, "to_numpy")):
            arr = p.to_numpy()

        # 4) CUDA devptr (new CUDA behavior): cannot serialize without copy-back.
        elif isinstance(t, int):
            raise RuntimeError(
                f"Cannot serialize CUDA parameter '{name}': got device pointer handle (int). "
                "Move parameters to CPU before saving, or implement a device->host copy path."
            )

        else:
            raise TypeError(
                f"Unsupported parameter storage for '{name}': type={type(t)!r}. "
                "Expected np.ndarray or Tensor/Parameter with to_numpy()."
            )

        out[str(name)] = ndarray_to_payload(np.asarray(arr))

    return out


def load_state_payload_(model: Any, payloads: Dict[str, Dict[str, Any]]) -> None:
    """
    In-place load of parameters from JSON payloads.

    Backward compatibility
    ----------------------
    Supports both legacy and current parameter storage conventions:

    - Legacy: `p.data` is Tensor-like and supports:
        - `.shape`
        - `.copy_from_numpy(arr)`
    - Current: `p.data` is the underlying storage:
        - CPU: `np.ndarray` (we copy into it)
        - CUDA: device pointer handle (int) -> not supported here without an
          explicit host->device copy path.

    Raises
    ------
    KeyError
        If a parameter key is missing in the checkpoint.
    ValueError
        If a parameter shape does not match.
    RuntimeError
        If attempting to load into a CUDA-resident parameter without a copy path.
    TypeError
        If parameter storage is unsupported.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    for name, p in named_params():
        key = str(name)
        if key not in payloads:
            raise KeyError(f"Missing parameter in checkpoint: '{key}'")

        # Target storage (legacy: tensor-like; new: ndarray or devptr int)
        t = getattr(p, "data", p)
        arr = payload_to_ndarray(payloads[key])

        # Enforce float32 if your framework stores params as float32-only.
        arr = arr.astype(np.float32, copy=False)

        # ---- Determine target shape safely ----
        if hasattr(t, "shape"):
            target_shape = tuple(getattr(t, "shape"))
        elif isinstance(t, np.ndarray):
            target_shape = tuple(t.shape)
        else:
            target_shape = None

        if target_shape is None:
            raise TypeError(
                f"Unsupported parameter target for '{key}': type={type(t)!r} "
                "(missing 'shape')."
            )

        if tuple(arr.shape) != tuple(target_shape):
            raise ValueError(
                f"Shape mismatch for '{key}': model {target_shape} vs ckpt {arr.shape}"
            )

        # ---- Write into the target ----
        # 1) Legacy tensor-like path: preferred if available.
        if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
            t.copy_from_numpy(arr)

        # 2) New CPU path: t is a NumPy array buffer.
        elif isinstance(t, np.ndarray):
            # In-place copy; preserves the existing array object referenced by Parameter.
            t[...] = arr

        # 3) CUDA path (devptr handle): not supported in JSON load for now.
        elif isinstance(t, int):
            raise RuntimeError(
                f"Cannot load CUDA parameter '{key}' from JSON payload into device memory. "
                "Move model parameters to CPU before loading, or implement a host->device "
                "upload path (cuda_malloc + cuda_memcpy_h2d) for parameters."
            )

        else:
            raise TypeError(
                f"Unsupported parameter target for '{key}': type={type(t)!r}. "
                "Expected Tensor-like with copy_from_numpy(), or np.ndarray."
            )
