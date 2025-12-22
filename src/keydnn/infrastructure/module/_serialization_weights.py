from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..encoding._b64 import ndarray_to_payload, payload_to_ndarray


def extract_state_payload(model: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameters into JSON payloads keyed by parameter name.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    out: dict[str, dict[str, Any]] = {}
    for name, p in named_params():
        t = getattr(p, "data", p)
        arr = t.to_numpy()
        out[str(name)] = ndarray_to_payload(np.asarray(arr))
    return out


def load_state_payload_(model: Any, payloads: Dict[str, Dict[str, Any]]) -> None:
    """
    In-place load of parameters from JSON payloads.

    Raises
    ------
    KeyError / ValueError on missing keys or shape mismatch.
    """
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        raise AttributeError("Model must implement named_parameters().")

    for name, p in named_params():
        key = str(name)
        if key not in payloads:
            raise KeyError(f"Missing parameter in checkpoint: '{key}'")

        t = getattr(p, "data", p)
        arr = payload_to_ndarray(payloads[key])

        if tuple(arr.shape) != tuple(t.shape):
            raise ValueError(
                f"Shape mismatch for '{key}': model {t.shape} vs ckpt {arr.shape}"
            )

        # Option: enforce float32 if your Tensor storage is float32-only
        arr = arr.astype(np.float32, copy=False)

        t.copy_from_numpy(arr)
