from __future__ import annotations

import base64
from typing import Any, Dict

import numpy as np


def bytes_to_b64_str(b: bytes) -> str:
    """
    Encode raw bytes into a base64 ASCII string (JSON-safe).
    """
    return base64.b64encode(b).decode("ascii")


def b64_str_to_bytes(s: str) -> bytes:
    """
    Decode a base64 ASCII string back into raw bytes.
    """
    return base64.b64decode(s.encode("ascii"))


def ndarray_to_payload(arr: np.ndarray) -> Dict[str, Any]:
    """
    Serialize a NumPy ndarray into a JSON-safe payload.

    Returns
    -------
    dict
        {
          "b64": "<base64>",
          "dtype": "<numpy dtype str>",
          "shape": [...],
          "order": "C"
        }
    """
    a = np.asarray(arr)
    # Ensure stable dtype representation. You can decide to always save float32
    # if you want: a = a.astype(np.float32, copy=False)
    payload = {
        "b64": bytes_to_b64_str(a.tobytes(order="C")),
        "dtype": a.dtype.str,  # e.g. "<f4"
        "shape": list(a.shape),
        "order": "C",
    }
    return payload


def payload_to_ndarray(payload: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize a JSON payload back into a NumPy ndarray.

    Notes
    -----
    - Uses np.frombuffer (zero-copy view on the bytes object), then reshape.
    - The resulting array is made contiguous (copy) to avoid surprises.
    """
    b = b64_str_to_bytes(str(payload["b64"]))
    dtype = np.dtype(str(payload["dtype"]))
    shape = tuple(int(x) for x in payload["shape"])

    arr = np.frombuffer(b, dtype=dtype)
    arr = arr.reshape(shape)

    # Make it a real owning array (optional but recommended)
    return np.array(arr, copy=True, order="C")
