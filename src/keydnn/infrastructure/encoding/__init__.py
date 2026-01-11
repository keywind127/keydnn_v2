"""
Base64 serialization utilities.

This module exposes public helpers for encoding and decoding binary data and
NumPy arrays using Base64-compatible payloads. These utilities are primarily
used for model serialization, checkpointing, and in-memory state transfer
where raw bytes or ndarray contents must be represented in JSON-safe form.

Exports
-------
- `bytes_to_b64_str`: Encode raw bytes into a Base64 string.
- `b64_str_to_bytes`: Decode a Base64 string back into raw bytes.
- `ndarray_to_payload`: Convert a NumPy array into a serializable payload.
- `payload_to_ndarray`: Reconstruct a NumPy array from a serialized payload.

Design intent
-------------
- Provide JSON-safe, deterministic serialization primitives.
- Avoid coupling serialization logic to model or layer implementations.
- Serve as a stable public surface over internal Base64 helpers.
"""

from ._b64 import (
    bytes_to_b64_str,
    b64_str_to_bytes,
    ndarray_to_payload,
    payload_to_ndarray,
)

__all__ = [
    bytes_to_b64_str.__name__,
    b64_str_to_bytes.__name__,
    ndarray_to_payload.__name__,
    payload_to_ndarray.__name__,
]
