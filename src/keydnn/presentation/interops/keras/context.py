# src/keydnn/presentation/interops/keras/context.py
"""
Conversion context for Keras interoperability.

This module defines a lightweight context object passed through the conversion
pipeline (importer -> registry -> converters). It centralizes shared conversion
settings such as device placement and dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class KerasImportContext:
    """
    Context passed to Keras converters during import.

    Parameters
    ----------
    device : Any
        KeyDNN device object (typically `Device("cpu")` or `Device("cuda:0")`).
    dtype : Any, optional
        Target dtype to use for KeyDNN parameter initialization and loading.
        Defaults to `np.float32`.
    strict : bool, optional
        If True, unsupported layers raise an error. If False, unsupported layers
        are returned as placeholders by the importer (future extension).
    allow_non_linear_activation : bool, optional
        If True, converters may accept non-linear activations on certain Keras
        layers. In Phase 1, it is recommended to keep activations explicit as
        separate KeyDNN modules.
    """

    device: Any
    dtype: Any = np.float32
    strict: bool = True
    allow_non_linear_activation: bool = False

    def with_overrides(
        self,
        *,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        strict: Optional[bool] = None,
        allow_non_linear_activation: Optional[bool] = None,
    ) -> "KerasImportContext":
        """
        Return a new context with selected fields overridden.
        """
        return KerasImportContext(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
            strict=self.strict if strict is None else strict,
            allow_non_linear_activation=(
                self.allow_non_linear_activation
                if allow_non_linear_activation is None
                else allow_non_linear_activation
            ),
        )
