"""
Keras interoperability entrypoints for KeyDNN.

This package provides functionality for importing Keras models into KeyDNN
using a converter-based architecture. Most users should only need the
`from_keras` function.
"""

from .importer import from_keras
from .context import KerasImportContext

__all__ = [
    "from_keras",
    "KerasImportContext",
]
