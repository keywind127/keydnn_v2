"""
Keras layer converters for KeyDNN.

This package contains converter implementations that translate individual
Keras layers into their KeyDNN equivalents. Each converter is responsible for
constructing the corresponding KeyDNN module and loading parameters from the
Keras layer.

Converters are typically discovered and used by a higher-level importer or
registry during Keras model conversion.
"""

from .dense import DenseConverter

__all__ = [
    "DenseConverter",
]
