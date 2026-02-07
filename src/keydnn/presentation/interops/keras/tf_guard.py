"""
TensorFlow import guard for Keras interoperability.

This module provides a small helper to lazily import TensorFlow only when
Keras interoperability features are used. It avoids introducing TensorFlow
as a hard dependency for the core KeyDNN package.
"""


def require_tensorflow() -> None:
    """
    Lazily import and return the TensorFlow module.

    Returns
    -------
    module
        The imported TensorFlow module.

    Raises
    ------
    ImportError
        If TensorFlow is not installed. The error message instructs how to
        install the optional Keras interoperability dependencies.
    """
    try:
        import tensorflow as tf

        return tf
    except ImportError as e:
        raise ImportError(
            "Keras interop requires TensorFlow. Install with: pip install keydnn[keras]"
        ) from e
