def require_tensorflow():
    try:
        import tensorflow as tf

        return tf
    except ImportError as e:
        raise ImportError(
            "Keras interop requires TensorFlow. Install with: pip install keydnn[keras]"
        ) from e
