from .....infrastructure.native_cuda.python._native_loader import (
    load_keydnn_cuda_native,
)


def cuda_available() -> bool:
    try:

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


__all__ = [
    "cuda_available",
]
