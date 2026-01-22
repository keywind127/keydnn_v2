from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np

from ....infrastructure.tensor import Tensor as _InfraTensor
from ....domain.device._device import Device


def Tensor(
    shape: Tuple[int, ...],
    device: Optional[Any] = None,
    *,
    requires_grad: bool = False,
    dtype: Any = np.float32,
    init_zeros: bool = True,
):
    """
    Presentation-layer Tensor constructor.

    Defaults to zero-initialized storage (`init_zeros=True`) because that is what
    most users expect from a high-level DL API.

    Notes
    -----
    Internally, KeyDNN may prefer `init_zeros=False` for performance-sensitive
    paths where the tensor will be immediately overwritten.
    """
    if device is None:
        device = Device("cpu")
    return _InfraTensor(
        shape=tuple(shape),
        device=device,
        requires_grad=requires_grad,
        dtype=np.dtype(dtype),
        init_zeros=bool(init_zeros),
    )


__all__ = ["Tensor"]
