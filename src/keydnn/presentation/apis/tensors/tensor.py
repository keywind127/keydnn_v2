"""
Presentation-layer Tensor abstraction for KeyDNN.

This module defines a user-facing `Tensor` class that subclasses the
infrastructure-level `Tensor` while preserving its full API surface,
including all classmethods and staticmethods (e.g., `zeros`, `ones`, `rand`).

Design goals
------------
- Provide ergonomic defaults expected by end users:
  - Default device is CPU when not explicitly specified.
  - Default initialization uses zero-filled storage.
- Preserve full compatibility with the infrastructure tensor:
  - No functionality is removed or hidden.
  - Autograd, device handling, and memory semantics remain unchanged.
- Avoid factory-function pitfalls:
  - Subclassing ensures classmethods and staticmethods continue to work
    naturally (e.g., `Tensor.rand(...)`).

Architecture notes
------------------
- This class is part of the *presentation layer* and is intended to be the
  primary tensor entry point for user code.
- Performance-sensitive internal paths may bypass this layer and construct
  infrastructure tensors directly with more explicit control (e.g.,
  `init_zeros=False`).
- Device normalization is handled eagerly at construction time for clarity
  and predictability.

Example
-------
>>> from keydnn.tensors import Tensor, Device
>>> x = Tensor((1024, 1024), device=Device("cuda:0"), requires_grad=True)
>>> y = (x @ x.T).mean()
>>> y.backward()
>>> print(y.item())
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

from ....infrastructure.tensor import Tensor as _InfraTensor
from ....domain.device._device import Device


class Tensor(_InfraTensor):
    """
    Presentation-layer Tensor.

    This subclass preserves the full infrastructure Tensor API (including
    classmethods/staticmethods like `zeros`, `ones`, `rand`, etc.) while
    applying presentation-friendly defaults for direct construction.

    Notes
    -----
    - Direct construction defaults to CPU if `device` is omitted.
    - Direct construction defaults to zero-initialized storage because that is
      what most users expect from a high-level DL API.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        device: Optional[Any] = None,
        *,
        requires_grad: bool = False,
        dtype: Any = np.float32,
        init_zeros: bool = True,
        ctx=None,
    ) -> None:
        """
        Construct a presentation-layer Tensor with user-friendly defaults.

        This initializer mirrors the infrastructure-level `Tensor` constructor
        while providing ergonomics expected from a high-level deep learning API.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the tensor to be created.
        device : Device | str | None, optional
            Target device for the tensor. If None, defaults to CPU.
            If a string is provided (e.g., ``"cpu"``, ``"cuda:0"``), it must be
            convertible to a `Device` by the infrastructure layer.
        requires_grad : bool, default=False
            Whether this tensor should participate in autograd and accumulate
            gradients during backpropagation.
        dtype : Any, default=np.float32
            Data type of the tensor elements.
        init_zeros : bool, default=True
            Whether to initialize storage with zeros.
            High-level user code typically expects zero-initialized tensors,
            whereas performance-sensitive internal paths may disable this.
        ctx : Optional[Any], optional
            Autograd context. This is primarily intended for internal use by
            the autograd engine and should usually be left as None by users.

        Notes
        -----
        - This constructor does **not** hide or restrict the infrastructure API.
          All classmethods and staticmethods (e.g., ``Tensor.zeros``,
          ``Tensor.ones``, ``Tensor.rand``) remain available.
        - For performance-critical internal code paths, prefer calling the
          infrastructure Tensor directly with ``init_zeros=False``.
        """
        if device is None:
            device = Device("cpu")
        super().__init__(
            shape=tuple(shape),
            device=device,
            requires_grad=requires_grad,
            dtype=np.dtype(dtype),
            init_zeros=bool(init_zeros),
            ctx=ctx,
        )


__all__ = ["Tensor"]
