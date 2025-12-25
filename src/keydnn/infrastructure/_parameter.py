"""
Concrete trainable parameter implementation.

This module defines `Parameter`, an infrastructure-level implementation of the
domain contract `IParameter`. A `Parameter` is a `Tensor` intended to be
optimized by training algorithms (e.g., SGD, Adam). It carries training-related
state such as `requires_grad` and an accumulated gradient buffer (`grad`).

Design notes
------------
- `Parameter` subclasses `Tensor` to reuse storage, shape/device behavior, and
  any tensor operations.
- The gradient buffer is stored separately from tensor data and is populated
  by the autograd engine.
- The `requires_grad` flag enables freezing/unfreezing parameters without
  changing module structure.
"""

from __future__ import annotations

from typing import Optional

from ..domain._parameter import IParameter
from .tensor._tensor import Tensor


class Parameter(Tensor, IParameter):
    """
    Trainable tensor wrapper.

    A `Parameter` is a `Tensor` that participates in optimization. It extends
    `Tensor` with training-related semantics:

    - `requires_grad`: controls whether gradients should be accumulated
    - `grad`: stores the accumulated gradient tensor (or None)
    - gradient utility methods used by autograd and optimizers

    Parameters
    ----------
    *args
        Positional arguments forwarded to the `Tensor` constructor.
    requires_grad : bool, optional
        Whether this parameter should accumulate gradients. Defaults to True.
    **kwargs
        Keyword arguments forwarded to the `Tensor` constructor.

    Notes
    -----
    - While `Tensor` may also expose `requires_grad`/`grad`, `Parameter` exists
      to make trainable state explicit and to serve as the primary object type
      returned by `Module.parameters()`.
    - This class is intentionally minimal and can be extended later with
      optimizer state hooks if needed.
    """

    def __init__(self, *args, requires_grad: bool = True, **kwargs) -> None:
        """
        Initialize a trainable parameter.

        This constructor forwards all tensor construction arguments to `Tensor`
        and then configures parameter-specific training state.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the `Tensor` constructor.
        requires_grad : bool, optional
            Whether this parameter should accumulate gradients.
        **kwargs
            Keyword arguments forwarded to the `Tensor` constructor.
        """
        # Forward all construction args to Tensor (shape/device/data/etc.)
        super().__init__(*args, **kwargs)
        self._requires_grad: bool = bool(requires_grad)
        self._grad: Optional[Tensor] = None

    @property
    def requires_grad(self) -> bool:
        """
        Indicate whether this parameter should accumulate gradients.

        Returns
        -------
        bool
            True if gradients should be accumulated, False if frozen.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient accumulation.

        Parameters
        ----------
        value : bool
            If True, gradients may be accumulated into `grad`.
            If False, incoming gradients are ignored.
        """
        self._requires_grad = bool(value)

    @property
    def grad(self) -> Optional[Tensor]:
        """
        Return the accumulated gradient for this parameter.

        Returns
        -------
        Optional[Tensor]
            The gradient tensor if present, otherwise None.
        """
        return self._grad

    def zero_grad(self) -> None:
        """
        Clear any accumulated gradient.

        Notes
        -----
        Training loops typically call this method before computing new
        gradients to prevent unintentional accumulation across steps.
        """
        self._grad = None

    # ---- Optional helpers for autograd/optimizers (safe to keep minimal) ----
    def set_grad(self, grad: Optional[Tensor]) -> None:
        """
        Overwrite the stored gradient (used by autograd).

        Parameters
        ----------
        grad : Optional[Tensor]
            The new gradient tensor to store, or None to clear.
        """
        self._grad = grad

    def accumulate_grad(self, grad: Tensor) -> None:
        """
        Accumulate an incoming gradient into this parameter.

        This method is intended for use by the autograd engine when multiple
        paths in the computation graph contribute gradients to the same
        parameter.

        Parameters
        ----------
        grad : Tensor
            Incoming gradient contribution to accumulate.

        Notes
        -----
        - If `requires_grad` is False, the gradient is ignored.
        - If no gradient is currently stored, the incoming gradient is stored
          directly.
        - Otherwise, the method attempts to add (`+`) to accumulate. If tensor
          addition is unavailable, it falls back to overwriting.
        """
        if not self._requires_grad:
            return

        if self._grad is None:
            self._grad = grad
        else:
            try:
                self._grad = self._grad + grad  # relies on Tensor.__add__
            except Exception:
                # Fallback: overwrite if addition isn't supported yet
                self._grad = grad
