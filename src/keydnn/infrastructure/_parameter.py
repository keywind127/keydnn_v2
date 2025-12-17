from __future__ import annotations

from typing import Optional

from ..domain._parameter import IParameter
from ._tensor import Tensor


class Parameter(Tensor, IParameter):
    """
    Trainable tensor wrapper.

    Parameter is a Tensor that is intended to be optimized (updated by an optimizer).
    It carries training-related metadata:
      - requires_grad: whether gradients should be accumulated
      - grad: the accumulated gradient tensor (or None)
    """

    def __init__(self, *args, requires_grad: bool = True, **kwargs) -> None:
        # Forward all construction args to Tensor (shape/device/data/etc.)
        super().__init__(*args, **kwargs)
        self._requires_grad: bool = bool(requires_grad)
        self._grad: Optional[Tensor] = None

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = bool(value)

    @property
    def grad(self) -> Optional[Tensor]:
        return self._grad

    def zero_grad(self) -> None:
        """Clear any accumulated gradient."""
        self._grad = None

    # ---- Optional helpers for autograd/optimizers (safe to keep minimal) ----
    def set_grad(self, grad: Optional[Tensor]) -> None:
        """Overwrite stored gradient (used by autograd)."""
        self._grad = grad

    def accumulate_grad(self, grad: Tensor) -> None:
        """
        Accumulate gradients (used by autograd).
        If your Tensor supports `+`, this will sum; otherwise replace.
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
