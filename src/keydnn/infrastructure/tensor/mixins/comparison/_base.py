"""
Comparison mixin defining elementwise Tensor comparison operators.

This module declares :class:`TensorMixinComparison`, an abstract mixin that
specifies the public API and semantics for elementwise comparison operations
on tensors, including greater-than, greater-than-or-equal, less-than, and
less-than-or-equal.

Comparison operations in this mixin are intentionally **non-differentiable**
in the current design: all comparison results have ``requires_grad=False`` and
do not participate in autograd. The outputs are numeric masks (float32) rather
than boolean tensors, which makes them convenient for use in subsequent
arithmetic expressions.
"""

from typing import Union
from abc import ABC

from .....domain._tensor import ITensor

Number = Union[int, float]
"""Scalar types accepted by Tensor comparison operators."""


class TensorMixinComparison(ABC):
    """
    Abstract mixin defining elementwise comparison operations for tensors.

    This mixin specifies method signatures, expected behavior, and output
    semantics for comparison operators. Concrete CPU/CUDA implementations are
    provided elsewhere and dispatched at runtime.

    Notes
    -----
    - All comparison operations return numeric masks (float32 tensors),
      where ``1.0`` represents ``True`` and ``0.0`` represents ``False``.
    - Comparison results never require gradients.
    - Broadcasting is not supported; operand shapes must match.
    """

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __gt__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise greater-than comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self > other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.
        - The exact backend implementation (CPU/CUDA) is selected at runtime.
        """
        ...

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __ge__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise greater-than-or-equal comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self >= other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.

        Implementation
        --------------
        Implemented in terms of greater-than and negation:

            ``a >= b  ⇔  not (b > a)``

        The boolean negation is represented numerically as ``1 - (b > a)``.
        """
        other_t = self._as_tensor_like(other, self)

        # Ensure same shape (and implicitly same device support path via __gt__)
        self._binary_op_shape_check(self, other_t)

        # ge = 1 - (other > self)
        gt = other_t.__gt__(self)  # float32 mask
        one = self._as_tensor_like(1.0, gt)
        out = one - gt

        out.requires_grad = False
        return out

    def __lt__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise less-than comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self < other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.

        Implementation
        --------------
        Implemented directly using greater-than:

            ``a < b  ⇔  b > a``
        """
        other_t = self._as_tensor_like(other, self)
        self._binary_op_shape_check(self, other_t)

        out = other_t.__gt__(self)
        out.requires_grad = False
        return out

    def __le__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise less-than-or-equal comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self <= other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.

        Implementation
        --------------
        Implemented in terms of greater-than and negation:

            ``a <= b  ⇔  not (a > b)``

        The boolean negation is represented numerically as ``1 - (a > b)``.
        """
        other_t = self._as_tensor_like(other, self)
        self._binary_op_shape_check(self, other_t)

        # le = 1 - (self > other)
        gt = self.__gt__(other_t)  # float32 mask
        one = self._as_tensor_like(1.0, gt)
        out = one - gt

        out.requires_grad = False
        return out
