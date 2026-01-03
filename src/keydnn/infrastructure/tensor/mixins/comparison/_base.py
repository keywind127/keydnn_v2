"""
Comparison mixin defining elementwise Tensor comparison operators.

This module declares :class:`TensorMixinComparison`, an abstract mixin that
specifies the public API and semantics for elementwise comparison operations
on tensors, including equality, inequality, and ordering comparisons.

Comparison operations in this mixin are intentionally **non-differentiable**
in the current design: all comparison results have ``requires_grad=False`` and
do not participate in autograd. The outputs are numeric masks (float32) rather
than boolean tensors, which makes them convenient for use in subsequent
arithmetic expressions.

Backend-specific implementations (CPU / CUDA) are registered separately and
selected at runtime via control-path dispatch.
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
    semantics for comparison operators. Concrete CPU and CUDA implementations
    are provided elsewhere and dispatched at runtime.

    General semantics
    -----------------
    - All comparison operations return numeric mask tensors of dtype float32:
      ``1.0`` represents ``True`` and ``0.0`` represents ``False``.
    - Comparison results never require gradients.
    - Broadcasting is not supported; operand shapes must match.
    - Scalar operands are supported:
        * CPU backends may promote scalars to tensor-like operands.
        * CUDA backends may dispatch to dedicated scalar kernels without
          materializing temporary tensors.
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
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self > other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.
        - Backend-specific implementations (CPU/CUDA) are selected at runtime.
        """
        ...

    def __ge__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise greater-than-or-equal comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self >= other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.
        - No broadcasting is performed.
        """
        ...

    def __lt__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise less-than comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self < other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.
        - No broadcasting is performed.
        """
        ...

    def __le__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise less-than-or-equal comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self <= other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd; the returned tensor
          always has ``requires_grad=False``.
        - No broadcasting is performed.
        """
        ...

    def __eq__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise equality comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self == other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd.
        - No broadcasting is performed.
        """
        ...

    def __ne__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Compute the elementwise not-equal comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. May be a tensor with matching shape and device,
            or a scalar value.

        Returns
        -------
        ITensor
            A float32 tensor with value ``1.0`` where ``self != other``,
            and ``0.0`` elsewhere.

        Notes
        -----
        - This operation does not participate in autograd.
        - No broadcasting is performed.
        """
        ...
