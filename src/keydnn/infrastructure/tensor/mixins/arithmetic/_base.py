"""
Arithmetic mixin defining elementwise Tensor operators.

This module declares :class:`TensorMixinArithmetic`, an abstract mixin that
specifies the public API and mathematical semantics for elementwise arithmetic
operations on tensors.

The mixin itself does not implement numerical kernels. Concrete CPU/CUDA
implementations are provided elsewhere and registered via a control-path
dispatch mechanism. This design keeps the Tensor core lightweight while
supporting device-specific execution strategies behind a unified interface.
"""

from typing import Optional, Union
from abc import ABC

from .....domain._tensor import ITensor

Number = Union[int, float]


class TensorMixinArithmetic(ABC):
    """
    Abstract mixin defining elementwise arithmetic operations for tensors.

    This mixin specifies the signatures, expected behavior, and backward
    semantics for arithmetic operators, including addition, subtraction,
    multiplication, and division.

    Notes
    -----
    - Methods defined here serve as interface declarations and documentation
      of mathematical contracts.
    - No computation is performed in this class.
    - Scalars are conceptually promoted to tensors matching the receiver's
      shape and device before applying the operation.
    - Backward rules described in method docstrings are contractual and must
      be respected by all concrete implementations.
    """

    # ----------------------------
    # True division
    # ----------------------------
    def __truediv__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise true division.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self / other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a / b) / da = 1 / b``
        - ``d(a / b) / db = -a / (b^2)``
        """
        ...

    def __rtruediv__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand true division to support ``scalar / Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other / self``.

        Notes
        -----
        This method promotes the scalar to a tensor compatible with ``self``
        and delegates to :meth:`__truediv__`.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__truediv__(self)

    # ----------------------------
    # Python 2 legacy division alias
    # ----------------------------
    def __div__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise division (legacy alias for true division).

        Notes
        -----
        Python 3 uses :meth:`__truediv__` to implement the ``/`` operator.
        This method exists solely for compatibility with legacy code that
        explicitly calls ``__div__``.
        """
        return self.__truediv__(other)

    def __rdiv__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand division (legacy alias for right true division).

        Notes
        -----
        Python 3 uses :meth:`__rtruediv__`. This method exists solely for
        compatibility with legacy code that explicitly calls ``__rdiv__``.
        """
        return self.__rtruediv__(other)

    # ----------------------------
    # Addition
    # ----------------------------
    def __add__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise addition.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self + other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a + b) / da = 1``
        - ``d(a + b) / db = 1``
        """
        ...

    def __radd__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand addition to support ``scalar + Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other + self``.

        Notes
        -----
        Addition is commutative, so this method simply delegates to
        :meth:`__add__`.
        """
        return self.__add__(other)

    def __sub__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise subtraction.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self - other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a - b) / da = 1``
        - ``d(a - b) / db = -1``
        """
        ...

    def __rsub__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand subtraction to support ``scalar - Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other - self``.

        Notes
        -----
        The scalar is promoted to a tensor compatible with ``self`` before
        applying subtraction.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__sub__(self)

    def __isub__(self: ITensor, other: ITensor) -> ITensor:
        # out-of-place compute
        out = self.__sub__(other)

        # IMPORTANT: write back into existing storage so model params mutate
        # This must be device-aware (CPU->CPU, CUDA->CUDA)
        self.copy_from(out)

        return self

    # ----------------------------
    # Multiplication
    # ----------------------------
    def __mul__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise multiplication.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self * other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a * b) / da = b``
        - ``d(a * b) / db = a``
        """
        ...

    def __rmul__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand multiplication to support ``scalar * Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other * self``.

        Notes
        -----
        Multiplication is commutative, so this method simply delegates to
        :meth:`__mul__`.
        """
        return self.__mul__(other)
