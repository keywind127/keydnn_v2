"""
Tensor interface definitions.

This module defines the domain-level interface for tensor-like objects using
structural typing. The interface captures the minimal, backend-agnostic
properties required for tensors to participate in computation graphs,
modules, and optimization workflows.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Union, runtime_checkable

from .device._device import Device
from .device._device_protocol import DeviceLike


Number = Union[int, float]


@runtime_checkable
class ITensor(Protocol):
    """
    Domain-level tensor interface.

    An `ITensor` represents a multi-dimensional array that participates in
    numerical computation and, optionally, automatic differentiation.

    This interface defines the minimal contract required by the domain layer,
    allowing concrete tensor implementations to vary across backends (e.g.,
    NumPy, CUDA) without affecting higher-level logic.

    Notes
    -----
    - Structural typing (duck typing) is used instead of inheritance.
    - This interface is intentionally minimal and may be extended by
      infrastructure-level implementations.
    - If autograd support is required by domain-level logic, the autograd-facing
      hooks (e.g., `backward`, `grad`, `requires_grad`) should remain part of
      the protocol; otherwise consider splitting them into a separate protocol.
    """

    # ---------------------------------------------------------------------
    # Core identity / placement
    # ---------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the tensor.

        The shape is represented as a tuple of integers, where each element
        corresponds to the size of the tensor along a particular dimension.

        Returns
        -------
        tuple[int, ...]
            The tensor's shape.
        """
        ...

    @property
    def device(self) -> DeviceLike:
        """
        Return the device on which this tensor resides.

        The device indicates where the tensor's underlying data is stored
        and where computations involving this tensor should be executed.

        Returns
        -------
        Device
            The device associated with this tensor.
        """
        ...

    # ---------------------------------------------------------------------
    # Autograd flags and gradient storage
    # ---------------------------------------------------------------------
    @property
    def requires_grad(self) -> bool:
        """
        Indicate whether this tensor should accumulate gradients.

        Returns
        -------
        bool
            True if gradients should be tracked/accumulated, False otherwise.
        """
        ...

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient accumulation for this tensor.

        Parameters
        ----------
        value : bool
            If True, differentiable operations involving this tensor may attach a
            backward context and accumulate gradients into `grad`.
        """
        ...

    @property
    def grad(self) -> Optional["ITensor"]:
        """
        Return the gradient tensor associated with this tensor (if any).

        Returns
        -------
        Optional[ITensor]
            The stored gradient tensor, or None if not computed or cleared.
        """
        ...

    def zero_grad(self) -> None:
        """
        Clear the stored gradient.

        Notes
        -----
        Training loops typically call `zero_grad()` before backpropagation to avoid
        unintentional accumulation across iterations.
        """
        ...

    # ---------------------------------------------------------------------
    # Host interop / initialization utilities
    # ---------------------------------------------------------------------
    def to_numpy(self) -> Any:
        """
        Convert the tensor to a NumPy-compatible array (CPU-only in current backend).

        Returns
        -------
        Any
            A backend-native array object. For a NumPy backend, this is `np.ndarray`.

        Raises
        ------
        RuntimeError
            If conversion is unavailable for the tensor's device/backend.
        """
        ...

    def copy_from_numpy(self, arr: Any) -> None:
        """
        Copy data from a NumPy-compatible array into this tensor (CPU-only in current backend).

        The input array must match this tensor's shape.

        Parameters
        ----------
        arr : Any
            Source array. For a NumPy backend, this is expected to be `np.ndarray`.

        Raises
        ------
        RuntimeError
            If copying is unavailable for the tensor's device/backend.
        ValueError
            If the array shape does not match this tensor's shape.
        """
        ...

    def copy_from(self, other: "ITensor") -> None:
        """
        Copy data from another tensor into this tensor (in-place).

        Parameters
        ----------
        other : ITensor
            Source tensor to copy from.

        Raises
        ------
        ValueError
            If tensor shapes are incompatible.
        RuntimeError
            If copying is unavailable for the tensor's device/backend.
        """
        ...

    def fill(self, value: float) -> None:
        """
        Fill the tensor with a scalar value (backend/device dependent).

        Parameters
        ----------
        value : float
            Scalar value used to fill the tensor storage.

        Raises
        ------
        RuntimeError
            If fill is unavailable for the tensor's device/backend.
        """
        ...

    def debug_storage_repr(self) -> str:
        """
        Return a human-readable description of the underlying storage.

        Returns
        -------
        str
            A short string describing storage internals (shape/dtype when available).
        """
        ...

    # ---------------------------------------------------------------------
    # Shape and indexing helpers
    # ---------------------------------------------------------------------
    def numel(self) -> int:
        """
        Return the total number of elements in the tensor.

        Returns
        -------
        int
            Product of all dimensions in the tensor shape.
        """
        ...

    def reshape(self, new_shape: tuple[int, ...]) -> "ITensor":
        """
        Return a reshaped tensor.

        Parameters
        ----------
        new_shape : tuple[int, ...]
            Requested output shape.

        Returns
        -------
        ITensor
            A tensor with the requested shape.

        Notes
        -----
        Concrete implementations may or may not share storage with the original tensor.
        If autograd is enabled, gradients must be reshaped back to the original shape.
        """
        ...

    def __getitem__(self, key: Any) -> "ITensor":
        """
        Slice or index into the tensor.

        Parameters
        ----------
        key : Any
            Indexing specification (backend-specific; e.g., Python slice objects,
            integers, tuples, or array-like indices).

        Returns
        -------
        ITensor
            The indexed/sliced result tensor.

        Notes
        -----
        Concrete implementations may return a view or a copy.
        If autograd is enabled, gradients should be scattered back to the source shape.
        """
        ...

    @staticmethod
    def stack(tensors: Sequence["ITensor"], axis: int = 0) -> "ITensor":
        """
        Stack a sequence of tensors along a new axis.

        Parameters
        ----------
        tensors : Sequence[ITensor]
            Input tensors to stack. Must be non-empty and shape-compatible.
        axis : int, optional
            Axis at which the new dimension is inserted. Supports negative axes.

        Returns
        -------
        ITensor
            The stacked tensor.

        Notes
        -----
        Concrete implementations define device constraints and gradient routing rules.
        """
        ...

    # ---------------------------------------------------------------------
    # Reductions and elementwise transforms
    # ---------------------------------------------------------------------
    def sum(self) -> "ITensor":
        """
        Sum all elements of the tensor into a scalar.

        Returns
        -------
        ITensor
            A scalar tensor containing the sum of all elements.

        Notes
        -----
        If autograd is enabled, the backward rule is conceptually:
            d(sum(x))/dx = 1
        """
        ...

    def mean(self) -> "ITensor":
        """
        Compute the mean of all elements in the tensor.

        Returns
        -------
        ITensor
            A scalar tensor containing the mean value.

        Notes
        -----
        If autograd is enabled, the backward rule is conceptually:
            d(mean(x))/dx = 1 / numel(x)
        """
        ...

    def log(self) -> "ITensor":
        """
        Compute the elementwise natural logarithm.

        Returns
        -------
        ITensor
            A tensor of the same shape as `self` containing `log(self)` elementwise.

        Notes
        -----
        If autograd is enabled, the backward rule is conceptually:
            d(log(x))/dx = 1 / x
        """
        ...

    # ---------------------------------------------------------------------
    # Arithmetic and comparison dunder ops
    # ---------------------------------------------------------------------
    def __neg__(self) -> "ITensor":
        """
        Elementwise negation.

        Returns
        -------
        ITensor
            A tensor containing `-self` elementwise.
        """
        ...

    def __add__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise addition.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars may be lifted by the implementation.

        Returns
        -------
        ITensor
            Result of `self + other` elementwise.
        """
        ...

    def __radd__(self, other: Number) -> "ITensor":
        """
        Right-hand addition to support `scalar + tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Result of `other + self`.
        """
        ...

    def __sub__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise subtraction.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars may be lifted by the implementation.

        Returns
        -------
        ITensor
            Result of `self - other` elementwise.
        """
        ...

    def __rsub__(self, other: Number) -> "ITensor":
        """
        Right-hand subtraction to support `scalar - tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Result of `other - self`.
        """
        ...

    def __mul__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise multiplication.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars may be lifted by the implementation.

        Returns
        -------
        ITensor
            Result of `self * other` elementwise.
        """
        ...

    def __rmul__(self, other: Number) -> "ITensor":
        """
        Right-hand multiplication to support `scalar * tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Result of `other * self`.
        """
        ...

    def __truediv__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise true division.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars may be lifted by the implementation.

        Returns
        -------
        ITensor
            Result of `self / other` elementwise.
        """
        ...

    def __rtruediv__(self, other: Number) -> "ITensor":
        """
        Right-hand division to support `scalar / tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Result of `other / self`.
        """
        ...

    def __gt__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise greater-than comparison.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand.

        Returns
        -------
        ITensor
            A tensor encoding the elementwise comparison result.

        Notes
        -----
        Many implementations treat comparisons as non-differentiable and return
        a tensor with `requires_grad=False`.
        """
        ...

    # ---------------------------------------------------------------------
    # Autograd execution
    # ---------------------------------------------------------------------
    def backward(self, grad_out: Optional["ITensor"] = None) -> None:
        """
        Backpropagate gradients from this tensor through the autograd graph.

        Parameters
        ----------
        grad_out : Optional[ITensor], optional
            Gradient w.r.t. this tensor. If omitted, this tensor must be a scalar
            (shape == ()) and the gradient is assumed to be 1.

        Raises
        ------
        ValueError
            If `grad_out` is None for a non-scalar tensor, or shape constraints fail.
        RuntimeError
            If backpropagation is unavailable for the tensor's device/backend.
        """
        ...

    # ---------------------------------------------------------------------
    # Internal autograd hooks (used by the current engine)
    # ---------------------------------------------------------------------
    def _set_ctx(self, ctx: Optional[Any]) -> None:
        """
        Attach or detach the backward context.

        Parameters
        ----------
        ctx : Optional[Any]
            Backend-defined autograd context object. Use None to detach.

        Notes
        -----
        This is an internal hook intended for use by differentiable operations
        and the autograd engine.
        """
        ...

    def _get_ctx(self) -> Optional[Any]:
        """
        Return the backward context attached to this tensor, if any.

        Returns
        -------
        Optional[Any]
            The backend-defined autograd context object, or None if absent.

        Notes
        -----
        This is an internal hook intended for use by the autograd engine.
        """
        ...

    def _accumulate_grad_(self, g: "ITensor") -> None:
        """
        In-place accumulate gradient `g` into `self.grad`.

        Parameters
        ----------
        g : ITensor
            Gradient to accumulate.

        Notes
        -----
        This method is intended to be used by the autograd engine when writing
        final accumulated gradients into leaf tensors.
        """
        ...

    @staticmethod
    def _detach_no_grad(t: "ITensor") -> "ITensor":
        """
        Return a detached copy of `t` that does not track gradients.

        Parameters
        ----------
        t : ITensor
            Source tensor.

        Returns
        -------
        ITensor
            A tensor with the same value as `t`, but with no autograd history.
        """
        ...

    @staticmethod
    def _add_no_grad(a: "ITensor", b: "ITensor") -> "ITensor":
        """
        Add two tensors without creating autograd history.

        Parameters
        ----------
        a : ITensor
            Left operand.
        b : ITensor
            Right operand.

        Returns
        -------
        ITensor
            The elementwise sum, not connected to the autograd graph.
        """
        ...
