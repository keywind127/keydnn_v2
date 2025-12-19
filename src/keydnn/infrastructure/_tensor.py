"""
Concrete Tensor implementation (NumPy backend) and autograd context.

This module provides a concrete `Tensor` implementation that satisfies the
domain-level `ITensor` protocol and stores data on a specified `Device`.
Currently, CPU tensors are backed by NumPy arrays, while CUDA tensors are
represented by a placeholder string (until a CUDA backend is implemented).

It also defines `Context`, a lightweight record attached to tensors produced
by differentiable operations. The context stores:
- parent tensors (inputs to the operation),
- a backward function that maps `grad_out` to gradients for each parent, and
- any saved tensors or metadata required to compute those gradients.

Design notes
------------
- This file sits in the infrastructure layer: it imports NumPy and concrete
  error types, and provides a concrete runtime implementation.
- Automatic differentiation is expressed by attaching an optional `Context`
  to output tensors. The autograd engine (elsewhere) can traverse `Context`
  links backward to propagate gradients.
- Broadcasting rules are intentionally not implemented yet; binary ops require
  exact shape matches.
"""

from __future__ import annotations

from typing import Any, Union, Callable, Optional, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..domain._tensor import ITensor
from ..domain._device import Device
from ..domain._errors import DeviceNotSupportedError

Number = Union[int, float]


@dataclass
class Context:
    """
    Backward context attached to a Tensor produced by an operation.

    A `Context` records the information required to compute gradients for an
    operation during backpropagation.

    Attributes
    ----------
    parents : Sequence[Tensor]
        The input tensors (and/or parameters) used to compute the output tensor.
        Gradients will be produced for these parents during the backward pass.
    backward_fn : Callable[[Tensor], Sequence[Optional[Tensor]]]
        A function that takes the gradient w.r.t. the output (`grad_out`) and
        returns gradients w.r.t. each `parents` entry, in the same order.
        Entries may be None for parents that do not require gradients.
    saved_tensors : list[Tensor]
        Tensors explicitly saved during the forward pass for use in backward.
        These are distinct from `parents`: they may include transformed values
        (e.g., outputs, masks, indices, cached intermediates).
    saved_meta : dict[str, Any]
        Non-tensor metadata required for backward (e.g., shapes, axes, indices).

    Notes
    -----
    `saved_tensors` and `saved_meta` are intentionally generic to support a wide
    range of operations without coupling the Context type to specific kernels.
    """

    parents: Sequence["Tensor"]
    backward_fn: Callable[["Tensor"], Sequence[Optional["Tensor"]]]
    saved_tensors: list["Tensor"] = field(default_factory=list)
    saved_meta: dict[str, Any] = field(default_factory=dict)

    def save_for_backward(self, *tensors: "Tensor") -> None:
        """
        Save tensors for use during the backward computation.

        Parameters
        ----------
        *tensors : Tensor
            Any number of tensors to be stored in `saved_tensors`.

        Notes
        -----
        Saved tensors are often intermediate results that are cheaper to reuse
        than to recompute during the backward pass (e.g., activation masks).
        """
        self.saved_tensors.extend(tensors)


class Tensor(ITensor):
    """
    Concrete tensor implementation (NumPy CPU backend, CUDA placeholder).

    This class provides a concrete runtime tensor that:
    - satisfies the domain-level `ITensor` protocol,
    - holds a shape and device placement,
    - stores underlying data (NumPy ndarray on CPU),
    - optionally participates in autograd via an attached `Context`.

    Parameters
    ----------
    shape : tuple[int, ...]
        Tensor shape.
    device : Device
        Target device placement for the tensor.
    requires_grad : bool, optional
        Whether this tensor should accumulate gradients during backprop.
        Defaults to False.
    ctx : Optional[Context], optional
        Backward context for autograd graph traversal. Typically set internally
        by differentiable operations. Defaults to None.

    Notes
    -----
    - For CPU tensors, `_data` is a NumPy ndarray of dtype float32.
    - For CUDA tensors, `_data` is currently a placeholder string.
    - Gradients (if any) are stored as another `Tensor` in `_grad`.
    """

    def __initialize_data(self) -> None:
        """
        Allocate and initialize underlying storage for the tensor.

        For CPU devices, storage is a NumPy ndarray initialized to zeros.
        For CUDA devices, a placeholder representation is currently used.

        Raises
        ------
        ValueError
            If the device type is unsupported.
        """
        match self._device:
            case Device() if self._device.is_cpu():
                self._data = np.zeros(self._shape, dtype=np.float32)
            case Device() if self._device.is_cuda():
                self._data = f"CUDA Tensor on device {self._device.index} with shape {self._shape}"
            case _:
                raise ValueError("Unsupported device type")

    def __init__(
        self,
        shape: tuple[int, ...],
        device: Device,
        *,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
    ) -> None:
        """
        Construct a new Tensor with allocated storage.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the tensor.
        device : Device
            Device placement.
        requires_grad : bool, optional
            Whether this tensor should participate in gradient accumulation.
        ctx : Optional[Context], optional
            Optional autograd context to attach to this tensor.
        """
        self._shape = shape
        self._device = device
        self.__initialize_data()

        # --- autograd fields (optional) ---
        self._requires_grad: bool = bool(requires_grad)
        self._grad: Optional["Tensor"] = None
        self._ctx: Optional[Context] = ctx

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the tensor shape.

        Returns
        -------
        tuple[int, ...]
            The tensor's shape.
        """
        return self._shape

    @property
    def device(self) -> Device:
        """
        Return the device on which this tensor resides.

        Returns
        -------
        Device
            The tensor's device placement descriptor.
        """
        return self._device

    @property
    def requires_grad(self) -> bool:
        """
        Indicate whether this tensor should accumulate gradients.

        Returns
        -------
        bool
            True if gradients should be tracked/accumulated, False otherwise.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient accumulation for this tensor.

        Parameters
        ----------
        value : bool
            If True, autograd operations involving this tensor may attach a
            backward context and accumulate gradients into `grad`.
        """
        self._requires_grad = bool(value)

    @property
    def grad(self) -> Optional["Tensor"]:
        """
        Return the gradient tensor associated with this tensor (if any).

        Returns
        -------
        Optional[Tensor]
            The stored gradient tensor, or None if not computed or cleared.
        """
        return self._grad

    def zero_grad(self) -> None:
        """
        Clear the stored gradient.

        Notes
        -----
        Training loops typically call `zero_grad()` before backprop to avoid
        unintentional accumulation across iterations.
        """
        self._grad = None

    def _set_ctx(self, ctx: Optional[Context]) -> None:
        """
        Attach or detach the backward context.

        Parameters
        ----------
        ctx : Optional[Context]
            Context to attach. Use None to detach.

        Notes
        -----
        This is an internal hook intended for use by differentiable operations
        and the autograd engine.
        """
        self._ctx = ctx

    def _get_ctx(self) -> Optional[Context]:
        """
        Return the backward context attached to this tensor, if any.

        Returns
        -------
        Optional[Context]
            The attached context, or None if this tensor is a leaf or has no
            autograd history.

        Notes
        -----
        This is an internal hook intended for use by the autograd engine.
        """
        return self._ctx

    def to_numpy(self) -> np.ndarray:
        """
        Convert the tensor to a NumPy ndarray (CPU-only).

        Returns
        -------
        np.ndarray
            A NumPy view/copy of the underlying CPU storage.

        Raises
        ------
        RuntimeError
            If called on a non-CPU tensor.
        """
        if not self._device.is_cpu():
            raise RuntimeError("to_numpy() is only available for CPU tensors.")
        return self._data

    def fill(self, value: float) -> None:
        """
        Fill the tensor with a scalar value (CPU-only).

        Parameters
        ----------
        value : float
            Scalar value used to fill the underlying array.

        Raises
        ------
        RuntimeError
            If called on a non-CPU tensor.
        """
        if not self._device.is_cpu():
            raise RuntimeError("fill() is only available for CPU tensors.")
        self._data.fill(value)

    def debug_storage_repr(self) -> str:
        """
        Return a human-readable description of the underlying storage.

        Returns
        -------
        str
            A short string describing storage (shape/dtype for CPU, placeholder
            string for CUDA).
        """
        if self._device.is_cpu():
            arr: np.ndarray = self._data
            return f"CPU ndarray shape={arr.shape} dtype={arr.dtype}"
        return str(self._data)

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        """
        Copy data from a NumPy array into this tensor (CPU-only).

        The input array is converted to float32 and must match the tensor shape.

        Parameters
        ----------
        arr : np.ndarray
            Source array.

        Raises
        ------
        RuntimeError
            If called on a non-CPU tensor.
        ValueError
            If the array shape does not match this tensor's shape.
        """
        if not self._device.is_cpu():
            raise RuntimeError("copy_from_numpy() is only available for CPU tensors.")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != self._shape:
            raise ValueError(
                f"Shape mismatch: tensor {self._shape} vs array {arr.shape}"
            )
        self._data[...] = arr

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _raise_device_not_supported(self, op: str) -> "None":
        """
        Raise a standardized 'device not supported' error for an operation.

        Parameters
        ----------
        op : str
            Operation name (e.g., "add", "mul", "neg").

        Raises
        ------
        DeviceNotSupportedError
            Always raised, indicating the operation is unavailable for the
            current device.
        """
        raise DeviceNotSupportedError(op=op, device=str(self._device))

    @staticmethod
    def _result_requires_grad(*parents: "Tensor") -> bool:
        """
        Determine whether an operation result should require gradients.

        Parameters
        ----------
        *parents : Tensor
            Parent tensors participating in an operation.

        Returns
        -------
        bool
            True if any parent requires gradients, False otherwise.
        """
        return any(p.requires_grad for p in parents)

    @staticmethod
    def _as_tensor_like(x: Union["Tensor", Number], like: "Tensor") -> "Tensor":
        """
        Convert an operand into a Tensor compatible with a reference tensor.

        If `x` is already a Tensor, it is returned as-is. If `x` is a Python
        scalar, a new Tensor is created with the same shape and device as
        `like`, filled with the scalar value.

        Parameters
        ----------
        x : Union[Tensor, Number]
            Operand to convert.
        like : Tensor
            Reference tensor providing shape/device for scalar lifting.

        Returns
        -------
        Tensor
            A tensor operand compatible with `like`.

        Raises
        ------
        TypeError
            If `x` is neither a Tensor nor a supported scalar type.
        """
        if isinstance(x, Tensor):
            return x
        if isinstance(x, (int, float)):
            t = Tensor(shape=like.shape, device=like.device, requires_grad=False)
            t.fill(float(x))
            return t
        raise TypeError(f"Unsupported operand type: {type(x)!r}")

    @staticmethod
    def _binary_op_shape_check(a: "Tensor", b: "Tensor") -> None:
        """
        Validate shape compatibility for binary elementwise operations.

        Currently, the framework enforces strict shape equality and does not
        implement broadcasting.

        Parameters
        ----------
        a : Tensor
            Left operand.
        b : Tensor
            Right operand.

        Raises
        ------
        ValueError
            If shapes do not match exactly.
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # ----------------------------
    # Unary ops
    # ----------------------------
    def __neg__(self) -> "Tensor":
        """
        Elementwise negation.

        Returns
        -------
        Tensor
            A tensor containing `-self` (elementwise).

        Notes
        -----
        If `self.requires_grad` is True, the returned tensor will carry an
        attached `Context` that computes `d(-x)/dx = -1`.
        """
        if self._device.is_cpu():
            out = Tensor(
                shape=self.shape, device=self.device, requires_grad=self.requires_grad
            )
            out.copy_from_numpy(-self.to_numpy())

            if self.requires_grad:
                ctx = Context(
                    parents=(self,),
                    backward_fn=lambda grad_out: (-(grad_out),),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("neg")

    # ----------------------------
    # Addition / Subtraction
    # ----------------------------
    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise addition.

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            Result of `self + other` (elementwise).

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - d(a + b)/da = 1
        - d(a + b)/db = 1
        """
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() + other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out, grad_out),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("add")

    def __radd__(self, other: Number) -> "Tensor":
        """
        Right-hand addition to support `scalar + Tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        Tensor
            Result of `other + self`.
        """
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise subtraction.

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            Result of `self - other` (elementwise).

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - d(a - b)/da = 1
        - d(a - b)/db = -1
        """
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() - other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out, -grad_out),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("sub")

    def __rsub__(self, other: Number) -> "Tensor":
        """
        Right-hand subtraction to support `scalar - Tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        Tensor
            Result of `other - self`.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__sub__(self)

    # ----------------------------
    # Multiplication
    # ----------------------------
    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise multiplication.

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            Result of `self * other` (elementwise).

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - d(a * b)/da = b
        - d(a * b)/db = a
        """
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() * other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out * other_t, grad_out * self),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("mul")

    def __rmul__(self, other: Number) -> "Tensor":
        """
        Right-hand multiplication to support `scalar * Tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        Tensor
            Result of `other * self`.
        """
        return self.__mul__(other)

    # ----------------------------
    # True division
    # ----------------------------
    def __truediv__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise true division.

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            Result of `self / other` (elementwise).

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - d(a / b)/da = 1 / b
        - d(a / b)/db = -a / (b^2)
        """
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() / other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (
                        grad_out / other_t,
                        -(grad_out * self) / (other_t * other_t),
                    ),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("truediv")

    def __rtruediv__(self, other: Number) -> "Tensor":
        """
        Right-hand division to support `scalar / Tensor`.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        Tensor
            Result of `other / self`.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__truediv__(self)

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __gt__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise greater-than comparison (no gradients).

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            A float32 tensor with 1.0 where `self > other`, else 0.0.

        Notes
        -----
        Comparison operations do not participate in autograd in this minimal
        implementation (the result always has `requires_grad=False`).
        """
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            out = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            out.copy_from_numpy(
                (self.to_numpy() > other_t.to_numpy()).astype(np.float32)
            )
            return out

        self._raise_device_not_supported("gt")
