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
from ..domain.device._device import Device
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
        # match self._device:
        #     case Device() if self._device.is_cpu():
        #         self._data = np.zeros(self._shape, dtype=np.float32)
        #     case Device() if self._device.is_cuda():
        #         self._data = f"CUDA Tensor on device {self._device.index} with shape {self._shape}"
        #     case _:
        #         raise ValueError("Unsupported device type")
        d = self._device

        # Duck typing: avoid class-identity issues.
        is_cpu = getattr(d, "is_cpu", None)
        if callable(is_cpu) and is_cpu():
            self._data = np.zeros(self._shape, dtype=np.float32)
            return

        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            self._data = f"CUDA Tensor on device {getattr(d, 'index', None)} with shape {self._shape}"
            return

        raise ValueError(f"Unsupported device type: {type(d)!r} value={d!r}")

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

    def numel(self) -> int:
        """
        Return the total number of elements in the tensor.

        Returns
        -------
        int
            Product of all dimensions in the tensor shape.
        """
        n = 1
        for d in self._shape:
            n *= d
        return n

    def sum(self) -> "Tensor":
        """
        Sum all elements of the tensor into a scalar.

        Returns
        -------
        Tensor
            A scalar tensor containing the sum of all elements.

        Notes
        -----
        Backward rule:
            d(sum(x))/dx = 1
        """
        if not self._device.is_cpu():
            self._raise_device_not_supported("sum")

        # Compute forward value
        value = float(np.sum(self._data))
        out = Tensor(shape=(), device=self.device, requires_grad=self.requires_grad)
        out.copy_from_numpy(np.array(value, dtype=np.float32))

        if self.requires_grad:

            def backward_fn(grad_out: "Tensor"):
                # grad_out is scalar; expand to input shape
                grad = Tensor(
                    shape=self.shape,
                    device=self.device,
                    requires_grad=False,
                )
                grad.copy_from_numpy(
                    np.ones(self.shape, dtype=np.float32)
                    * float(np.asarray(grad_out.to_numpy()))
                )
                return (grad,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    def mean(self) -> "Tensor":
        """
        Compute the mean of all elements in the tensor.

        Returns
        -------
        Tensor
            A scalar tensor containing the mean value.

        Notes
        -----
        Backward rule:
            d(mean(x))/dx = 1 / numel(x)
        """
        if not self._device.is_cpu():
            self._raise_device_not_supported("mean")

        n = self.numel()
        value = float(np.sum(self._data) / n)
        out = Tensor(shape=(), device=self.device, requires_grad=self.requires_grad)
        out.copy_from_numpy(np.array(value, dtype=np.float32))

        if self.requires_grad:

            def backward_fn(grad_out: "Tensor"):
                grad = Tensor(
                    shape=self.shape,
                    device=self.device,
                    requires_grad=False,
                )
                grad.copy_from_numpy(
                    np.ones(self.shape, dtype=np.float32)
                    * (float(np.asarray(grad_out.to_numpy())) / n)
                )
                return (grad,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    def log(self) -> "Tensor":
        """
        Compute the elementwise natural logarithm of the tensor.

        Returns
        -------
        Tensor
            A tensor of the same shape as `self`, where each element is
            replaced by its natural logarithm.

        Notes
        -----
        - This operation is defined only for CPU tensors in the current
        implementation.
        - No broadcasting is performed; the output tensor has the same
        shape as the input.
        - If `self.requires_grad` is True, the returned tensor participates
        in autograd with the backward rule:

            d(log(x)) / dx = 1 / x

        - The backward pass propagates gradients elementwise using this rule
        and relies on strict shape matching.
        - The behavior for non-positive input values is undefined and will
        follow NumPy's semantics (e.g., `-inf` or `nan`).
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("log")

        out = Tensor(
            shape=self.shape,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        out.copy_from_numpy(np.log(self.to_numpy()))

        if self.requires_grad:
            ctx = Context(
                parents=(self,),
                backward_fn=lambda grad_out: (grad_out / self,),
            )
            out._set_ctx(ctx)

        return out

    def copy_from(self, other: "Tensor") -> None:
        """
        Copy data from another tensor into this tensor (in-place).

        Notes
        -----
        This method encapsulates backend-specific copying (CPU->CPU, CUDA->CUDA).
        """
        if self.device != other.device:
            self._raise_device_not_supported("copy_from_cross_device")  # for now
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        if self.device.is_cpu():
            self._data[...] = other.to_numpy()
            return

        self._raise_device_not_supported("copy_from")

    # ----------------------------
    # Autograd engine (graph traversal)
    # ----------------------------
    def backward(self, grad_out: Optional["Tensor"] = None) -> None:
        """
        Backpropagate gradients from this tensor through the autograd graph.

        Parameters
        ----------
        grad_out : Optional[Tensor], optional
            Gradient w.r.t. this tensor. If omitted, this tensor must be a scalar
            (shape == ()) and the gradient is assumed to be 1.0.

        Raises
        ------
        ValueError
            If grad_out is None and this tensor is not a scalar.
        RuntimeError
            If backward is invoked on a non-CPU tensor in the current implementation.

        Notes
        -----
        - Gradients are accumulated into `.grad` of leaf tensors that have
          `requires_grad=True`.
        - This implementation performs a reverse topological traversal.
        - Only CPU tensors are supported for now.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("backward")

        # Seed gradient
        if grad_out is None:
            if self.shape != ():
                raise ValueError(
                    "grad_out must be provided for non-scalar tensors. "
                    f"Got shape={self.shape}."
                )
            grad_out = Tensor(shape=(), device=self.device, requires_grad=False)
            grad_out.copy_from_numpy(np.array(1.0, dtype=np.float32))
        else:
            if not isinstance(grad_out, Tensor):
                raise TypeError(f"grad_out must be a Tensor, got {type(grad_out)!r}")
            if grad_out.shape != self.shape:
                raise ValueError(
                    f"grad_out shape mismatch: expected {self.shape}, got {grad_out.shape}"
                )
            if grad_out.device != self.device:
                raise ValueError("grad_out must be on the same device as self")

        # Build reverse topological order of nodes reachable from `self`
        topo: list[Tensor] = []
        visited: set[int] = set()

        def dfs(t: "Tensor") -> None:
            tid = id(t)
            if tid in visited:
                return
            visited.add(tid)

            ctx = t._get_ctx()
            if ctx is not None:
                for p in ctx.parents:
                    dfs(p)

            topo.append(t)

        dfs(self)

        # Map from tensor id -> accumulated gradient tensor
        grads: dict[int, Tensor] = {id(self): grad_out}

        # Traverse in reverse topo order (from outputs back to leaves)
        for t in reversed(topo):
            ctx = t._get_ctx()
            if ctx is None:
                continue

            grad_t = grads.get(id(t))
            if grad_t is None:
                # No gradient flowing to this node; skip
                continue

            parent_grads = ctx.backward_fn(grad_t)
            if len(parent_grads) != len(ctx.parents):
                raise RuntimeError(
                    "backward_fn must return one grad per parent. "
                    f"Got {len(parent_grads)} grads for {len(ctx.parents)} parents."
                )

            for parent, g in zip(ctx.parents, parent_grads):
                if g is None:
                    continue
                if not isinstance(g, Tensor):
                    raise TypeError(
                        f"backward_fn must return Tensor or None, got {type(g)!r}"
                    )
                if g.device != parent.device:
                    raise ValueError("Gradient device must match parent device")
                if g.shape != parent.shape:
                    raise ValueError(
                        f"Gradient shape mismatch for parent: expected {parent.shape}, got {g.shape}"
                    )

                # Accumulate gradient in the grads dict
                pid = id(parent)
                if pid in grads:
                    grads[pid] = self._add_no_grad(grads[pid], g)
                else:
                    # Ensure stored grads do not track grad
                    grads[pid] = self._detach_no_grad(g)

        # Write accumulated grads into leaf tensors that require grad
        for tid, g in grads.items():
            # Find the actual Tensor object by scanning visited/topo
            # (topo contains all visited tensors)
            # This keeps it simple without weakrefs.
            # Note: O(n^2) worst-case, but graphs are small for now.
            for t in topo:
                if id(t) == tid:
                    if t.requires_grad:
                        t._accumulate_grad_(g)
                    break

    def _accumulate_grad_(self, g: "Tensor") -> None:
        """
        In-place accumulate gradient `g` into `self.grad` (CPU-only).
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("accumulate_grad")

        g0 = self._detach_no_grad(g)

        if self._grad is None:
            self._grad = g0
            return

        # In-place add into existing grad storage
        if not self._grad.device.is_cpu():
            self._raise_device_not_supported("accumulate_grad_non_cpu")

        if self._grad.shape != g0.shape:
            raise ValueError(f"Grad shape mismatch: {self._grad.shape} vs {g0.shape}")

        # Add underlying numpy arrays directly (avoid creating autograd edges)
        self._grad._data[...] = self._grad._data + g0._data  # type: ignore[attr-defined]

    @staticmethod
    def _detach_no_grad(t: "Tensor") -> "Tensor":
        """
        Return a detached copy of `t` that does not track gradients and has no ctx.
        """
        out = Tensor(shape=t.shape, device=t.device, requires_grad=False, ctx=None)
        if t.device.is_cpu():
            out.copy_from_numpy(t.to_numpy())
            return out
        t._raise_device_not_supported("detach_no_grad")

    @staticmethod
    def _add_no_grad(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Add two tensors without creating autograd history.
        """
        if a.device != b.device:
            raise ValueError("Device mismatch in _add_no_grad")
        if a.shape != b.shape:
            raise ValueError("Shape mismatch in _add_no_grad")

        out = Tensor(shape=a.shape, device=a.device, requires_grad=False, ctx=None)
        if a.device.is_cpu():
            out.copy_from_numpy(a.to_numpy() + b.to_numpy())
            return out
        a._raise_device_not_supported("add_no_grad")

    def __getitem__(self, key: Any) -> "Tensor":
        """
        Slice/index into a tensor (CPU-only), producing a new Tensor.

        Notes
        -----
        - This returns a *copy* (not a view) for simplicity.
        - Backward rule scatters grad_out back into the parent tensor shape.
        - Supports basic slicing and NumPy-style fancy indexing.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("getitem")

        # Forward: NumPy slice/index
        src = self.to_numpy()
        sliced = src[key]

        # Normalize scalar outputs to shape=()
        if np.isscalar(sliced) or getattr(sliced, "shape", None) == ():
            sliced_arr = np.array(sliced, dtype=np.float32)  # shape ()
            out_shape = ()
        else:
            sliced_arr = np.asarray(sliced, dtype=np.float32)
            out_shape = sliced_arr.shape

        req = self.requires_grad
        out = Tensor(shape=out_shape, device=self.device, requires_grad=req)
        out.copy_from_numpy(sliced_arr)

        if req:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g_out_np = grad_out.to_numpy()
                grad_parent_np = np.zeros(self.shape, dtype=np.float32)

                # Determine whether this key uses fancy/advanced indexing,
                # where `grad_parent_np[key] += ...` would NOT accumulate correctly.
                def _is_fancy(k: Any) -> bool:
                    # Bool mask, list, ndarray => fancy
                    if isinstance(k, (list, np.ndarray)):
                        return True
                    # Tuple: if any component is fancy => fancy
                    if isinstance(k, tuple):
                        return any(isinstance(kk, (list, np.ndarray)) for kk in k)
                    return False

                fancy = _is_fancy(key)

                if fancy:
                    # np.add.at correctly accumulates for repeated indices
                    np.add.at(grad_parent_np, key, g_out_np)
                else:
                    # Basic slicing/integer indexing: still conceptually an accumulation op
                    grad_parent_np[key] += g_out_np

                grad_parent = Tensor(
                    shape=self.shape, device=self.device, requires_grad=False, ctx=None
                )
                grad_parent.copy_from_numpy(grad_parent_np)
                return (grad_parent,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            # Optional: keep debug metadata
            ctx.saved_meta["getitem_key"] = key
            ctx.saved_meta["parent_shape"] = self.shape

            out._set_ctx(ctx)

        return out

    @staticmethod
    def stack(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """
        Stack a sequence of tensors along a new axis (CPU-only).

        This is the differentiable counterpart of `np.stack`. All input tensors must:
        - live on CPU
        - share the same shape
        - share the same device

        Parameters
        ----------
        tensors : Sequence[Tensor]
            Input tensors to stack. Must be non-empty and same-shape.
        axis : int, optional
            Axis at which the new dimension is inserted. Supports negative axes.
            Defaults to 0.

        Returns
        -------
        Tensor
            A new tensor whose shape is:
                out.shape = in.shape[:axis] + (len(tensors),) + in.shape[axis:]

        Notes
        -----
        - Forward returns a *copy* (not a view).
        - Backward splits `grad_out` along `axis` and routes each slice back to
          the corresponding parent tensor.
        """
        if len(tensors) == 0:
            raise ValueError("Tensor.stack() requires a non-empty sequence")

        # ---- Validate devices and shapes ----
        first = tensors[0]
        if not first.device.is_cpu():
            first._raise_device_not_supported("stack")

        dev = first.device
        in_shape = first.shape

        for i, t in enumerate(tensors):
            if not t.device.is_cpu():
                t._raise_device_not_supported("stack")
            if str(t.device) != str(dev):
                raise ValueError(
                    f"Tensor.stack() requires all tensors on the same device; "
                    f"tensors[0] is {dev!r} but tensors[{i}] is {t.device!r}"
                )
            if t.shape != in_shape:
                raise ValueError(
                    f"Tensor.stack() requires all tensors to have the same shape; "
                    f"expected {in_shape}, got {t.shape} at index {i}"
                )

        # Normalize axis (np.stack does this too, but we want predictable error messages)
        ndim = len(in_shape)
        # axis is in [-ndim-1, ndim]
        if axis < 0:
            axis = axis + (ndim + 1)
        if axis < 0 or axis > ndim:
            raise ValueError(
                f"axis {axis} out of bounds for stack with input ndim {ndim}"
            )

        # ---- Forward ----
        arrs = [t.to_numpy() for t in tensors]
        stacked = np.stack(arrs, axis=axis).astype(np.float32, copy=False)

        req = any(t.requires_grad for t in tensors)
        out = Tensor(shape=stacked.shape, device=dev, requires_grad=req, ctx=None)
        out.copy_from_numpy(stacked)

        # ---- Backward ----
        if req:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy()  # shape: stacked.shape

                grads: list[Optional["Tensor"]] = []
                for i, t in enumerate(tensors):
                    if not t.requires_grad:
                        grads.append(None)
                        continue

                    # Select the i-th slice along the stacked axis.
                    # Use take() to avoid view complexities; it returns an array.
                    gi_np = np.take(g, indices=i, axis=axis).astype(
                        np.float32, copy=False
                    )

                    gi = Tensor(
                        shape=t.shape, device=dev, requires_grad=False, ctx=None
                    )
                    gi.copy_from_numpy(gi_np)
                    grads.append(gi)

                return tuple(grads)

            ctx = Context(
                parents=tuple(tensors),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    def reshape(self, new_shape: tuple[int, ...]) -> "Tensor":
        """
        Return a reshaped view of this tensor.

        Notes
        -----
        - CPU-only.
        - Forward uses NumPy reshape (view when possible).
        - Backward reshapes grad_out back to the original tensor shape.
        - No data copy in forward.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("reshape")

        # NumPy reshape (may be a view)
        src_np = self.to_numpy()
        try:
            reshaped_np = src_np.reshape(new_shape)
        except Exception as e:
            raise ValueError(f"Invalid reshape from {self.shape} to {new_shape}") from e

        req = self.requires_grad
        out = Tensor(
            shape=reshaped_np.shape,
            device=self.device,
            requires_grad=req,
            ctx=None,
        )
        out.copy_from_numpy(reshaped_np)

        if req:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g_out_np = grad_out.to_numpy()

                # Reshape gradient back to input shape
                grad_parent_np = g_out_np.reshape(self.shape)

                grad_parent = Tensor(
                    shape=self.shape,
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                )
                grad_parent.copy_from_numpy(grad_parent_np)

                return (grad_parent,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    @staticmethod
    def concat(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """
        Concatenate a sequence of tensors along an existing axis (CPU-only).

        This is the differentiable counterpart of `np.concatenate`.

        Requirements
        ------------
        - `tensors` must be non-empty
        - all tensors must be CPU tensors
        - all tensors must share the same device
        - shapes must match on all dimensions except `axis`

        Parameters
        ----------
        tensors : Sequence[Tensor]
            Input tensors to concatenate.
        axis : int, optional
            Axis along which to concatenate. Supports negative axes.
            Defaults to 0.

        Returns
        -------
        Tensor
            Concatenated tensor.

        Notes
        -----
        Backward rule:
        - Split `grad_out` along `axis` into slices matching each input's size
          along that axis, and route each slice back to the corresponding parent.
        """
        if len(tensors) == 0:
            raise ValueError("Tensor.concat() requires a non-empty sequence")

        first = tensors[0]
        if not first.device.is_cpu():
            first._raise_device_not_supported("concat")

        dev = first.device
        ref_shape = first.shape

        # Validate ndim and normalize axis
        ndim = len(ref_shape)
        if ndim == 0:
            raise ValueError("Tensor.concat() does not support scalar tensors (ndim=0)")

        if axis < 0:
            axis = axis + ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {axis} out of bounds for concat with input ndim {ndim}"
            )

        # Validate devices and shapes (all dims except axis must match)
        sizes_along_axis: list[int] = []
        for i, t in enumerate(tensors):
            if not t.device.is_cpu():
                t._raise_device_not_supported("concat")
            if str(t.device) != str(dev):
                raise ValueError(
                    f"Tensor.concat() requires all tensors on the same device; "
                    f"tensors[0] is {dev!r} but tensors[{i}] is {t.device!r}"
                )
            if len(t.shape) != ndim:
                raise ValueError(
                    f"Tensor.concat() requires all tensors to have same ndim; "
                    f"expected {ndim}, got {len(t.shape)} at index {i}"
                )

            for d in range(ndim):
                if d == axis:
                    continue
                if t.shape[d] != ref_shape[d]:
                    raise ValueError(
                        "Tensor.concat() shape mismatch on non-concat dimension: "
                        f"dim={d}, expected {ref_shape[d]}, got {t.shape[d]} at index {i}"
                    )

            sizes_along_axis.append(t.shape[axis])

        # Forward
        arrs = [t.to_numpy() for t in tensors]
        out_np = np.concatenate(arrs, axis=axis).astype(np.float32, copy=False)

        req = any(t.requires_grad for t in tensors)
        out = Tensor(shape=out_np.shape, device=dev, requires_grad=req, ctx=None)
        out.copy_from_numpy(out_np)

        # Backward
        if req:
            # Build slice boundaries along axis
            # e.g. sizes [2,3,1] -> offsets [0,2,5,6]
            offsets = [0]
            for s in sizes_along_axis:
                offsets.append(offsets[-1] + int(s))

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy()
                grads: list[Optional["Tensor"]] = []

                for i, t in enumerate(tensors):
                    if not t.requires_grad:
                        grads.append(None)
                        continue

                    start = offsets[i]
                    end = offsets[i + 1]

                    # Build slicing object for all dims
                    slicer = [slice(None)] * ndim
                    slicer[axis] = slice(start, end)

                    gi_np = g[tuple(slicer)].astype(np.float32, copy=False)

                    gi = Tensor(
                        shape=t.shape, device=dev, requires_grad=False, ctx=None
                    )
                    gi.copy_from_numpy(gi_np)
                    grads.append(gi)

                return tuple(grads)

            ctx = Context(
                parents=tuple(tensors),
                backward_fn=backward_fn,
            )
            # Optional debug metadata
            ctx.saved_meta["concat_axis"] = axis
            ctx.saved_meta["concat_sizes"] = sizes_along_axis

            out._set_ctx(ctx)

        return out

        # ----------------------------

    # Matrix ops (2D)
    # ----------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication (2D): out = self @ other.

        Requirements
        ------------
        - CPU-only
        - both operands must be 2D
        - inner dimensions must match: (N, K) @ (K, M) -> (N, M)

        Backward
        --------
        If out = A @ B, then:
        - dL/dA = dL/dout @ B^T
        - dL/dB = A^T @ dL/dout
        """
        if not self.device.is_cpu() or not other.device.is_cpu():
            self._raise_device_not_supported("matmul")

        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(
                f"matmul requires 2D tensors, got {self.shape} and {other.shape}"
            )

        n, k1 = self.shape
        k2, m = other.shape
        if k1 != k2:
            raise ValueError(
                f"matmul shape mismatch: {self.shape} @ {other.shape} (inner dims {k1} vs {k2})"
            )

        req = self._result_requires_grad(self, other)

        out = Tensor(shape=(n, m), device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(self.to_numpy() @ other.to_numpy())

        if req:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")
                if grad_out.shape != (n, m):
                    raise ValueError(
                        f"grad_out shape mismatch: expected {(n, m)}, got {grad_out.shape}"
                    )

                grad_a = None
                grad_b = None

                # dA = dOut @ B^T
                if self.requires_grad:
                    ga_np = grad_out.to_numpy() @ other.to_numpy().T
                    grad_a = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                    )
                    grad_a.copy_from_numpy(ga_np)

                # dB = A^T @ dOut
                if other.requires_grad:
                    gb_np = self.to_numpy().T @ grad_out.to_numpy()
                    grad_b = Tensor(
                        shape=other.shape,
                        device=other.device,
                        requires_grad=False,
                        ctx=None,
                    )
                    grad_b.copy_from_numpy(gb_np)

                return (grad_a, grad_b)

            ctx = Context(
                parents=(self, other),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Operator overload for matrix multiplication: self @ other.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"@ only supports Tensor operands, got {type(other)!r}")
        return self.matmul(other)

    # ----------------------------
    # Transpose (2D)
    # ----------------------------
    def transpose(self) -> "Tensor":
        """
        2D transpose: out[i, j] = self[j, i].

        Requirements
        ------------
        - CPU-only
        - input must be 2D

        Backward
        --------
        If out = A^T, then dL/dA = (dL/dout)^T
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("transpose")

        if len(self.shape) != 2:
            raise ValueError(f"transpose requires a 2D tensor, got shape={self.shape}")

        r, c = self.shape
        req = self.requires_grad

        out = Tensor(shape=(c, r), device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(self.to_numpy().T)

        if req:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")
                if grad_out.shape != (c, r):
                    raise ValueError(
                        f"grad_out shape mismatch: expected {(c, r)}, got {grad_out.shape}"
                    )

                g_np = grad_out.to_numpy().T  # back to (r, c)
                grad_parent = Tensor(
                    shape=self.shape, device=self.device, requires_grad=False, ctx=None
                )
                grad_parent.copy_from_numpy(g_np)
                return (grad_parent,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            out._set_ctx(ctx)

        return out

    @property
    def T(self) -> "Tensor":
        """
        Convenience property for 2D transpose.
        """
        return self.transpose()

    # ----------------------------
    # Sum with axis
    # ----------------------------
    def sum(self, axis: Optional[int] = None) -> "Tensor":
        """
        Sum elements of the tensor.

        Parameters
        ----------
        axis : Optional[int]
            - None: sum all elements -> scalar (existing behavior)
            - int: sum along the given axis -> tensor with that axis removed

        Notes
        -----
        Backward rule:
        - axis is None:
            d(sum(x))/dx = 1
        - axis is int:
            grad_out is broadcast back to input shape along the reduced axis
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("sum")

        import numpy as np  # local import ok inside Tensor

        # axis=None: keep your original behavior, but with the new signature
        if axis is None:
            value = float(np.sum(self._data))
            out = Tensor(shape=(), device=self.device, requires_grad=self.requires_grad)
            out.copy_from_numpy(np.array(value, dtype=np.float32))

            if self.requires_grad:

                def backward_fn(grad_out: "Tensor"):
                    grad = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                    )
                    grad.copy_from_numpy(
                        np.ones(self.shape, dtype=np.float32)
                        * float(np.asarray(grad_out.to_numpy()))
                    )
                    return (grad,)

                ctx = Context(
                    parents=(self,),
                    backward_fn=backward_fn,
                )
                out._set_ctx(ctx)

            return out

        # axis=int: reduce along that axis
        if not isinstance(axis, int):
            raise TypeError(f"sum(axis=...) expects int or None, got {type(axis)!r}")

        ndim = len(self.shape)
        if ndim == 0:
            raise ValueError("sum(axis=...) is not defined for scalar tensors")

        # normalize negative axis
        if axis < 0:
            axis = axis + ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {axis} out of bounds for tensor with ndim {ndim}")

        out_np = np.sum(self._data, axis=axis).astype(np.float32, copy=False)
        out_shape = out_np.shape  # already reduced

        out = Tensor(
            shape=out_shape,
            device=self.device,
            requires_grad=self.requires_grad,
            ctx=None,
        )
        out.copy_from_numpy(out_np)

        if self.requires_grad:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")
                if grad_out.shape != out_shape:
                    raise ValueError(
                        f"grad_out shape mismatch: expected {out_shape}, got {grad_out.shape}"
                    )

                g = grad_out.to_numpy()

                # Re-insert the reduced axis as size-1, then broadcast to input shape.
                g_exp = np.expand_dims(g, axis=axis)
                g_broadcast = np.broadcast_to(g_exp, self.shape).astype(
                    np.float32, copy=False
                )

                grad_parent = Tensor(
                    shape=self.shape,
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                )
                grad_parent.copy_from_numpy(g_broadcast)
                return (grad_parent,)

            ctx = Context(
                parents=(self,),
                backward_fn=backward_fn,
            )
            ctx.saved_meta["sum_axis"] = axis
            out._set_ctx(ctx)

        return out

    def copy_from(self, other: "Tensor") -> None:
        """
        Copy data from another tensor into this tensor (in-place).

        This is the canonical tensor-to-tensor copy operation and should be preferred
        over `to_numpy()` + `copy_from_numpy()` in internal framework code.

        Parameters
        ----------
        other : Tensor
            Source tensor.

        Raises
        ------
        ValueError
            If shapes do not match.
        RuntimeError
            If devices are incompatible or unsupported.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"copy_from expects a Tensor, got {type(other)!r}")

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        if str(self.device) != str(other.device):
            raise RuntimeError(
                f"Device mismatch in copy_from: {self.device} vs {other.device}"
            )

        # ---- CPU backend ----
        if self.device.is_cpu():
            # IMPORTANT:
            # We do not attach ctx, do not propagate requires_grad.
            self._data[...] = other._data
            return

        # ---- CUDA backend (future) ----
        if self.device.is_cuda():
            # Placeholder for:
            # cudaMemcpy / async copy / stream-aware copy
            raise NotImplementedError("CUDA copy_from not yet implemented")

        raise RuntimeError(f"Unsupported device type: {self.device}")
