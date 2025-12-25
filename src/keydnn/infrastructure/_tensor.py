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

from typing import Any, Union, Optional, Sequence

import numpy as np

from ..domain._tensor import ITensor
from ..domain.device._device import Device
from ..domain._errors import DeviceNotSupportedError
from .tensor._tensor_context import Context

Number = Union[int, float]


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
        For CUDA devices, this implementation does not auto-allocate device memory
        (because the allocator lives in the native CUDA DLL, which is not available
        at construction time here). CUDA tensors should be constructed via
        `Tensor._from_devptr` or filled by higher-level CUDA allocation utilities.

        Raises
        ------
        ValueError
            If the device type is unsupported.
        """
        d = self._device

        is_cpu = getattr(d, "is_cpu", None)
        if callable(is_cpu) and is_cpu():
            self._data = np.zeros(self._shape, dtype=np.float32)
            return

        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            # Device pointer handle (DevPtr). 0 means "not allocated / not set".
            self._data = 0
            return

        raise ValueError(f"Unsupported device type: {type(d)!r} value={d!r}")

    @property
    def data(self) -> int | np.ndarray:
        """
        Return the underlying storage for this tensor.

        Returns
        -------
        numpy.ndarray or int
            - If the tensor is on CPU: a NumPy ndarray backing the tensor.
            - If the tensor is on CUDA: a device pointer handle (DevPtr) as a Python int.

        Notes
        -----
        - For CUDA tensors, the returned integer is a raw device pointer value
          (uintptr_t) produced by the native CUDA allocator.
        - A CUDA tensor created via the regular constructor may have `data == 0`
          (uninitialized device pointer). Prefer constructing CUDA tensors with
          `Tensor._from_devptr(...)` when you already have an allocated buffer.
        """
        d = self._device

        is_cpu = getattr(d, "is_cpu", None)
        if callable(is_cpu) and is_cpu():
            return self._data

        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            # DevPtr (uintptr_t) represented as Python int
            return int(self._data)

        raise ValueError(f"Unsupported device type: {type(d)!r} value={d!r}")

    @classmethod
    def _from_devptr(
        cls,
        dev_ptr: int,
        *,
        shape: tuple[int, ...],
        device: Device,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
    ) -> "Tensor":
        """
        Construct a CUDA tensor backed by an existing device pointer (DevPtr).

        Parameters
        ----------
        dev_ptr : int
            Raw device pointer handle (uintptr_t) returned by the native CUDA allocator.
        shape : tuple[int, ...]
            Tensor shape.
        device : Device
            CUDA device descriptor. Must satisfy `device.is_cuda() == True`.
        requires_grad : bool, optional
            Whether this tensor should accumulate gradients during backprop.
        ctx : Optional[Context], optional
            Optional autograd context to attach.

        Returns
        -------
        Tensor
            A tensor whose `.data` is the provided device pointer handle.

        Raises
        ------
        ValueError
            If `device` is not a CUDA device, or if `dev_ptr` is invalid.

        Notes
        -----
        - This function does not take ownership semantics beyond storing the pointer.
          The caller is responsible for freeing device memory (typically via your
          native CUDA `cuda_free` wrapper) when appropriate.
        - This constructor bypasses `__init__` and does not allocate memory.
        """
        is_cuda = getattr(device, "is_cuda", None)
        if not (callable(is_cuda) and is_cuda()):
            raise ValueError(
                f"_from_devptr requires a CUDA device; got device={device!r}"
            )

        if dev_ptr is None:
            raise ValueError("dev_ptr must be an int (uintptr_t), got None")

        # Accept dev_ptr == 0 only if you *intentionally* use it as a null handle.
        # If you prefer stricter behavior, change this to `if int(dev_ptr) == 0: raise ...`.
        dp = int(dev_ptr)

        obj = cls.__new__(cls)  # bypass __init__
        obj._shape = shape
        obj._device = device
        obj._data = dp

        # --- autograd fields (optional) ---
        obj._requires_grad = bool(requires_grad)
        obj._grad = None
        obj._ctx = ctx

        return obj

    def __repr__(self) -> str:
        """
        Return a human-readable string representation of the tensor.

        Returns
        -------
        str
            A string describing the tensor's shape, device, and storage details.
        """
        d = self._device
        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            return f"Tensor(shape={self._shape}, device={d}, data=DevPtr({int(self._data)}))"
        return f"Tensor(shape={self._shape}, device={d}, dtype={self._data.dtype})"

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
        Return a stable, human-readable description of underlying storage.

        Contract
        --------
        - CPU tensors: describe the NumPy ndarray storage.
        - CUDA tensors: return a stable placeholder string that includes:
            - device index
            - tensor shape
            - (if available) the device pointer value

        Notes
        -----
        This is a debugging aid and intentionally does not expose full contents.
        """
        d = self._device

        is_cpu = getattr(d, "is_cpu", None)
        if callable(is_cpu) and is_cpu():
            # CPU: we expect a NumPy ndarray
            try:
                arr = self._data  # typically np.ndarray
                return f"CPU ndarray dtype={getattr(arr, 'dtype', None)} shape={self._shape}"
            except Exception:
                return f"CPU storage shape={self._shape}"

        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            dev_index = getattr(d, "index", None)

            # If CUDA storage is already a string placeholder, keep it.
            if isinstance(self._data, str):
                return self._data

            # If CUDA storage is a devptr handle (int), format a stable message.
            if isinstance(self._data, int):
                return (
                    f"CUDA Tensor on device {dev_index} with shape {self._shape} "
                    f"(devptr={self._data})"
                )

            # Fallback if something unexpected is stored
            return f"CUDA Tensor on device {dev_index} with shape {self._shape}"

        return f"Unknown device storage shape={self._shape}"

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

    def exp(self) -> "Tensor":
        """
        Compute the elementwise exponential of the tensor.

        Returns
        -------
        Tensor
            A tensor of the same shape as `self` with exp applied elementwise.

        Notes
        -----
        Backward rule:
            d(exp(x))/dx = exp(x)
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("exp")

        out = Tensor(
            shape=self.shape, device=self.device, requires_grad=self.requires_grad
        )
        out.copy_from_numpy(np.exp(self.to_numpy()).astype(np.float32, copy=False))

        if self.requires_grad:
            # Save output to reuse in backward
            ctx = Context(
                parents=(self,),
                backward_fn=lambda grad_out: (grad_out * out,),
            )
            out._set_ctx(ctx)

        return out

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """
        Sum elements of the tensor.

        Parameters
        ----------
        axis : Optional[int], optional
            Axis along which to sum. If None, sums all elements into a scalar.
        keepdims : bool, optional
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        Tensor
            Reduced tensor.

        Notes
        -----
        Backward rule:
            Gradient is broadcast back to input shape.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("sum")

        x_np = self.to_numpy()
        if axis is None:
            value = np.sum(x_np)
            out_shape = () if not keepdims else tuple(1 for _ in self.shape)
        else:
            if not isinstance(axis, int):
                raise TypeError("axis must be int or None")
            ndim = x_np.ndim
            axis_ = axis if axis >= 0 else ndim + axis
            if axis_ < 0 or axis_ >= ndim:
                raise ValueError(f"axis {axis} out of bounds for ndim {ndim}")
            value = np.sum(x_np, axis=axis_, keepdims=keepdims)
            out_shape = value.shape

        out = Tensor(
            shape=out_shape, device=self.device, requires_grad=self.requires_grad
        )
        out.copy_from_numpy(np.asarray(value, dtype=np.float32))

        if self.requires_grad:

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = np.asarray(grad_out.to_numpy(), dtype=np.float32)

                # Expand grad to input shape
                if axis is None:
                    grad_np = np.ones(self.shape, dtype=np.float32) * float(
                        np.asarray(g)
                    )
                else:
                    # If keepdims=False, need to re-insert the reduced axis
                    if not keepdims:
                        g = np.expand_dims(g, axis=axis_)

                    grad_np = np.ones(self.shape, dtype=np.float32) * g

                grad = Tensor(shape=self.shape, device=self.device, requires_grad=False)
                grad.copy_from_numpy(grad_np)
                return (grad,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            ctx.saved_meta["axis"] = axis
            ctx.saved_meta["keepdims"] = keepdims
            out._set_ctx(ctx)

        return out

    def max(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
        """
        Compute the maximum along an axis (CPU-only).

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the max. Defaults to -1.
        keepdims : bool, optional
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        Tensor
            Tensor of max values.

        Notes
        -----
        Backward rule:
            Gradient is routed to positions equal to the max (ties split by mask),
            i.e., dx = grad_out * 1[x == max(x)].
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("max")

        x_np = self.to_numpy()
        ndim = x_np.ndim
        axis_ = axis if axis >= 0 else ndim + axis
        if axis_ < 0 or axis_ >= ndim:
            raise ValueError(f"axis {axis} out of bounds for ndim {ndim}")

        m_np = np.max(x_np, axis=axis_, keepdims=keepdims).astype(
            np.float32, copy=False
        )
        out = Tensor(
            shape=m_np.shape, device=self.device, requires_grad=self.requires_grad
        )
        out.copy_from_numpy(m_np)

        if self.requires_grad:
            # Save input and max output for backward mask
            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy().astype(np.float32, copy=False)

                # If keepdims=False, expand dims for mask alignment
                m = m_np
                g_aligned = g
                if not keepdims:
                    m = np.expand_dims(m, axis=axis_)
                    g_aligned = np.expand_dims(g_aligned, axis=axis_)

                mask = (x_np == m).astype(np.float32)
                grad_np = mask * g_aligned

                grad = Tensor(shape=self.shape, device=self.device, requires_grad=False)
                grad.copy_from_numpy(grad_np.astype(np.float32, copy=False))
                return (grad,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            ctx.saved_meta["axis"] = axis_
            ctx.saved_meta["keepdims"] = keepdims
            out._set_ctx(ctx)

        return out

    def broadcast_to(self, shape: tuple[int, ...]) -> "Tensor":
        """
        Broadcast this tensor to a target shape by explicit expansion (CPU-only).

        Parameters
        ----------
        shape : tuple[int, ...]
            Target shape to broadcast to.

        Returns
        -------
        Tensor
            Broadcasted tensor (materialized copy).

        Notes
        -----
        - This is an explicit, opt-in broadcasting primitive to keep binary ops strict.
        - Backward reduces gradients by summing over broadcasted dimensions.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("broadcast_to")

        src = self.to_numpy()
        try:
            out_np = np.broadcast_to(src, shape).astype(np.float32, copy=False)
        except Exception as e:
            raise ValueError(f"Cannot broadcast shape {self.shape} to {shape}") from e

        req = self.requires_grad
        out = Tensor(shape=shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(out_np)

        if req:
            src_shape = self.shape

            def backward_fn(grad_out: "Tensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy().astype(np.float32, copy=False)

                # Align src_shape to target rank by left-padding with ones
                src_rank = len(src_shape)
                tgt_rank = len(shape)
                padded_src = (1,) * (tgt_rank - src_rank) + src_shape

                # Sum over axes that were broadcasted (src dim == 1 and target dim > 1)
                reduce_axes = []
                for i, (sd, td) in enumerate(zip(padded_src, shape)):
                    if sd == 1 and td != 1:
                        reduce_axes.append(i)

                if reduce_axes:
                    g = np.sum(g, axis=tuple(reduce_axes), keepdims=True)

                # Remove left padding dims to return to src_shape
                if tgt_rank != src_rank:
                    for _ in range(tgt_rank - src_rank):
                        g = np.squeeze(g, axis=0)

                grad = Tensor(shape=src_shape, device=self.device, requires_grad=False)
                grad.copy_from_numpy(g.astype(np.float32, copy=False))
                return (grad,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            ctx.saved_meta["broadcast_from"] = src_shape
            ctx.saved_meta["broadcast_to"] = shape
            out._set_ctx(ctx)

        return out

    def item(self) -> float:
        """
        Return the value of a scalar (or single-element) CPU tensor as a Python float.

        Raises
        ------
        RuntimeError
            If called on a non-CPU tensor.
        ValueError
            If the tensor is not scalar and does not contain exactly 1 element.
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("item")

        # Accept scalar shape () OR any shape with exactly one element.
        if self.shape == ():
            # _data is a 0-d ndarray on CPU
            return float(self._data)  # type: ignore[arg-type]
        if self.numel() != 1:
            raise ValueError(
                f"Tensor.item() requires a scalar/1-element tensor, got shape={self.shape}"
            )

        # For (1,) or (1,1,...) tensors
        return float(self._data.reshape(-1)[0])

    def sqrt(self) -> "Tensor":
        """
        Elementwise square root.

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, containing sqrt(self) elementwise.

        Notes
        -----
        - CPU-only for now.
        - NumPy is used only inside Tensor to perform the actual numeric kernel.
        - Autograd: if `self.requires_grad` is True, attaches a Context with parent (self,).
        """
        if not self.device.is_cpu():
            self._raise_device_not_supported("sqrt")

        x_np = self.to_numpy()
        y_np = np.sqrt(x_np).astype(np.float32, copy=False)

        out = Tensor(
            shape=self.shape, device=self.device, requires_grad=self.requires_grad
        )
        out.copy_from_numpy(y_np)

        if self.requires_grad:
            # Save output for backward (so we don't recompute sqrt).
            def backward_fn(grad_out: "Tensor") -> Sequence[Optional["Tensor"]]:
                if not grad_out.device.is_cpu():
                    grad_out._raise_device_not_supported("sqrt_backward")

                # d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / out
                # Use numpy here because we're inside Tensor (your requirement).
                go = grad_out.to_numpy().astype(np.float32, copy=False)
                y = out.to_numpy().astype(np.float32, copy=False)

                # Be careful about division-by-zero; follow numpy semantics.
                gx_np = (go * (0.5 / y)).astype(np.float32, copy=False)

                gx = Tensor(shape=self.shape, device=self.device, requires_grad=False)
                gx.copy_from_numpy(gx_np)
                return (gx,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            # Store `out` for backward (reused derivative)
            ctx.save_for_backward(out)
            out._set_ctx(ctx)

        return out

    @staticmethod
    def _from_numpy(arr: ITensor, *, device, requires_grad: bool = False) -> "Tensor":
        """
        Construct a Tensor from a NumPy array.

        This is a low-level factory method that creates a new `Tensor` with
        storage allocated on the specified device and initializes its contents
        by copying data from a NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            Source NumPy array containing the tensor data. The array's shape
            determines the shape of the resulting tensor.
        device : Device
            Target device on which the tensor should be created.
        requires_grad : bool, optional
            Whether the resulting tensor should participate in automatic
            differentiation. Defaults to False.

        Returns
        -------
        Tensor
            A newly created tensor whose contents are copied from `arr`.

        Notes
        -----
        - This method centralizes the NumPy → Tensor boundary inside the `Tensor`
        implementation to keep higher-level code NumPy-free.
        - The input array is copied into the tensor's internal storage; subsequent
        modifications to `arr` do not affect the tensor.
        - No autograd context is attached during construction. Gradient tracking
        begins only when this tensor is used in differentiable operations.
        """
        t = Tensor(
            shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None
        )
        t.copy_from_numpy(arr)
        return t

    @staticmethod
    def rand(shape, *, device, requires_grad: bool = False) -> "Tensor":
        """
        Create a tensor filled with uniform random values in [0, 1) on the given device.

        Notes
        -----
        - CPU-only for now.
        - Random generation is intentionally implemented inside Tensor so higher-level
        modules (e.g., Dropout) do not depend on NumPy directly.
        """
        if not device.is_cpu():
            # match your existing CPU-only policy
            raise RuntimeError("rand is only supported for CPU tensors for now.")

        # NumPy is allowed here by your rule: only inside Tensor.
        import numpy as np

        arr = np.random.rand(*shape).astype(np.float32, copy=False)
        t = Tensor(shape=shape, device=device, requires_grad=requires_grad, ctx=None)
        t.copy_from_numpy(arr)
        return t

    @staticmethod
    def full(
        shape,
        fill_value: float,
        *,
        device,
        requires_grad: bool = False,
    ) -> "Tensor":
        """
        Create a tensor filled with a constant value.

        This is a convenience factory that allocates a new tensor on the requested
        device and fills all elements with `fill_value`.

        Parameters
        ----------
        shape
            Desired tensor shape. May be any shape accepted by NumPy (including
            `()` for a scalar tensor).
        fill_value : float
            Constant value to write into every element.
        device
            Target device placement. Currently only CPU is supported.
        requires_grad : bool, optional
            Whether the returned tensor should participate in autograd.
            Defaults to False.

        Returns
        -------
        Tensor
            A newly allocated tensor with the given shape, filled with `fill_value`.

        Raises
        ------
        RuntimeError
            If `device` is not CPU.

        Notes
        -----
        - NumPy is used internally to generate the initial filled buffer.
        Higher-level code should prefer calling `Tensor.full(...)` rather than
        allocating NumPy arrays directly.
        - The returned tensor is created with `ctx=None` (no autograd history).
        """
        if not device.is_cpu():
            raise RuntimeError("full is only supported for CPU tensors for now.")

        import numpy as np

        arr = np.full(shape, fill_value, dtype=np.float32)
        t = Tensor(
            shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None
        )
        t.copy_from_numpy(arr)
        return t

    def tanh(self) -> "Tensor":
        """
        Elementwise hyperbolic tangent.

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, with tanh applied elementwise.

        Notes
        -----
        This method delegates to the autograd tanh Function/op and does not use
        NumPy here (NumPy remains inside Tensor/ops only).
        """
        from ._function import TanhFn  # adjust if needed

        ctx = Context(parents=(self,), backward_fn=None)
        out = TanhFn.forward(ctx, self)
        ctx.backward_fn = lambda grad_out: (TanhFn.backward(ctx, grad_out),)
        out._set_ctx(ctx)
        return out

    def sigmoid(self) -> "Tensor":
        """
        Elementwise logistic sigmoid.

        Computes:

            sigmoid(x) = 1 / (1 + exp(-x))

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, with `sigmoid` applied
            elementwise.

        Notes
        -----
        This is a thin convenience wrapper around `SigmoidFn` located in
        `._function`.
        """
        from ._function import SigmoidFn  # adjust import to your project layout

        # Build context with parents AND a callable backward_fn
        ctx = Context(parents=(self,), backward_fn=None)

        out = SigmoidFn.forward(ctx, self)

        # IMPORTANT: Tensor.backward() expects ctx.backward_fn to be callable
        ctx.backward_fn = lambda grad_out: (SigmoidFn.backward(ctx, grad_out),)

        out._set_ctx(ctx)
        return out

    @staticmethod
    def zeros(
        *, shape: tuple[int, ...], device, requires_grad: bool = False
    ) -> "Tensor":
        """
        Construct a tensor filled with zeros.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape.
        device : Device
            Target device.
        requires_grad : bool, optional
            Whether the result participates in autograd.

        Returns
        -------
        Tensor
            A tensor of given shape filled with 0.0 (float32).
        """
        import numpy as np

        arr = np.zeros(shape, dtype=np.float32)
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
        out.copy_from_numpy(arr)
        return out

    @staticmethod
    def ones(
        *, shape: tuple[int, ...], device, requires_grad: bool = False
    ) -> "Tensor":
        """
        Construct a tensor filled with ones.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape.
        device : Device
            Target device.
        requires_grad : bool, optional
            Whether the result participates in autograd.

        Returns
        -------
        Tensor
            A tensor of given shape filled with 1.0 (float32).
        """
        import numpy as np

        arr = np.ones(shape, dtype=np.float32)
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
        out.copy_from_numpy(arr)
        return out
