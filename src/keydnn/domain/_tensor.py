"""
Tensor interface definitions.

This module defines the domain-level interface for tensor-like objects using
structural typing. The interface captures the minimal, backend-agnostic
properties required for tensors to participate in computation graphs,
modules, and optimization workflows.

Notes
-----
Historically this protocol aimed to be minimal. In the current codebase, the
infrastructure Tensor implementation exposes additional public operations
(e.g., exp/max/broadcast_to/concat). This protocol mirrors that surface area
so domain code can type against the full tensor API when desired.
"""

from __future__ import annotations

from typing import Any, Type, Optional, Protocol, Sequence, Union, runtime_checkable

from .device._device_protocol import DeviceLike
from .device._device import Device
from .types._numpy import NDArrayLike

Number = Union[int, float]


@runtime_checkable
class ITensor(Protocol):
    """
    Tensor interface.

    An `ITensor` represents a multi-dimensional array that participates in
    numerical computation and, optionally, automatic differentiation.

    This protocol uses structural typing (duck typing) so that different
    concrete backends (e.g., NumPy, CUDA) can satisfy the same contract.

    Notes
    -----
    - The protocol includes autograd-facing hooks (`backward`, `grad`,
      `requires_grad`) because current domain/infrastructure code expects
      them to exist.
    - The protocol currently mirrors the public API of the NumPy-backed
      `Tensor` implementation to keep typing consistent across layers.
    - If you later want a stricter layering boundary, consider splitting this
      into smaller protocols (e.g., `ITensorCore`, `ITensorAutograd`,
      `ITensorOps`).
    """

    # ---------------------------------------------------------------------
    # Core identity / placement
    # ---------------------------------------------------------------------

    @property
    def dtype(self) -> Any:
        """
        Return the element dtype of this tensor.

        Returns
        -------
        np.dtype
            NumPy dtype representing the tensor element type.

        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the tensor.

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

        Returns
        -------
        DeviceLike
            The tensor's device placement descriptor.
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
    def to_numpy(self) -> NDArrayLike:
        """
        Convert the tensor to a backend-native array object.

        In the current NumPy backend, this returns `np.ndarray` and is only
        available for CPU tensors.

        Returns
        -------
        NDArrayLike
            Backend-native array (e.g., `np.ndarray`).

        Raises
        ------
        RuntimeError
            If conversion is unavailable for the tensor's device/backend.
        """
        ...

    def copy_from_numpy(self, arr: NDArrayLike) -> None:
        """
        Copy data from a backend-native array object into this tensor.

        In the current NumPy backend, the input is expected to be array-like
        and shape-compatible.

        Parameters
        ----------
        arr : NDArrayLike
            Source array-like object.

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
        Fill the tensor with a scalar value.

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
        """
        ...

    @staticmethod
    def concat(tensors: Sequence["ITensor"], axis: int = 0) -> "ITensor":
        """
        Concatenate a sequence of tensors along an existing axis.

        Parameters
        ----------
        tensors : Sequence[ITensor]
            Input tensors to concatenate. Must be non-empty and compatible.
        axis : int, optional
            Axis along which to concatenate. Supports negative axes.

        Returns
        -------
        ITensor
            Concatenated tensor.

        Notes
        -----
        If autograd is enabled, gradients are split along `axis` and routed back
        to the corresponding parents.
        """
        ...

    def broadcast_to(self, shape: tuple[int, ...]) -> "ITensor":
        """
        Broadcast this tensor to a target shape by explicit expansion.

        Parameters
        ----------
        shape : tuple[int, ...]
            Target shape to broadcast to.

        Returns
        -------
        ITensor
            Broadcasted tensor.

        Notes
        -----
        This is an explicit broadcasting primitive used to keep binary ops strict
        (i.e., no implicit broadcasting in add/mul/etc.). If autograd is enabled,
        the backward pass reduces gradients by summing over broadcasted dimensions.
        """
        ...

    # ---------------------------------------------------------------------
    # Reductions and elementwise transforms
    # ---------------------------------------------------------------------
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "ITensor":
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
        ITensor
            Reduced tensor.

        Notes
        -----
        If autograd is enabled, gradients are broadcast back to the input shape.
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

    def exp(self) -> "ITensor":
        """
        Compute the elementwise exponential of the tensor.

        Returns
        -------
        ITensor
            A tensor of the same shape as `self` containing `exp(self)` elementwise.

        Notes
        -----
        If autograd is enabled, the backward rule is conceptually:
            d(exp(x))/dx = exp(x)
        """
        ...

    def max(self, axis: int = -1, keepdims: bool = False) -> "ITensor":
        """
        Compute the maximum along an axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the max. Supports negative axes.
        keepdims : bool, optional
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        ITensor
            Tensor of max values.

        Notes
        -----
        Many implementations route gradients to positions equal to the max.
        Handling of ties is backend-defined (often a mask-based split).
        """
        ...

    # ---------------------------------------------------------------------
    # Matrix ops (2D) and transpose (2D)
    # ---------------------------------------------------------------------
    def matmul(self, other: "ITensor") -> "ITensor":
        """
        Matrix multiplication (2D): out = self @ other.

        Parameters
        ----------
        other : ITensor
            Right-hand operand.

        Returns
        -------
        ITensor
            The matrix product with shape (N, M) for (N, K) @ (K, M).

        Notes
        -----
        If autograd is enabled and out = A @ B, then conceptually:
        - dL/dA = dL/dout @ B^T
        - dL/dB = A^T @ dL/dout
        """
        ...

    def __matmul__(self, other: "ITensor") -> "ITensor":
        """
        Operator overload for matrix multiplication: self @ other.
        """
        ...

    def transpose(self) -> "ITensor":
        """
        2D transpose: out[i, j] = self[j, i].

        Returns
        -------
        ITensor
            The transposed tensor with shape (C, R) for input shape (R, C).

        Notes
        -----
        If autograd is enabled and out = A^T, then conceptually:
            dL/dA = (dL/dout)^T
        """
        ...

    @property
    def T(self) -> "ITensor":
        """
        Convenience property for 2D transpose (equivalent to `transpose()`).
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
        """
        ...

    def __gt__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise greater-than comparison.

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
        """
        ...

    def _get_ctx(self) -> Optional[Any]:
        """
        Return the backward context attached to this tensor, if any.

        Returns
        -------
        Optional[Any]
            The backend-defined autograd context object, or None if absent.
        """
        ...

    def _accumulate_grad_(self, g: "ITensor") -> None:
        """
        In-place accumulate gradient `g` into `self.grad`.

        Parameters
        ----------
        g : ITensor
            Gradient to accumulate.
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
        ...

    def sqrt(self) -> "ITensor":
        """
        Elementwise square root.

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, containing sqrt(self) elementwise.
        """
        ...

    @classmethod
    def _from_numpy(
        cls: Type[ITensor], arr: ITensor, *, device: Device, requires_grad: bool = False
    ) -> "ITensor":
        """
        Create a tensor from a NumPy-compatible array.

        This is a low-level construction utility intended for use by backend
        implementations and internal framework code. The returned tensor is
        initialized with data copied from the provided array and placed on
        the specified device.

        Parameters
        ----------
        arr : Any
            A NumPy-compatible array object containing the source data.
            Concrete implementations may require this to be an `np.ndarray`.
        device : DeviceLike
            Target device on which the tensor should be created.
        requires_grad : bool, optional
            Whether the resulting tensor should participate in automatic
            differentiation. Defaults to False.

        Returns
        -------
        ITensor
            A newly created tensor containing the data from `arr`.

        Notes
        -----
        - This method is intentionally underscored to signal that it is not
        part of the public, user-facing tensor API.
        - Implementations may copy or cast the input data as required by the
        backend (e.g., enforcing `float32` dtype on CPU).
        - Autograd context is not attached during construction; any gradient
        tracking begins from subsequent differentiable operations.
        """
        ...

    @staticmethod
    def rand(shape, *, device, requires_grad: bool = False) -> "ITensor":
        """
        Create a tensor filled with uniform random values in [0, 1) on the given device.

        Notes
        -----
        - CPU-only for now.
        - Random generation is intentionally implemented inside Tensor so higher-level
        modules (e.g., Dropout) do not depend on NumPy directly.
        """
        ...

    @classmethod
    def full(
        cls: Type[ITensor],
        shape: tuple,
        fill_value: float,
        *,
        device: Device,
        requires_grad: bool = False,
    ) -> "ITensor":
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
        ...

    def tanh(self) -> "ITensor":
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
        ...

    def sigmoid(self) -> "ITensor":
        """
        Elementwise logistic sigmoid.

        Computes:

            sigmoid(x) = 1 / (1 + exp(-x))

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, with `sigmoid` applied
            elementwise.
        """
        ...

    @classmethod
    def zeros(
        cls: Type[ITensor],
        *,
        shape: tuple[int, ...],
        device,
        requires_grad: bool = False,
    ) -> "ITensor":
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
        ...

    @classmethod
    def ones(
        cls: Type[ITensor],
        *,
        shape: tuple[int, ...],
        device,
        requires_grad: bool = False,
    ) -> "ITensor":
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
        ITensor
            A tensor of given shape filled with 1.0 (float32).
        """
        ...

    @property
    def data(self) -> Union[int, NDArrayLike]:
        """
        Return the underlying storage for this tensor.

        Returns
        -------
        numpy.ndarray or int
            - If the tensor is on CPU: a NumPy ndarray backing the tensor.
            - If the tensor is on CUDA: a device pointer handle (DevPtr) as a Python int.

        """
        ...

    def __repr__(self) -> str:
        """
        Return a human-readable string representation of the tensor.

        Returns
        -------
        str
            A string describing the tensor's shape, device, and storage details.
        """
        ...

    def __div__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise division (legacy alias for true division).

        Notes
        -----
        Python 3 uses `__truediv__` for `/`. This method exists for compatibility
        with code that still calls `__div__` explicitly.
        """
        ...

    def __rdiv__(self, other: Number) -> "ITensor":
        """
        Right-hand division (legacy alias for right true division).

        Notes
        -----
        Python 3 uses `__rtruediv__`. This method exists for compatibility with
        code that still calls `__rdiv__` explicitly.
        """
        ...

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __ge__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise greater-than-or-equal comparison (no gradients).

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            A float32 tensor with 1.0 where `self >= other`, else 0.0.

        Notes
        -----
        Comparison operations do not participate in autograd in this minimal
        implementation (the result always has `requires_grad=False`).

        Implementation
        --------------
        Uses only `gt` + `neg`:
            a >= b  <=>  not (b > a)
        """
        ...

    def __lt__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise less-than comparison (no gradients).

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            A float32 tensor with 1.0 where `self < other`, else 0.0.

        Notes
        -----
        Comparison operations do not participate in autograd in this minimal
        implementation (the result always has `requires_grad=False`).

        Implementation
        --------------
        Uses only `gt`:
            a < b  <=>  b > a
        """
        ...

    def __le__(self, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise less-than-or-equal comparison (no gradients).

        Parameters
        ----------
        other : Union[Tensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        Tensor
            A float32 tensor with 1.0 where `self <= other`, else 0.0.

        Notes
        -----
        Comparison operations do not participate in autograd in this minimal
        implementation (the result always has `requires_grad=False`).

        Implementation
        --------------
        Uses only `gt` + `neg`:
            a <= b  <=>  not (a > b)
        """
        ...

    def to(self, device: "Device", *, copy: bool = True) -> "ITensor":
        """
        Return a tensor on the specified device.

        Parameters
        ----------
        device : Device
            Target device (e.g., Device("cpu"), Device("cuda:0")).
        copy : bool, optional
            If True (default), always returns a new tensor.
            If False and the device matches, may return self.

        Notes
        -----
        - This method performs a device transfer via NumPy as an intermediate
        representation when moving between CPU and CUDA.
        - Gradients are NOT transferred automatically.
        - Autograd history is NOT preserved across device moves.
        """
        ...

    def clone(self) -> "ITensor":
        """
        Create a deep copy of this tensor.

        The cloned tensor:
        - has the same shape, dtype, and device as `self`
        - owns independent storage (CPU or CUDA)
        - does NOT share autograd history (ctx is None)
        - does NOT propagate requires_grad (set to False)

        Returns
        -------
        ITensor
            A new tensor with copied data.
        """
        ...

    def _ensure_cuda_alloc(self, *, dtype) -> None:
        """
        Ensure that a CUDA tensor has an allocated device buffer.

        This method allocates device memory sized to
        `numel * dtype.itemsize` and stores the resulting device pointer on
        the tensor instance. Memory contents are left uninitialized.

        Notes
        -----
        - If the tensor already has an allocated CUDA buffer, this is a no-op
        (idempotent) unless the requested dtype would require a different
        allocation size.
        - If a re-allocation is required (dtype/size mismatch), the old buffer
        should be freed (TODO: implement/verify cuda_free usage).
        """
        ...

    @staticmethod
    def _get_cuda_lib():
        """
        Lazily load and cache the KeyDNN native CUDA shared library.

        This method performs a local import to avoid importing CUDA-related
        symbols during CPU-only execution. The loaded library handle is cached
        as an attribute on the function object, acting as a simple singleton.

        Returns
        -------
        object
            Handle to the loaded KeyDNN CUDA native library.

        Notes
        -----
        - The library is loaded at most once per process.
        - This method does not perform device selection; callers are responsible
        for invoking `cuda_set_device` as needed.
        """
        ...

    def _numel_from_shape(self) -> int:
        """
        Compute the total number of elements implied by the tensor shape.

        This method multiplies all dimensions in `self.shape` to obtain the
        total element count. Dimensions are explicitly cast to `int` to avoid
        propagation of NumPy scalar types.

        Returns
        -------
        int
            Total number of elements represented by the tensor shape.

        Notes
        -----
        - An empty shape (e.g., `()`) yields 1 by convention.
        - This helper is used by both CPU and CUDA allocation paths.
        """
        ...

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _raise_device_not_supported(self, op: str) -> None:
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
        ...

    @staticmethod
    def _result_requires_grad(*parents: "ITensor") -> bool:
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
        ...

    @staticmethod
    def _as_tensor_like(x: Union["ITensor", Number], like: "ITensor") -> "ITensor":
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
        ...

    @staticmethod
    def _binary_op_shape_check(a: "ITensor", b: "ITensor") -> None:
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
        ...

    def __imul__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise multiplication: self *= other.

        CUDA behavior
        -------------
        - If self is CUDA AND it is safe to mutate storage (no autograd tracking),
        dispatch to native in-place CUDA kernels to avoid intermediate allocations.
        - Otherwise, fall back to out-of-place multiply + copy_from (same semantics
        pattern as __isub__), which is graph-safer.

        CPU behavior
        ------------
        - Uses out-of-place multiply + copy_from (keeps behavior consistent and avoids
        relying on internal numpy storage layout).
        """
        ...

    def __iadd__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise addition: self += other.

        CUDA fast-path:
        - If safe to mutate (no autograd tracking), dispatch to native CUDA in-place
          kernels to avoid intermediate allocations.

        Fallback:
        - out-of-place add + copy_from.
        """
        ...

    def __isub__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise subtraction: self -= other.

        Same in-place safety rules as __iadd__/__imul__.
        """
        ...

    def __itruediv__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise true division: self /= other.

        Same in-place safety rules as __iadd__/__imul__.
        """
        ...

    # Python 2 legacy alias (optional, but nice to keep symmetry with __div__)
    def __idiv__(self: ITensor, other: Union["ITensor", Number]) -> ITensor: ...
