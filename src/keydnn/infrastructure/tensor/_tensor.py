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

from typing import Any, Union, Optional, Sequence, List, Tuple, Dict
from functools import lru_cache
from dataclasses import dataclass, field


import numpy as np

from ...domain._tensor import ITensor
from ...domain.device._device import Device
from ...domain._errors import DeviceNotSupportedError
from ._tensor_context import Context
from ._cuda_storage import _CudaStorage

Number = Union[int, float]


@dataclass
class _BWSectionStats:
    cpu_ms: float = 0.0
    gpu_ms: float = 0.0
    count: int = 0


@dataclass
class BackwardProfile:
    sections: Dict[str, _BWSectionStats] = field(default_factory=dict)
    per_op: Dict[str, _BWSectionStats] = field(default_factory=dict)
    per_node: List[Tuple[str, float, float]] = field(
        default_factory=list
    )  # (op_name, cpu_ms, gpu_ms)

    def add(self, name: str, cpu_ms: float, gpu_ms: float) -> None:
        s = self.sections.get(name)
        if s is None:
            s = _BWSectionStats()
            self.sections[name] = s
        s.cpu_ms += cpu_ms
        s.gpu_ms += gpu_ms
        s.count += 1

    def add_op(self, op: str, cpu_ms: float, gpu_ms: float) -> None:
        s = self.per_op.get(op)
        if s is None:
            s = _BWSectionStats()
            self.per_op[op] = s
        s.cpu_ms += cpu_ms
        s.gpu_ms += gpu_ms
        s.count += 1

    def report(self, topk: int = 20) -> str:
        lines: List[str] = []
        total_cpu = sum(s.cpu_ms for s in self.sections.values())
        total_gpu = sum(s.gpu_ms for s in self.sections.values())

        lines.append("==== Tensor.backward() CUDA profile ====")
        lines.append(
            f"Total CPU wall: {total_cpu:.3f} ms | Total GPU (events): {total_gpu:.3f} ms"
        )
        lines.append("")
        lines.append("[Sections]")
        for k, s in sorted(
            self.sections.items(), key=lambda kv: kv[1].gpu_ms, reverse=True
        ):
            cpu_pct = (s.cpu_ms / total_cpu * 100.0) if total_cpu > 0 else 0.0
            gpu_pct = (s.gpu_ms / total_gpu * 100.0) if total_gpu > 0 else 0.0
            lines.append(
                f"- {k:24s} cpu={s.cpu_ms:9.3f} ms ({cpu_pct:5.1f}%) "
                f"gpu={s.gpu_ms:9.3f} ms ({gpu_pct:5.1f}%) "
                f"n={s.count}"
            )

        if self.per_op:
            lines.append("")
            lines.append(f"[Per-op top {topk} by GPU]")
            for op, s in sorted(
                self.per_op.items(), key=lambda kv: kv[1].gpu_ms, reverse=True
            )[:topk]:
                lines.append(
                    f"- {op:24s} cpu={s.cpu_ms:9.3f} ms  gpu={s.gpu_ms:9.3f} ms  n={s.count}"
                )

        return "\n".join(lines)


class _CudaEventTimer:
    """
    Minimal CUDA event timing helper using your existing ctypes wrappers.

    Requires maxpool2d_ctypes to expose:
      - cuda_event_create, cuda_event_record, cuda_event_elapsed_ms, cuda_event_destroy
      - cuda_synchronize or cudaDeviceSynchronize equivalent
    If you don't have these yet, see the fallback note below.
    """

    def __init__(self, lib, m):
        self.lib = lib
        self.m = m

    def time_block(self, fn) -> float:
        # returns gpu_ms
        start = self.m.cuda_event_create(self.lib)
        end = self.m.cuda_event_create(self.lib)
        try:
            self.m.cuda_event_record(self.lib, start)
            fn()
            self.m.cuda_event_record(self.lib, end)
            self.m.cuda_event_synchronize(self.lib, end)
            ms = float(self.m.cuda_event_elapsed_ms(self.lib, start, end))
            return ms
        finally:
            try:
                self.m.cuda_event_destroy(self.lib, start)
            except Exception:
                pass
            try:
                self.m.cuda_event_destroy(self.lib, end)
            except Exception:
                pass


from .mixins.unary import TensorMixinUnary
from .mixins.memory import TensorMixinMemory
from .mixins.reduction import TensorMixinReduction
from .mixins.comparison import TensorMixinComparison
from .mixins.arithmetic import TensorMixinArithmetic


class Tensor(
    TensorMixinUnary,
    TensorMixinMemory,
    TensorMixinReduction,
    TensorMixinComparison,
    TensorMixinArithmetic,
    ITensor,
):
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
    dtype : np.dtype, optional
        Element dtype for this tensor. Defaults to np.float32.

    Notes
    -----
    - For CPU tensors, `_data` is a NumPy ndarray of dtype `self._dtype`.
    - For CUDA tensors, `_data` is a device pointer handle (DevPtr) stored as an int.
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
            self._data = np.zeros(self._shape, dtype=self._dtype)
            return

        is_cuda = getattr(d, "is_cuda", None)
        if callable(is_cuda) and is_cuda():
            # Device pointer handle (DevPtr). 0 means "not allocated / not set".
            self._data = 0
            return

        raise ValueError(f"Unsupported device type: {type(d)!r} value={d!r}")

    @property
    def dtype(self) -> np.dtype:
        """
        Return the element dtype of this tensor.

        Returns
        -------
        np.dtype
            NumPy dtype representing the tensor element type.

        Notes
        -----
        - For CPU tensors, this matches the underlying ndarray dtype.
        - For CUDA tensors, this is metadata used for kernel dispatch and tests.
        """
        return self._dtype

    @property
    def data(self) -> int | np.ndarray:
        """
        Return the underlying data handle for this tensor.

        Semantics
        ---------
        - **CPU tensors**:
            Returns the underlying NumPy ndarray storing the tensor data.
        - **CUDA tensors**:
            Returns the raw device pointer (`dev_ptr`) as an integer.

            The pointer is resolved as follows:
            1) If the tensor is backed by a `_CudaStorage` object, return
            `storage.dev_ptr`.
            2) Otherwise, fall back to the legacy `_data` field, which may
            contain a raw device pointer set by older code paths or tests.

        Notes
        -----
        - For CUDA tensors, the returned value is **not** a NumPy array and
        should be treated as an opaque device pointer handle.
        - A return value of `0` indicates that no device memory is currently
        allocated (e.g., uninitialized tensor, freed tensor, or zero-sized
        tensor).
        - New code should prefer storage-backed tensors; the `_data` fallback
        exists only for backward compatibility during migration.

        Returns
        -------
        int | np.ndarray
            - NumPy ndarray for CPU tensors.
            - Integer device pointer (`uintptr_t`) for CUDA tensors.

        Raises
        ------
        ValueError
            If the tensor is on an unsupported or unknown device.
        """

        d = self._device
        if d.is_cpu():
            return self._data

        if d.is_cuda():
            st = getattr(self, "_storage", None)
            if st is not None:
                return int(st.dev_ptr)
            # fallback for older code/tests that set _data directly
            return int(getattr(self, "_data", 0) or 0)

        raise ValueError(...)

    @classmethod
    def _from_storage(
        cls,
        storage,
        *,
        shape: tuple[int, ...],
        device: Device,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
        dtype: np.dtype = np.float32,
    ) -> "Tensor":
        """
        Construct a CUDA tensor backed by an existing `_CudaStorage` object.

        This constructor is the **preferred** way to create CUDA tensors that
        *own* their device memory. The resulting tensor participates fully in
        storage reference counting and lifetime management.

        Ownership semantics
        -------------------
        - The provided `storage` is assumed to represent allocated CUDA memory.
        - This method increments the storage reference count (`storage.incref()`),
        indicating that the returned tensor holds a strong reference.
        - When all tensors referencing the same storage have released it
        (`decref()`), the underlying device memory is freed.

        Parameters
        ----------
        storage : _CudaStorage
            Storage object managing a CUDA device allocation.
        shape : tuple[int, ...]
            Logical tensor shape associated with the storage.
        device : Device
            CUDA device descriptor. Must satisfy `device.is_cuda() == True`.
        requires_grad : bool, optional
            Whether this tensor should accumulate gradients during backpropagation.
            Defaults to False.
        ctx : Optional[Context], optional
            Optional autograd context to attach to the tensor.
        dtype : np.dtype, optional
            Element dtype metadata for kernel dispatch. Defaults to `np.float32`.

        Returns
        -------
        Tensor
            A CUDA tensor owning a reference to the provided storage.

        Raises
        ------
        ValueError
            If `storage` is None. Use `_from_devptr` for borrowed/raw device pointers.

        Notes
        -----
        - This method bypasses `__init__` and does **not** allocate new memory.
        - A legacy mirror of the device pointer is stored in `_data` for
        debugging and backward compatibility only.
        - Tensors created by this method are **not borrowed**; they participate
        in storage lifetime management.
        """
        if storage is None:
            raise ValueError(
                "_from_storage(storage=None): use _from_devptr for borrowed pointers"
            )
        obj = cls.__new__(cls)
        obj._shape = tuple(shape)
        obj._device = device
        obj._dtype = np.dtype(dtype)

        obj._storage = storage
        storage.incref()
        obj._borrowed_devptr = False

        # keep legacy pointer mirror if you want (optional, helps debugging)
        obj._data = int(storage.dev_ptr)

        obj._requires_grad = bool(requires_grad)
        obj._grad = None
        obj._ctx = ctx
        return obj

    @classmethod
    def _from_devptr(
        cls,
        dev_ptr: int,
        *,
        shape: tuple[int, ...],
        device: Device,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
        dtype: np.dtype = np.float32,
    ) -> "Tensor":
        """
        Construct a CUDA tensor backed by a borrowed raw device pointer.

        This constructor wraps an existing CUDA device pointer **without**
        taking ownership of the underlying memory. It is intended for
        interoperability with external CUDA code or legacy kernels that
        return raw `dev_ptr` handles.

        Ownership semantics
        -------------------
        - The returned tensor does **not** own the device memory.
        - No `_CudaStorage` object is created.
        - Calling `free_()` on the resulting tensor will **not** free the
        underlying device pointer.
        - The caller is responsible for managing the lifetime of `dev_ptr`.

        Parameters
        ----------
        dev_ptr : int
            Raw CUDA device pointer (`uintptr_t`) referring to already-allocated
            device memory.
        shape : tuple[int, ...]
            Logical tensor shape associated with the device pointer.
        device : Device
            CUDA device descriptor. Must satisfy `device.is_cuda() == True`.
        requires_grad : bool, optional
            Whether this tensor should accumulate gradients during backpropagation.
            Defaults to False.
        ctx : Optional[Context], optional
            Optional autograd context to attach to the tensor.
        dtype : np.dtype, optional
            Element dtype metadata for kernel dispatch. Defaults to `np.float32`.

        Returns
        -------
        Tensor
            A CUDA tensor that *borrows* the provided device pointer.

        Raises
        ------
        ValueError
            If `device` is not a CUDA device.
        ValueError
            If `dev_ptr` is None.

        Notes
        -----
        - This method bypasses `__init__` and performs no allocation.
        - The `_borrowed_devptr` flag is set to True to prevent accidental frees.
        - This constructor exists primarily as a **transitional API** while
        migrating legacy devptr-based code to storage-backed tensors.
        - New internal code should prefer `_from_storage` whenever possible.
        """
        is_cuda = getattr(device, "is_cuda", None)
        if not (callable(is_cuda) and is_cuda()):
            raise ValueError(
                f"_from_devptr requires a CUDA device; got device={device!r}"
            )

        if dev_ptr is None:
            raise ValueError("dev_ptr must be an int (uintptr_t), got None")

        dp = int(dev_ptr)

        obj = cls.__new__(cls)  # bypass __init__
        obj._shape = tuple(shape)
        obj._device = device

        # IMPORTANT: keep legacy pointer field for borrowed semantics
        obj._data = dp

        obj._dtype = np.dtype(dtype)

        # IMPORTANT: establish invariant even when bypassing __init__
        # (Borrowed pointer => no storage ownership by default)
        obj._storage = None  # type: ignore[attr-defined]

        # --- autograd fields (optional) ---
        obj._requires_grad = bool(requires_grad)
        obj._grad = None
        obj._ctx = ctx

        # borrowed only
        obj._storage = None
        obj._borrowed_devptr = True

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
            return (
                f"Tensor(shape={self._shape}, device={d}, dtype={self._dtype}, "
                f"data=DevPtr({int(self.data)}))"
            )
        return f"Tensor(shape={self._shape}, device={d}, dtype={self.data.dtype})"

    def __init__(
        self,
        shape: tuple[int, ...],
        device: Device,
        *,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
        dtype: np.dtype = np.float32,
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
        dtype : np.dtype, optional
            Element dtype for this tensor. Defaults to np.float32.
        """
        self._shape = shape
        self._device = device
        self._dtype = np.dtype(dtype)
        self.__initialize_data()
        self._storage: Optional[_CudaStorage] = None

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

            if self._storage is not None:
                return (
                    f"CUDA Tensor on device {dev_index} with shape {self._shape} "
                    f"(storage={self._storage})"
                )

            # If CUDA storage is already a string placeholder, keep it.
            # if isinstance(self.data, str):
            #     return self.data

            # If CUDA storage is a devptr handle (int), format a stable message.
            if isinstance(self.data, int):
                return (
                    f"CUDA Tensor on device {dev_index} with shape {self._shape} "
                    f"(devptr={self.data})"
                )

            # Fallback if something unexpected is stored
            return f"CUDA Tensor on device {dev_index} with shape {self._shape}"

        return f"Unknown device storage shape={self._shape}"

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

    @staticmethod
    def stack(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """
        Stack a sequence of tensors along a new axis.

        This is the differentiable counterpart of `np.stack`.

        Requirements
        ------------
        - `tensors` must be non-empty
        - all tensors must share the same shape
        - all tensors must share the same device
        - CPU: forward/backward uses NumPy (backward compatible)
        - CUDA: forward/backward uses `stack_cuda_ext` kernels (device-pointer based)
        and does NOT call `to_numpy()` on CUDA tensors.

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
        - CUDA backward overwrites dx buffers (no accumulation inside kernel).
        """
        import numpy as np

        if len(tensors) == 0:
            raise ValueError("Tensor.stack() requires a non-empty sequence")

        first = tensors[0]
        dev = first.device
        in_shape = tuple(first.shape)

        # Normalize axis against input ndim (same logic as before)
        ndim = len(in_shape)
        if axis < 0:
            axis = axis + (ndim + 1)
        if axis < 0 or axis > ndim:
            raise ValueError(
                f"axis {axis} out of bounds for stack with input ndim {ndim}"
            )
        axis_n = int(axis)

        # Validate all tensors: same device + same shape
        for i, t in enumerate(tensors):
            if str(t.device) != str(dev):
                raise ValueError(
                    f"Tensor.stack() requires all tensors on the same device; "
                    f"tensors[0] is {dev!r} but tensors[{i}] is {t.device!r}"
                )
            if tuple(t.shape) != in_shape:
                raise ValueError(
                    f"Tensor.stack() requires all tensors to have the same shape; "
                    f"expected {in_shape}, got {tuple(t.shape)} at index {i}"
                )

        req = any(t.requires_grad for t in tensors)
        K = int(len(tensors))
        out_shape = tuple(in_shape[:axis_n]) + (K,) + tuple(in_shape[axis_n:])

        # ------------------------------------------------------------------
        # CUDA path (new)
        # ------------------------------------------------------------------
        if dev.is_cuda():
            import numpy as np
            from ..ops.stack_cuda_ext import (
                stack_forward as _stack_fwd_cuda,
                stack_backward as _stack_bwd_cuda,
            )

            # Enforce dtype support for kernels (module checks too, but keep errors local)
            dt = np.dtype(first.dtype)
            if dt not in (np.float32, np.float64):
                raise TypeError(
                    f"Tensor.stack CUDA supports float32/float64 only; got {dt}"
                )

            # IMPORTANT: do not silently allocate inputs; require allocated buffers
            for i, t in enumerate(tensors):
                if int(t.data) == 0:
                    raise RuntimeError(
                        f"Tensor.stack CUDA requires allocated device buffers (data != 0); "
                        f"tensors[{i}].data == 0"
                    )
                if np.dtype(t.dtype) != dt:
                    raise TypeError(
                        f"Tensor.stack CUDA requires same dtype for all tensors; "
                        f"expected {dt}, got {np.dtype(t.dtype)} at index {i}"
                    )

            # Forward (CUDA): returns a CUDA Tensor already backed by devptr
            out = _stack_fwd_cuda(
                tensors,
                axis=axis_n,
                device=int(getattr(dev, "index", 0) or 0),
            )
            # preserve autograd flags
            out._requires_grad = bool(req)

            if req:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA Tensor.stack backward"
                        )
                    if str(grad_out.device) != str(dev):
                        raise RuntimeError(
                            f"grad_out must be on the same CUDA device as output; "
                            f"got {grad_out.device} vs {dev}"
                        )
                    if tuple(grad_out.shape) != out_shape:
                        raise ValueError(
                            f"grad_out shape mismatch: expected {out_shape}, got {tuple(grad_out.shape)}"
                        )

                    # Compute CUDA grads for all K, then mask by requires_grad
                    grads_all = _stack_bwd_cuda(
                        grad_out,
                        x_shape=in_shape,
                        axis=axis_n,
                        K=K,
                        device=int(getattr(dev, "index", 0) or 0),
                    )

                    grads: list[Optional["Tensor"]] = []
                    for i, t in enumerate(tensors):
                        if not t.requires_grad:
                            grads.append(None)
                            continue
                        grads.append(grads_all[i])
                    return tuple(grads)

                ctx = Context(parents=tuple(tensors), backward_fn=backward_fn)
                ctx.saved_meta["stack_axis"] = axis_n
                ctx.saved_meta["stack_K"] = K
                ctx.saved_meta["stack_in_shape"] = in_shape
                out._set_ctx(ctx)

            return out

        # ------------------------------------------------------------------
        # CPU path (unchanged behavior)
        # ------------------------------------------------------------------
        if not dev.is_cpu():
            first._raise_device_not_supported("stack")

        arrs = [t.to_numpy() for t in tensors]
        stacked = np.stack(arrs, axis=axis_n).astype(np.float32, copy=False)

        out = Tensor(shape=stacked.shape, device=dev, requires_grad=req, ctx=None)
        out.copy_from_numpy(stacked)

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

                    gi_np = np.take(g, indices=i, axis=axis_n).astype(
                        np.float32, copy=False
                    )

                    gi = Tensor(
                        shape=t.shape, device=dev, requires_grad=False, ctx=None
                    )
                    gi.copy_from_numpy(gi_np)
                    grads.append(gi)

                return tuple(grads)

            ctx = Context(parents=tuple(tensors), backward_fn=backward_fn)
            out._set_ctx(ctx)

        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication (2D): out = self @ other.

        Requirements
        ------------
        - CPU: unchanged behavior (backward compatible)
        - CUDA: both operands must be CUDA, 2D, inner dims match
        - CUDA inputs MUST already have allocated device buffers (data != 0).
            (No implicit allocation for inputs.)

        Backward
        --------
        If out = A @ B, then:
        - dL/dA = dL/dout @ B^T
        - dL/dB = A^T @ dL/dout
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"matmul expects Tensor, got {type(other)!r}")

        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(
                f"matmul requires 2D tensors, got {self.shape} and {other.shape}"
            )

        n, k1 = self.shape
        k2, m = other.shape
        if int(k1) != int(k2):
            raise ValueError(
                f"matmul shape mismatch: {self.shape} @ {other.shape} (inner dims {k1} vs {k2})"
            )

        req = self._result_requires_grad(self, other)

        # ------------------------------------------------------------------
        # CUDA path
        # ------------------------------------------------------------------
        if self.device.is_cuda() or other.device.is_cuda():

            # For now, require both on CUDA (no mixed-device implicit copies)
            if not (self.device.is_cuda() and other.device.is_cuda()):
                raise RuntimeError(
                    f"matmul requires both tensors on the same device; got {self.device} and {other.device}"
                )

            # Enforce same CUDA device index if your Device encodes it
            if str(self.device) != str(other.device):
                raise RuntimeError(
                    f"matmul requires both tensors on the same CUDA device; got {self.device} and {other.device}"
                )

            import ctypes  # local import so CPU-only runs don't pay cost
            import numpy as np
            from ..ops.matmul_cuda import matmul2d_cuda  # ops-layer opt ext wrapper
            from ..ops.transpose_cuda import transpose2d_cuda  # used in backward

            def _cuda_device_index(dev: object) -> int:
                # Prefer explicit attribute if you really have it
                idx = getattr(dev, "index", None)
                if idx is not None:
                    try:
                        return int(idx)
                    except Exception:
                        pass

                # Fallback: parse from string like "cuda:0"
                s = str(dev)
                if "cuda" not in s:
                    return 0
                if ":" in s:
                    tail = s.split(":", 1)[1].strip()
                    try:
                        return int(tail)
                    except Exception:
                        return 0
                return 0

            device_index = _cuda_device_index(self.device)

            dtype = np.dtype(self.dtype)
            if np.dtype(other.dtype) != dtype:
                raise TypeError(f"matmul dtype mismatch: {self.dtype} vs {other.dtype}")

            # IMPORTANT: do NOT implicitly allocate inputs.
            a_dev = int(self.data)
            b_dev = int(other.data)
            if a_dev == 0 or b_dev == 0:
                raise RuntimeError(
                    "CUDA matmul requires allocated device buffers for both inputs (data != 0)"
                )

            lib = self._get_cuda_lib()

            # --------------------------------------------------------------
            # CRITICAL: set device BEFORE any allocations/syncs in this method
            # --------------------------------------------------------------
            if hasattr(lib, "keydnn_cuda_set_device"):
                fn = lib.keydnn_cuda_set_device
                fn.argtypes = [ctypes.c_int]
                fn.restype = ctypes.c_int
                st = int(fn(int(device_index)))
                if st != 0:
                    raise RuntimeError(
                        f"cuda_set_device({int(device_index)}) failed: status={st}"
                    )

            def _cuda_sync() -> None:
                # Best-effort synchronize to avoid async hazards (Windows-friendly).
                if hasattr(lib, "keydnn_cuda_synchronize"):
                    fn = lib.keydnn_cuda_synchronize
                    fn.argtypes = []
                    fn.restype = ctypes.c_int
                    st = int(fn())
                    if st != 0:
                        raise RuntimeError(f"cuda_synchronize failed: status={st}")

            # Ensure any prior H2D copies / kernels that produced inputs are complete.
            _cuda_sync()

            # Allocate output (now guaranteed on correct device)
            out = Tensor(
                shape=(int(n), int(m)),
                device=self.device,
                requires_grad=req,
                ctx=None,
                dtype=dtype,
            )
            out._ensure_cuda_alloc(dtype=dtype)
            y_dev = int(out.data)
            if y_dev == 0:
                raise RuntimeError("CUDA matmul failed to allocate output buffer")

            # Forward: C = A @ B
            matmul2d_cuda(
                lib,
                a_dev=int(a_dev),
                b_dev=int(b_dev),
                c_dev=int(y_dev),
                n=int(n),  # rows of A
                k=int(k1),  # inner dim
                m=int(m),  # cols of B
                dtype=dtype,
                sync=True,
                device_index=int(device_index),
            )

            if req:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA matmul backward"
                        )
                    if str(grad_out.device) != str(self.device):
                        raise RuntimeError(
                            f"grad_out must be on the same CUDA device as output; got {grad_out.device} vs {self.device}"
                        )
                    if grad_out.shape != (int(n), int(m)):
                        raise ValueError(
                            f"grad_out shape mismatch: expected {(int(n), int(m))}, got {grad_out.shape}"
                        )

                    go_dev = int(grad_out.data)
                    if go_dev == 0:
                        raise RuntimeError(
                            "grad_out CUDA tensor has no allocated devptr (data == 0)"
                        )

                    # Ensure correct device before allocating temporaries in backward too.
                    if hasattr(lib, "keydnn_cuda_set_device"):
                        fn = lib.keydnn_cuda_set_device
                        fn.argtypes = [ctypes.c_int]
                        fn.restype = ctypes.c_int
                        st = int(fn(int(device_index)))
                        if st != 0:
                            raise RuntimeError(
                                f"cuda_set_device({int(device_index)}) failed: status={st}"
                            )

                    grad_a = None
                    grad_b = None

                    _cuda_sync()

                    # ---- dA path ----
                    if self.requires_grad:
                        bt = Tensor(
                            shape=(int(m), int(k1)),
                            device=self.device,
                            requires_grad=False,
                            ctx=None,
                            dtype=dtype,
                        )
                        bt._ensure_cuda_alloc(dtype=dtype)
                        bt_dev = int(bt.data)
                        if bt_dev == 0:
                            raise RuntimeError(
                                "CUDA matmul backward failed to allocate Bt buffer"
                            )

                        transpose2d_cuda(
                            lib,
                            x_dev=int(b_dev),
                            y_dev=int(bt_dev),
                            rows=int(k1),
                            cols=int(m),
                            dtype=dtype,
                            sync=True,
                        )

                        grad_a = Tensor(
                            shape=(int(n), int(k1)),
                            device=self.device,
                            requires_grad=False,
                            ctx=None,
                            dtype=dtype,
                        )
                        grad_a._ensure_cuda_alloc(dtype=dtype)
                        ga_dev = int(grad_a.data)
                        if ga_dev == 0:
                            raise RuntimeError(
                                "CUDA matmul backward failed to allocate grad_a buffer"
                            )

                        matmul2d_cuda(
                            lib,
                            a_dev=int(go_dev),
                            b_dev=int(bt_dev),
                            c_dev=int(ga_dev),
                            n=int(n),
                            k=int(m),
                            m=int(k1),
                            dtype=dtype,
                            sync=True,
                            device_index=int(device_index),
                        )

                    # ---- dB path ----
                    if other.requires_grad:
                        at = Tensor(
                            shape=(int(k1), int(n)),
                            device=self.device,
                            requires_grad=False,
                            ctx=None,
                            dtype=dtype,
                        )
                        at._ensure_cuda_alloc(dtype=dtype)
                        at_dev = int(at.data)
                        if at_dev == 0:
                            raise RuntimeError(
                                "CUDA matmul backward failed to allocate At buffer"
                            )

                        transpose2d_cuda(
                            lib,
                            x_dev=int(a_dev),
                            y_dev=int(at_dev),
                            rows=int(n),
                            cols=int(k1),
                            dtype=dtype,
                            sync=True,
                        )

                        grad_b = Tensor(
                            shape=(int(k1), int(m)),
                            device=self.device,
                            requires_grad=False,
                            ctx=None,
                            dtype=dtype,
                        )
                        grad_b._ensure_cuda_alloc(dtype=dtype)
                        gb_dev = int(grad_b.data)
                        if gb_dev == 0:
                            raise RuntimeError(
                                "CUDA matmul backward failed to allocate grad_b buffer"
                            )

                        matmul2d_cuda(
                            lib,
                            a_dev=int(at_dev),
                            b_dev=int(go_dev),
                            c_dev=int(gb_dev),
                            n=int(k1),
                            k=int(n),
                            m=int(m),
                            dtype=dtype,
                            sync=True,
                            device_index=int(device_index),
                        )

                    return (grad_a, grad_b)

                ctx = Context(parents=(self, other), backward_fn=backward_fn)
                out._set_ctx(ctx)

            return out

        # ------------------------------------------------------------------
        # CPU path (UNCHANGED)
        # ------------------------------------------------------------------
        if not self.device.is_cpu() or not other.device.is_cpu():
            self._raise_device_not_supported("matmul")

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

                if self.requires_grad:
                    ga_np = grad_out.to_numpy() @ other.to_numpy().T
                    grad_a = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                    )
                    grad_a.copy_from_numpy(ga_np)

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

            ctx = Context(parents=(self, other), backward_fn=backward_fn)
            out._set_ctx(ctx)

        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Operator overload for matrix multiplication: self @ other.
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"@ only supports Tensor operands, got {type(other)!r}")
        return self.matmul(other)

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

    def backward(
        self,
        grad_out: Optional["Tensor"] = None,
        *,
        profile: bool = False,
        profile_topk: int = 20,
    ) -> None:
        """
        Backpropagate gradients from this tensor through the autograd graph.

        Parameters
        ----------
        grad_out : Optional[Tensor], optional
            Gradient w.r.t. this tensor. If omitted, this tensor must be a scalar
            (shape == ()) and the gradient is assumed to be 1.0.

        Notes
        -----
        - Gradients are accumulated into `.grad` of leaf tensors that have
        `requires_grad=True`.
        - This implementation performs a reverse topological traversal.
        - CPU behavior is unchanged (same logic; only extra timing when profile=True).
        """
        import time
        import numpy as np

        prof = BackwardProfile() if profile else None

        # ---------------------------------------------------------------------
        # CPU path (profile-able)
        # ---------------------------------------------------------------------
        if self.device.is_cpu():

            def _time_section_cpu(name: str, fn):
                if prof is None:
                    fn()
                    return
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                cpu_ms = (t1 - t0) * 1000.0
                gpu_ms = cpu_ms  # CPU: keep same field for report compatibility
                prof.add(name, cpu_ms=cpu_ms, gpu_ms=gpu_ms)

            # Seed gradient
            def _seed_grad_cpu():
                nonlocal grad_out
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
                        raise TypeError(
                            f"grad_out must be a Tensor, got {type(grad_out)!r}"
                        )
                    if grad_out.shape != self.shape:
                        raise ValueError(
                            f"grad_out shape mismatch: expected {self.shape}, got {grad_out.shape}"
                        )
                    if grad_out.device != self.device:
                        raise ValueError("grad_out must be on the same device as self")

            _time_section_cpu("seed_grad", _seed_grad_cpu)

            # Build topo
            topo: list[Tensor] = []
            visited: set[int] = set()

            def _build_topo_cpu():
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

            _time_section_cpu("build_topo_dfs", _build_topo_cpu)

            nodes: dict[int, Tensor] = {}
            _time_section_cpu(
                "build_nodes_dict", lambda: nodes.update({id(t): t for t in topo})
            )

            grads: dict[int, Tensor] = {id(self): grad_out}  # type: ignore[arg-type]

            # Reverse traversal
            def _reverse_traverse_cpu():
                for t in reversed(topo):
                    ctx = t._get_ctx()
                    if ctx is None:
                        continue

                    grad_t = grads.get(id(t))
                    if grad_t is None:
                        continue

                    op_name = (
                        getattr(ctx, "op_name", None)
                        or getattr(ctx, "name", None)
                        or type(ctx).__name__
                    )

                    # time backward_fn
                    if prof is None:
                        parent_grads = ctx.backward_fn(grad_t)
                    else:
                        t0 = time.perf_counter()
                        parent_grads = ctx.backward_fn(grad_t)
                        t1 = time.perf_counter()
                        cpu_ms = (t1 - t0) * 1000.0
                        prof.add("backward_fn_total", cpu_ms, cpu_ms)
                        prof.add_op(op_name, cpu_ms, cpu_ms)

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

                        pid = id(parent)
                        if pid in grads:
                            _time_section_cpu(
                                "grad_accum_add",
                                lambda pid=pid, g=g: grads.__setitem__(
                                    pid, self._add_no_grad(grads[pid], g)
                                ),
                            )
                        else:
                            _time_section_cpu(
                                "grad_accum_set",
                                lambda pid=pid, g=g: grads.__setitem__(
                                    pid, self._detach_no_grad(g)
                                ),
                            )

            _time_section_cpu("reverse_traverse_total", _reverse_traverse_cpu)

            # Writeback
            def _writeback_cpu():
                for tid, g in grads.items():
                    t = nodes.get(tid)
                    if t is None or not t.requires_grad:
                        continue
                    t._accumulate_grad_(g)

            _time_section_cpu("writeback_leaf_accum", _writeback_cpu)

            if prof is not None:
                print(prof.report(topk=profile_topk))
            return

        # ---------------------------------------------------------------------
        # CUDA path (your existing profiler)
        # ---------------------------------------------------------------------
        if not self.device.is_cuda():
            self._raise_device_not_supported("backward")

        # share single lib handle
        def _get_cuda_lib():
            lib = getattr(Tensor, "_CUDA_LIB", None)
            if lib is None:
                from ...infrastructure.native_cuda.python import maxpool2d_ctypes as m

                lib = m.load_keydnn_cuda_native()
                setattr(Tensor, "_CUDA_LIB", lib)
            return lib

        from ...infrastructure.native_cuda.python import maxpool2d_ctypes as m

        lib = _get_cuda_lib()

        # ---- choose timing backend ----
        have_events = all(
            hasattr(m, name)
            for name in (
                "cuda_event_create",
                "cuda_event_record",
                "cuda_event_synchronize",
                "cuda_event_elapsed_ms",
                "cuda_event_destroy",
            )
        )
        event_timer = _CudaEventTimer(lib, m) if have_events else None

        def _sync():
            if hasattr(m, "cuda_synchronize"):
                m.cuda_synchronize(lib)
            elif hasattr(m, "cudaDeviceSynchronize"):
                m.cudaDeviceSynchronize(lib)
            else:
                try:
                    from ...infrastructure.native_cuda.python.global_avgpool2d_ctypes import (
                        cuda_synchronize,
                    )

                    cuda_synchronize(lib)
                except Exception:
                    pass

        def _time_section(name: str, fn):
            if prof is None:
                fn()
                return

            t0 = time.perf_counter()
            if event_timer is not None:
                gpu_ms = event_timer.time_block(fn)
                t1 = time.perf_counter()
                cpu_ms = (t1 - t0) * 1000.0
            else:
                _sync()
                t0b = time.perf_counter()
                fn()
                _sync()
                t1b = time.perf_counter()
                cpu_ms = (t1b - t0b) * 1000.0
                gpu_ms = cpu_ms

            prof.add(name, cpu_ms=cpu_ms, gpu_ms=gpu_ms)

        # -----------------------------
        # Seed gradient (CUDA)
        # -----------------------------
        def _seed_grad():
            nonlocal grad_out
            if grad_out is None:
                if self.shape != ():
                    raise ValueError(
                        "grad_out must be provided for non-scalar tensors."
                    )
                host = np.array(1.0, dtype=np.float32)
                device_index = int(getattr(self.device, "index", 0) or 0)

                # Ensure correct device
                if hasattr(m, "cuda_set_device"):
                    m.cuda_set_device(lib, device_index)

                dev_ptr = int(m.cuda_malloc(lib, int(host.nbytes)))
                if dev_ptr == 0:
                    raise RuntimeError("cuda_malloc returned 0")

                try:
                    m.cudaMemcpyHtoD(lib, int(dev_ptr), host, int(host.nbytes))

                    storage = _CudaStorage(
                        lib=lib,
                        device_index=device_index,
                        dev_ptr=int(dev_ptr),
                        nbytes=int(host.nbytes),
                        dtype=np.float32,
                    )

                    grad_out = Tensor._from_storage(
                        storage=storage,
                        shape=(),
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                        dtype=np.float32,
                    )

                    # If _from_storage() does incref(), then you MUST drop local ref:
                    # storage.decref()

                except Exception:
                    try:
                        m.cuda_free(lib, int(dev_ptr))
                    except Exception:
                        pass
                    raise
            else:
                if not isinstance(grad_out, Tensor):
                    raise TypeError(
                        f"grad_out must be a Tensor, got {type(grad_out)!r}"
                    )
                if grad_out.shape != self.shape:
                    raise ValueError(
                        f"grad_out shape mismatch: expected {self.shape}, got {grad_out.shape}"
                    )
                if grad_out.device != self.device:
                    raise ValueError("grad_out must be on the same device as self")

        _time_section("seed_grad", _seed_grad)

        # -----------------------------
        # Build topo (DFS) + nodes dict
        # -----------------------------
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo():
            def dfs_cuda(t: "Tensor") -> None:
                tid = id(t)
                if tid in visited:
                    return
                visited.add(tid)
                ctx = t._get_ctx()
                if ctx is not None:
                    for p in ctx.parents:
                        dfs_cuda(p)
                topo.append(t)

            dfs_cuda(self)

        _time_section("build_topo_dfs", _build_topo)

        nodes: dict[int, Tensor] = {}
        _time_section(
            "build_nodes_dict", lambda: nodes.update({id(t): t for t in topo})
        )

        grads: dict[int, Tensor] = {id(self): grad_out}  # type: ignore[arg-type]

        # -----------------------------
        # Reverse traversal
        # -----------------------------
        def _reverse_traverse():
            for t in reversed(topo):
                ctx = t._get_ctx()
                if ctx is None:
                    continue

                grad_t = grads.get(id(t))
                if grad_t is None:
                    continue

                op_name = (
                    getattr(ctx, "op_name", None)
                    or getattr(ctx, "name", None)
                    or type(ctx).__name__
                )

                # ------------------------------------------------------------
                # Call backward_fn
                # - If not profiling: do NOT synchronize or create CUDA events.
                # - If profiling: time with CUDA events when available, else sync timing.
                # ------------------------------------------------------------
                if prof is None:
                    parent_grads = ctx.backward_fn(grad_t)
                else:
                    t0 = time.perf_counter()
                    if event_timer is not None:

                        def _call():
                            nonlocal parent_grads
                            parent_grads = ctx.backward_fn(grad_t)

                        parent_grads: list[Optional[Tensor]]
                        gpu_ms = event_timer.time_block(_call)
                        cpu_ms = (time.perf_counter() - t0) * 1000.0
                    else:
                        _sync()
                        t0b = time.perf_counter()
                        parent_grads = ctx.backward_fn(grad_t)
                        _sync()
                        cpu_ms = (time.perf_counter() - t0b) * 1000.0
                        gpu_ms = cpu_ms

                    prof.add("backward_fn_total", cpu_ms, gpu_ms)
                    prof.add_op(op_name, cpu_ms, gpu_ms)
                    prof.per_node.append((op_name, cpu_ms, gpu_ms))

                if __debug__ and len(parent_grads) != len(ctx.parents):
                    raise RuntimeError(
                        "backward_fn must return one grad per parent. "
                        f"Got {len(parent_grads)} grads for {len(ctx.parents)} parents."
                    )

                for parent, g in zip(ctx.parents, parent_grads):
                    if g is None:
                        continue
                    if __debug__ and not isinstance(g, Tensor):
                        raise TypeError(
                            f"backward_fn must return Tensor or None, got {type(g)!r}"
                        )
                    if __debug__ and g.device != parent.device:
                        raise ValueError("Gradient device must match parent device")
                    if __debug__ and g.shape != parent.shape:
                        raise ValueError(
                            f"Gradient shape mismatch for parent: expected {parent.shape}, got {g.shape}"
                        )

                    pid = id(parent)
                    if g.requires_grad or (g._get_ctx() is not None):
                        g = Tensor._cuda_detach_view_no_grad(g)

                    if pid in grads:

                        def _acc():
                            grads[pid] = Tensor._add_no_grad_cuda(grads[pid], g)

                        _time_section("grad_accum_add", _acc)
                    else:

                        def _set():
                            grads[pid] = Tensor._cuda_detach_view_no_grad(g)

                        _time_section("grad_accum_set", _set)

        _time_section("reverse_traverse_total", _reverse_traverse)

        # -----------------------------
        # Writeback into leaf .grad
        # -----------------------------
        def _writeback():
            for tid, g in grads.items():
                t = nodes.get(tid)
                if t is None or not t.requires_grad:
                    continue
                t._accumulate_grad_cuda_(g)

        _time_section("writeback_leaf_accum", _writeback)

        if prof is not None:
            print(prof.report(topk=profile_topk))

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
        n = 1
        for d in self.shape:
            n *= int(d)
        return int(n)

    @staticmethod
    @lru_cache(maxsize=1)
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
        # Local import to avoid importing CUDA on CPU-only runs
        from ..native_cuda.python.avgpool2d_ctypes import load_keydnn_cuda_native

        # cache on the function object (simple singleton)
        if not hasattr(Tensor._get_cuda_lib, "_lib"):
            Tensor._get_cuda_lib._lib = load_keydnn_cuda_native()
        return Tensor._get_cuda_lib._lib

    @staticmethod
    def rand(shape, *, device, requires_grad: bool = False) -> "Tensor":
        """
        Create a tensor filled with uniform random values in [0, 1) on the given device.

        Notes
        -----
        - CPU: random values are generated using NumPy.
        - CUDA: random values are generated on CPU and transferred to device memory.
        No CUDA RNG kernel is used.
        - This mirrors the initialization strategy used by many frameworks and keeps
        behavior deterministic and easy to test.
        - The returned tensor has dtype float32 and ctx=None.
        """
        import numpy as np

        # Generate on CPU using NumPy (single source of truth)
        arr = np.random.rand(*shape).astype(np.float32, copy=False)

        # CPU path (unchanged)
        if device.is_cpu():
            t = Tensor(
                shape=arr.shape,
                device=device,
                requires_grad=requires_grad,
                ctx=None,
                dtype=np.float32,
            )
            t.copy_from_numpy(arr)
            return t

        # CUDA path: CPU -> CUDA transfer
        if device.is_cuda():
            # Stage on CPU first
            t_cpu = Tensor(
                shape=arr.shape,
                device=Device("cpu"),
                requires_grad=False,
                ctx=None,
                dtype=np.float32,
            )
            t_cpu.copy_from_numpy(arr)

            # Transfer to CUDA using existing mechanism
            # (to() or copy_from with allow_cross_device)
            if hasattr(t_cpu, "to"):
                t_cuda = t_cpu.to(device)
            else:
                t_cuda = Tensor(
                    shape=arr.shape,
                    device=device,
                    requires_grad=False,
                    ctx=None,
                    dtype=np.float32,
                )
                t_cuda.copy_from(t_cpu, allow_cross_device=True)

            # requires_grad applies to the final tensor
            t_cuda.requires_grad = requires_grad
            return t_cuda

        raise RuntimeError(f"rand is not supported for device={device!r}")

    @staticmethod
    def concat(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """
        Concatenate a sequence of tensors along an existing axis.

        CPU behavior
        ------------
        - Fully supports all axes via NumPy.

        CUDA behavior (current)
        -----------------------
        - Supports only axis == 0 for CUDA tensors using device-to-device memcpy.
        - General axis concatenation on CUDA requires a kernel (pending).

        Requirements
        ------------
        - `tensors` must be non-empty
        - all tensors must share the same device
        - shapes must match on all dimensions except `axis`
        - dtypes must match (CUDA path enforces this; CPU path preserves current float32-cast behavior)

        Backward rule
        -------------
        - Split `grad_out` along `axis` into slices matching each input's size
        along that axis, and route each slice back to the corresponding parent.
        """
        if len(tensors) == 0:
            raise ValueError("Tensor.concat() requires a non-empty sequence")

        first = tensors[0]
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

        # Validate devices, ndim, and shapes (all dims except axis must match)
        sizes_along_axis: list[int] = []
        dtype0 = getattr(first, "dtype", np.float32)

        for i, t in enumerate(tensors):
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

            sizes_along_axis.append(int(t.shape[axis]))

        # Build output shape
        out_shape = list(ref_shape)
        out_shape[axis] = int(sum(sizes_along_axis))
        out_shape_t = tuple(int(x) for x in out_shape)

        req = any(t.requires_grad for t in tensors)

        # ============================================================
        # CPU path (keep existing semantics)
        # ============================================================
        if dev.is_cpu():
            arrs = [t.to_numpy() for t in tensors]
            # preserve your existing behavior: float32 output
            out_np = np.concatenate(arrs, axis=axis).astype(np.float32, copy=False)

            out = Tensor(shape=out_np.shape, device=dev, requires_grad=req, ctx=None)
            out.copy_from_numpy(out_np)

            if req:
                offsets = [0]
                for s in sizes_along_axis:
                    offsets.append(offsets[-1] + int(s))

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cpu():
                        raise RuntimeError(
                            "grad_out must be CPU in current implementation"
                        )

                    g = grad_out.to_numpy()
                    grads: list[Optional["Tensor"]] = []

                    for i, t in enumerate(tensors):
                        if not t.requires_grad:
                            grads.append(None)
                            continue

                        start = offsets[i]
                        end = offsets[i + 1]

                        slicer = [slice(None)] * ndim
                        slicer[axis] = slice(start, end)

                        gi_np = g[tuple(slicer)].astype(np.float32, copy=False)

                        gi = Tensor(
                            shape=t.shape, device=dev, requires_grad=False, ctx=None
                        )
                        gi.copy_from_numpy(gi_np)
                        grads.append(gi)

                    return tuple(grads)

                ctx = Context(parents=tuple(tensors), backward_fn=backward_fn)
                ctx.saved_meta["concat_axis"] = axis
                ctx.saved_meta["concat_sizes"] = sizes_along_axis
                out._set_ctx(ctx)

            return out

        # ============================================================
        # CUDA path (correctness-first fallback via CPU)
        # ============================================================
        if dev.is_cuda():
            # ------------------------------------------------------------------
            # IMPORTANT:
            # Raw device-to-device memcpy is NOT generally correct for tensor
            # concatenation, even for axis == 0, due to row-major memory layout
            # and striding. Without a dedicated CUDA kernel, attempting to
            # concatenate via flat byte copies leads to interleaved / corrupted
            # results (see unit tests).
            #
            # Therefore, we intentionally fall back to:
            #   CUDA -> CPU -> np.concatenate -> CUDA
            #
            # This ensures correctness and preserves autograd semantics.
            #
            # TODO (performance):
            # - Implement a dedicated CUDA concat kernel supporting arbitrary axis
            # - OR add a restricted fast-path ONLY for 1D tensors (ndim == 1)
            # ------------------------------------------------------------------

            # Materialize inputs on CPU
            arrs = [t.to_numpy() for t in tensors]
            out_np = np.concatenate(arrs, axis=axis)

            out = Tensor(
                shape=out_np.shape,
                device=dev,
                requires_grad=req,
                ctx=None,
            )
            out.copy_from_numpy(out_np)

            if req:
                offsets = [0]
                for s in sizes_along_axis:
                    offsets.append(offsets[-1] + int(s))

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA concat backward"
                        )

                    # Pull grad_out to CPU for correct slicing
                    g_np = grad_out.to_numpy()

                    grads: list[Optional["Tensor"]] = []
                    for i, t in enumerate(tensors):
                        if not t.requires_grad:
                            grads.append(None)
                            continue

                        start = offsets[i]
                        end = offsets[i + 1]

                        slicer = [slice(None)] * ndim
                        slicer[axis] = slice(start, end)

                        gi_np = g_np[tuple(slicer)]

                        gi = Tensor(
                            shape=t.shape,
                            device=dev,
                            requires_grad=False,
                            ctx=None,
                        )
                        gi.copy_from_numpy(gi_np)
                        grads.append(gi)

                    return tuple(grads)

                ctx = Context(
                    parents=tuple(tensors),
                    backward_fn=backward_fn,
                )
                ctx.saved_meta["concat_axis"] = axis
                ctx.saved_meta["concat_sizes"] = sizes_along_axis
                ctx.saved_meta["cuda_fallback_via_cpu"] = True
                ctx.saved_meta["todo"] = "implement CUDA concat kernel"

                out._set_ctx(ctx)

            return out

    def __getitem__(self, key: Any) -> "Tensor":
        """
        Slice/index into a tensor, producing a new Tensor.

        CPU behavior
        ------------
        - Uses NumPy slicing/indexing and returns a *copy* (not a view).
        - Backward rule scatters grad_out back into the parent tensor shape.
        - Supports basic slicing and NumPy-style fancy indexing.

        CUDA behavior (workaround)
        --------------------------
        - Currently implemented as a CPU fallback:
            1) D2H copy of `self` to CPU
            2) NumPy slicing/indexing on CPU
            3) H2D copy of the sliced result back to CUDA
        - Backward is also CPU-scatter + H2D copy.

        TODO
        ----
        - Implement native CUDA gather for forward and scatter/add for backward
        (especially to support fancy indexing efficiently and avoid round-trips).
        """
        import numpy as np

        # -------------------------
        # Helpers
        # -------------------------
        def _is_fancy(k: Any) -> bool:
            # Bool mask, list, ndarray => fancy
            if isinstance(k, (list, np.ndarray)):
                return True
            if isinstance(k, tuple):
                return any(isinstance(kk, (list, np.ndarray)) for kk in k)
            return False

        # Keep dtype handling consistent across CPU/CUDA paths.
        # (Many of your kernels/tests assume float32; if you later support float64,
        # this will follow `self.dtype`.)
        dt = np.dtype(getattr(self, "dtype", np.float32))

        # -------------------------
        # CPU path (unchanged semantics)
        # -------------------------
        if self.device.is_cpu():
            src = self.to_numpy()
            sliced = src[key]

            # Normalize scalar outputs to shape=()
            if np.isscalar(sliced) or getattr(sliced, "shape", None) == ():
                sliced_arr = np.array(sliced, dtype=dt)  # shape ()
                out_shape = ()
            else:
                # Ensure a real ndarray of the desired dtype (still a copy semantic).
                sliced_arr = np.asarray(sliced, dtype=dt)
                out_shape = sliced_arr.shape

            req = self.requires_grad
            out = Tensor(shape=out_shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(sliced_arr)

            if req:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cpu():
                        raise RuntimeError(
                            "grad_out must be CPU in current implementation"
                        )

                    g_out_np = grad_out.to_numpy()
                    grad_parent_np = np.zeros(self.shape, dtype=dt)

                    fancy = _is_fancy(key)
                    if fancy:
                        np.add.at(grad_parent_np, key, g_out_np)
                    else:
                        grad_parent_np[key] += g_out_np

                    grad_parent = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                    )
                    grad_parent.copy_from_numpy(grad_parent_np)
                    return (grad_parent,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                ctx.saved_meta["getitem_key"] = key
                ctx.saved_meta["parent_shape"] = self.shape
                out._set_ctx(ctx)

            return out

        # -------------------------
        # CUDA path (CPU fallback workaround)
        # -------------------------
        if self.device.is_cuda():
            # 1) D2H: bring self to CPU
            x_cpu = self.to(Device("cpu"), copy=True)

            # 2) NumPy slice/index on CPU
            src = x_cpu.to_numpy()
            sliced = src[key]

            if np.isscalar(sliced) or getattr(sliced, "shape", None) == ():
                sliced_arr = np.array(sliced, dtype=dt)  # shape ()
                out_shape = ()
            else:
                # CRITICAL FIX:
                # - NumPy slicing often returns a *strided, non-contiguous view*.
                # - Raw memcpy assumes contiguous bytes. Force contiguity here.
                sliced_arr = np.ascontiguousarray(sliced, dtype=dt)
                out_shape = sliced_arr.shape

            req = self.requires_grad
            out = Tensor(
                shape=out_shape, device=self.device, requires_grad=req, ctx=None
            )
            out._ensure_cuda_alloc(dtype=dt)

            # 3) H2D: copy sliced result back to CUDA
            from ..native_cuda.python.ops import memcpy_ctypes as mc

            lib_out = out._get_cuda_lib()

            # Ensure the host buffer passed to memcpy is contiguous
            sliced_host = np.ascontiguousarray(sliced_arr, dtype=dt)
            mc.memcpy_htod(
                lib_out,
                dst_dev=int(out.data),
                src_host=sliced_host,
                nbytes=int(sliced_host.nbytes),
                sync=True,
            )

            if req:
                fancy = _is_fancy(key)

                def backward_fn(grad_out: "Tensor"):
                    # Expect CUDA grad_out for CUDA parent
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA getitem backward"
                        )
                    if str(grad_out.device) != str(self.device):
                        raise RuntimeError(
                            f"grad_out device mismatch: expected {self.device!r}, got {grad_out.device!r}"
                        )

                    # CPU fallback backward:
                    # - D2H grad_out
                    go_cpu = grad_out.to(Device("cpu"), copy=True)
                    g_out_np = go_cpu.to_numpy()

                    grad_parent_np = np.zeros(self.shape, dtype=dt)

                    if fancy:
                        np.add.at(grad_parent_np, key, g_out_np)
                    else:
                        grad_parent_np[key] += g_out_np

                    # H2D scatter result as grad for parent
                    grad_parent = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                    )
                    grad_parent._ensure_cuda_alloc(dtype=dt)

                    lib_gp = grad_parent._get_cuda_lib()

                    # Ensure contiguous before memcpy (even though zeros_like is contiguous,
                    # this keeps the rule consistent and future-proof).
                    gp_host = np.ascontiguousarray(grad_parent_np, dtype=dt)

                    mc.memcpy_htod(
                        lib_gp,
                        dst_dev=int(grad_parent.data),
                        src_host=gp_host,
                        nbytes=int(gp_host.nbytes),
                        sync=True,
                    )
                    return (grad_parent,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                ctx.saved_meta["getitem_key"] = key
                ctx.saved_meta["parent_shape"] = self.shape
                ctx.saved_meta["cuda_cpu_fallback"] = (
                    True  # TODO: remove after CUDA kernel
                )
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("getitem")
        raise RuntimeError("Unreachable")

    @staticmethod
    def _cuda_detach_view_no_grad(t: "Tensor") -> "Tensor":
        """
        CUDA-only: return a view Tensor sharing the same devptr, with no autograd tracking.

        Supports both:
        - storage-managed CUDA tensors (t._storage is not None)
        - legacy/borrowed CUDA tensors (t._storage is None but t.data != 0)
        """
        import numpy as np

        if not t.device.is_cuda():
            t._raise_device_not_supported("cuda_detach_view_no_grad")

        dp = int(t.data)
        if dp == 0 and t.numel() != 0:
            raise RuntimeError(
                "cuda_detach_view_no_grad: tensor has no allocated devptr (data == 0)"
            )

        st = getattr(t, "_storage", None)

        if st is not None:
            return Tensor._from_storage(
                st,
                shape=tuple(t.shape),
                device=t.device,
                requires_grad=False,
                ctx=None,
                dtype=np.dtype(t.dtype),
            )

        # Borrowed/raw pointer fallback (no ownership)
        out = Tensor._from_devptr(
            dev_ptr=dp,
            shape=tuple(t.shape),
            device=t.device,
            requires_grad=False,
            ctx=None,
            dtype=np.dtype(t.dtype),
        )
        # Ensure invariant
        out._storage = None
        return out

    @staticmethod
    def _add_no_grad_cuda(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        CUDA-only: add without autograd edges using tensor_arithmetic_cuda_ext.add.
        """
        if not (a.device.is_cuda() and b.device.is_cuda()):
            raise RuntimeError("_add_no_grad_cuda expects CUDA tensors")
        if str(a.device) != str(b.device):
            raise ValueError("Device mismatch in _add_no_grad_cuda")
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError("Shape mismatch in _add_no_grad_cuda")
        if np.dtype(a.dtype) != np.dtype(b.dtype):
            raise TypeError("Dtype mismatch in _add_no_grad_cuda")

        from ..ops.tensor_arithmetic_cuda_ext import add as cuda_add

        # NOTE: device index: prefer Device.index; fallback parse "cuda:N"
        idx = getattr(a.device, "index", None)
        if idx is None:
            s = str(a.device)
            idx = int(s.split(":")[1]) if ":" in s else 0

        out = cuda_add(a, b, device=int(idx))
        # out is already requires_grad=False by wrapper; keep it explicit
        out.requires_grad = False
        out._set_ctx(None)
        return out

    def _accumulate_grad_cuda_(self, g: "Tensor") -> None:
        """
        CUDA-only: accumulate gradient into self._grad (no CPU fallback).
        Mirrors CPU _accumulate_grad_ semantics.
        """
        if not self.device.is_cuda():
            self._raise_device_not_supported("accumulate_grad_cuda")

        # Make sure gradient tensor does not carry autograd metadata
        g0 = (
            self._cuda_detach_view_no_grad(g)
            if (g.requires_grad or g._get_ctx() is not None)
            else g
        )
        if self._grad is None:
            self._grad = self._cuda_detach_view_no_grad(g0)
            return

        # In-place would be better, but out-of-place is fine if add_inplace isn't available yet.
        self._grad = Tensor._add_no_grad_cuda(self._grad, g0)

    def free_(self) -> None:
        """
        Explicitly release CUDA backing memory (if owned).

        Semantics
        ---------
        - If `_storage` is present: decrement refcount; free happens when it hits 0.
        - If `_storage` is absent:
            - If this tensor is marked as borrowed, we do NOT free the devptr.
            - If marked as "owning borrowed devptr" (transitional), we cuda_free it.
        - Idempotent: safe to call multiple times.
        """
        if not self.device.is_cuda():
            return

        # 1) Storage-backed path (preferred)
        st = getattr(self, "_storage", None)
        if st is not None:
            try:
                st.decref()
            finally:
                self._storage = None
                # keep legacy mirror consistent
                self._data = 0
            return

        # 2) No storage: devptr-only legacy path
        dp = int(getattr(self, "_data", 0) or 0)
        if dp == 0:
            # already freed / never allocated / numel==0 case
            self._data = 0
            return

        # Borrowed pointers must not be freed here.
        if bool(getattr(self, "_borrowed_devptr", False)):
            # just detach
            self._data = 0
            return

        # Optional: transitional ownership flag for devptr-only tensors.
        # Only free if you *explicitly* marked this tensor as owning the devptr.
        if bool(getattr(self, "_owns_devptr", False)):
            try:
                lib = self._get_cuda_lib()
                # Use your canonical wrapper (pick the one you standardized on)
                from ..native_cuda.python import maxpool2d_ctypes as m

                # Ensure correct device before free (Windows correctness)
                device_index = int(getattr(self.device, "index", 0) or 0)
                if hasattr(m, "cuda_set_device"):
                    m.cuda_set_device(lib, device_index)

                m.cuda_free(lib, int(dp))
            finally:
                self._data = 0
            return

        # Default safest behavior: do not free unknown devptr ownership.
        # Detach so we don't keep referencing it.
        self._data = 0
