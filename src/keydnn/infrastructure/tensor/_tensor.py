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

from ...domain._tensor import ITensor
from ...domain.device._device import Device
from ...domain._errors import DeviceNotSupportedError
from ._tensor_context import Context

Number = Union[int, float]


from .mixins.reduction import TensorMixinReduction


class Tensor(TensorMixinReduction, ITensor):
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
        dtype: np.dtype = np.float32,
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
        dtype : np.dtype, optional
            Element dtype metadata for kernel dispatch (e.g., np.float32/np.float64).
            Defaults to np.float32.

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

        dp = int(dev_ptr)

        obj = cls.__new__(cls)  # bypass __init__
        obj._shape = shape
        obj._device = device
        obj._data = dp
        obj._dtype = np.dtype(dtype)

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
            return (
                f"Tensor(shape={self._shape}, device={d}, dtype={self._dtype}, "
                f"data=DevPtr({int(self._data)}))"
            )
        return f"Tensor(shape={self._shape}, device={d}, dtype={self._data.dtype})"

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

    @property
    def T(self) -> "Tensor":
        """
        Convenience property for 2D transpose.
        """
        return self.transpose()

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
        from .._function import TanhFn  # adjust if needed

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
        from .._function import SigmoidFn  # adjust import to your project layout

        # Build context with parents AND a callable backward_fn
        ctx = Context(parents=(self,), backward_fn=None)

        out = SigmoidFn.forward(ctx, self)

        # IMPORTANT: Tensor.backward() expects ctx.backward_fn to be callable
        ctx.backward_fn = lambda grad_out: (SigmoidFn.backward(ctx, grad_out),)

        out._set_ctx(ctx)
        return out

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
            If backward is invoked on an unsupported device.

        Notes
        -----
        - Gradients are accumulated into `.grad` of leaf tensors that have
        `requires_grad=True`.
        - This implementation performs a reverse topological traversal.
        - CPU behavior is unchanged.
        """
        # ---------------------------------------------------------------------
        # CPU path: KEEP EXACTLY the original semantics (backward compatible).
        # ---------------------------------------------------------------------
        if self.device.is_cpu():
            # ---- BEGIN original implementation (CPU-only) ----

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
                    raise TypeError(
                        f"grad_out must be a Tensor, got {type(grad_out)!r}"
                    )
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
                for t in topo:
                    if id(t) == tid:
                        if t.requires_grad:
                            t._accumulate_grad_(g)
                        break

            # ---- END original implementation (CPU-only) ----
            return

        # ---------------------------------------------------------------------
        # CUDA path: additive support (minimal, does NOT change CPU behavior).
        # ---------------------------------------------------------------------
        if not self.device.is_cuda():
            self._raise_device_not_supported("backward")

        # Helper: share a single CUDA DLL handle to avoid multi-handle issues.
        def _get_cuda_lib():
            lib = getattr(Tensor, "_CUDA_LIB", None)
            if lib is None:
                from ...infrastructure.native_cuda.python import (
                    maxpool2d_ctypes as m,
                )

                lib = m.load_keydnn_cuda_native()
                setattr(Tensor, "_CUDA_LIB", lib)
            return lib

        # Seed gradient (CUDA)
        if grad_out is None:
            if self.shape != ():
                raise ValueError(
                    "grad_out must be provided for non-scalar tensors. "
                    f"Got shape={self.shape}."
                )

            from ...infrastructure.native_cuda.python import (
                maxpool2d_ctypes as m,
            )

            lib = _get_cuda_lib()
            host = np.array(1.0, dtype=np.float32)
            dev_ptr = int(m.cuda_malloc(lib, int(host.nbytes)))

            try:
                m.cudaMemcpyHtoD(lib, int(dev_ptr), host, int(host.nbytes))

                grad_out = Tensor._from_devptr(
                    dev_ptr=int(dev_ptr),
                    shape=(),  # scalar
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                    dtype=np.float32,
                )
            except Exception:
                try:
                    m.cuda_free(lib, int(dev_ptr))
                except Exception:
                    pass
                raise
        else:
            if not isinstance(grad_out, Tensor):
                raise TypeError(f"grad_out must be a Tensor, got {type(grad_out)!r}")
            if grad_out.shape != self.shape:
                raise ValueError(
                    f"grad_out shape mismatch: expected {self.shape}, got {grad_out.shape}"
                )
            if grad_out.device != self.device:
                raise ValueError("grad_out must be on the same device as self")

        # Build reverse topo on CUDA graph (same structure as CPU)
        topo: list[Tensor] = []
        visited: set[int] = set()

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

        grads: dict[int, Tensor] = {id(self): grad_out}

        # CUDA: no generic accumulation yet (requires add kernel),
        # so we only support single-contribution gradients per tensor id.
        for t in reversed(topo):
            ctx = t._get_ctx()
            if ctx is None:
                continue

            grad_t = grads.get(id(t))
            if grad_t is None:
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

                # Enforce "non-tracking" grads on CUDA to avoid needing detach helpers.
                if getattr(g, "requires_grad", False):
                    raise RuntimeError(
                        "CUDA backward_fn must return requires_grad=False gradient tensors."
                    )
                if g._get_ctx() is not None:
                    raise RuntimeError(
                        "CUDA backward_fn must return gradient tensors with ctx=None."
                    )

                pid = id(parent)
                if pid in grads:
                    raise RuntimeError(
                        "CUDA backward currently does not support accumulating multiple "
                        "gradient contributions into the same tensor."
                    )
                grads[pid] = g

        # Write accumulated grads into leaf tensors that require grad
        # NOTE: _accumulate_grad_ is CPU-only, so set _grad directly for CUDA.
        for tid, g in grads.items():
            for t in topo:
                if id(t) != tid:
                    continue
                if not t.requires_grad:
                    break

                # Best-effort "leaf grad set" semantics:
                # - if no existing grad: set
                # - else: would require CUDA add kernel
                if getattr(t, "_grad", None) is None:
                    t._grad = g
                else:
                    raise RuntimeError(
                        "CUDA backward currently does not support accumulating into an existing .grad."
                    )
                break

    def to_numpy(self) -> np.ndarray:
        """
        Convert the tensor to a NumPy ndarray.

        Returns
        -------
        np.ndarray
            A NumPy array containing the tensor data on the host.

        Raises
        ------
        RuntimeError
            If device-to-host transfer is unavailable or the tensor's dtype is unknown.

        Notes
        -----
        - CPU tensors return a view/copy of the underlying CPU storage (unchanged behavior).
        - CUDA tensors are copied from device to host via a DtoH memcpy.
        """
        # -----------------------
        # CPU path (unchanged)
        # -----------------------
        if self._device.is_cpu():
            return self._data

        # -----------------------
        # CUDA path
        # -----------------------
        if not self._device.is_cuda():
            raise RuntimeError(
                f"to_numpy() is not available for device {self._device!s}"
            )

        # Helper: share a single CUDA DLL handle to avoid multi-handle issues.
        def _get_cuda_lib():
            lib = getattr(Tensor, "_CUDA_LIB", None)
            if lib is None:
                from ...infrastructure.native_cuda.python import (
                    maxpool2d_ctypes as m,
                )

                lib = m.load_keydnn_cuda_native()
                setattr(Tensor, "_CUDA_LIB", lib)
            return lib

        try:
            from ...infrastructure.native_cuda.python import (
                maxpool2d_ctypes as m,
            )
        except Exception as e:
            raise RuntimeError(
                f"to_numpy() CUDA transfer requires native CUDA ctypes wrappers: {e!r}"
            )

        # Figure out dtype
        dtype = getattr(self, "dtype", None)
        if dtype is None:
            # Some codebases store dtype in _dtype; try that as a fallback.
            dtype = getattr(self, "_dtype", None)
        if dtype is None:
            raise RuntimeError("CUDA Tensor has no dtype; cannot materialize to NumPy.")

        dtype = np.dtype(dtype)

        # Allocate host array and copy device -> host
        out = np.empty(self.shape, dtype=dtype, order="C")

        dev_ptr = int(getattr(self, "data"))
        nbytes = int(out.nbytes)

        lib = _get_cuda_lib()
        # Expected signature: (lib, dst_host_ndarray, src_dev_ptr, nbytes)
        m.cudaMemcpyDtoH(lib, out, dev_ptr, nbytes)

        return out

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

    # def _ensure_cuda_alloc(self, *, dtype) -> None:
    #     """
    #     Ensure that a CUDA tensor has an allocated device buffer.

    #     This method allocates device memory sized to
    #     `numel * dtype.itemsize` and stores the resulting device pointer on
    #     the tensor instance. Memory contents are left uninitialized.

    #     Parameters
    #     ----------
    #     dtype
    #         Desired NumPy-compatible data type for the device buffer.

    #     Raises
    #     ------
    #     RuntimeError
    #         If called on a tensor whose device is not CUDA.

    #     Notes
    #     -----
    #     - Zero-sized tensors do not allocate device memory; their device pointer
    #     is set to `0`.
    #     - This method does not perform any data initialization (e.g., zeros or
    #     ones); it is purely an allocation helper.
    #     - Device selection is performed internally based on `self.device.index`.
    #     """
    #     import numpy as np
    #     from ..native_cuda.python.avgpool2d_ctypes import cuda_set_device, cuda_malloc

    #     if not self.device.is_cuda():
    #         raise RuntimeError("_ensure_cuda_alloc is only valid for CUDA tensors.")

    #     dtype = np.dtype(dtype)
    #     numel = self._numel_from_shape()

    #     # zero-sized: represent as null pointer
    #     if numel == 0:
    #         self._data = (
    #             0  # store device ptr here (adjust if you have a dedicated field)
    #         )
    #         self._dtype = dtype
    #         return

    #     lib = self._get_cuda_lib()
    #     cuda_set_device(lib, int(self.device.index or 0))

    #     nbytes = int(numel) * int(dtype.itemsize)
    #     dev_ptr = cuda_malloc(lib, nbytes)

    #     # Store device pointer on tensor. Adjust if your tensor uses another attribute.
    #     self._data = int(dev_ptr)
    #     self._dtype = dtype

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
        import numpy as np
        from ..native_cuda.python.avgpool2d_ctypes import cuda_set_device, cuda_malloc

        # If you have cuda_free available somewhere, import it too:
        # from ..native_cuda.python.avgpool2d_ctypes import cuda_free

        if not self.device.is_cuda():
            raise RuntimeError("_ensure_cuda_alloc is only valid for CUDA tensors.")

        dtype = np.dtype(dtype)
        numel = self._numel_from_shape()

        # zero-sized: represent as null pointer
        if numel == 0:
            self._data = 0
            self._dtype = dtype
            return

        # Compute required size
        required_nbytes = int(numel) * int(dtype.itemsize)

        # If already allocated, only accept it if compatible
        cur_ptr = int(getattr(self, "_data", 0) or 0)
        cur_dtype = getattr(self, "_dtype", None)
        if cur_ptr != 0 and cur_dtype is not None:
            cur_dtype = np.dtype(cur_dtype)
            cur_nbytes = int(numel) * int(cur_dtype.itemsize)

            # Same shape implied; if size matches, keep existing buffer
            if cur_nbytes == required_nbytes:
                # Keep current allocation, just update dtype metadata if needed
                self._dtype = dtype
                return

            # Otherwise, we need to reallocate (shape same but dtype size differs).
            # TODO: free old device pointer to avoid leaks once cuda_free is wired.
            # cuda_free(lib, cur_ptr)
            # self._data = 0

        lib = self._get_cuda_lib()
        cuda_set_device(lib, int(self.device.index or 0))

        dev_ptr = cuda_malloc(lib, required_nbytes)
        self._data = int(dev_ptr)
        self._dtype = dtype

    @staticmethod
    def zeros(
        *, shape: tuple[int, ...], device: Device, requires_grad: bool = False
    ) -> "Tensor":
        """
        Create a tensor filled with zeros on the specified device.

        This factory method constructs a tensor with the given shape and device.
        For CPU tensors, a NumPy array is allocated and zero-initialized.
        For CUDA tensors, device memory is allocated and zeroed via `cudaMemset`.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the output tensor.
        device : Device
            Target device placement (CPU or CUDA).
        requires_grad : bool, optional
            Whether the tensor should track gradients for autograd.

        Returns
        -------
        Tensor
            Newly created tensor filled with zeros.

        Notes
        -----
        - The dtype is currently fixed to `float32`.
        - Zero-sized tensors are valid and return immediately without invoking
        CUDA kernels.
        """
        import numpy as np

        dtype = np.float32
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)

        if device.is_cpu():
            out._data = np.zeros(shape, dtype=dtype)
            return out

        out._ensure_cuda_alloc(dtype=dtype)
        numel = out._numel_from_shape()
        if numel == 0:
            return out

        from ..ops.fill_cuda import zeros_cuda

        lib = out._get_cuda_lib()
        zeros_cuda(lib, y_dev=int(out._data), numel=numel, dtype=dtype, sync=True)
        return out

    @staticmethod
    def ones(
        *, shape: tuple[int, ...], device: Device, requires_grad: bool = False
    ) -> "Tensor":
        """
        Create a tensor filled with ones on the specified device.

        This factory method constructs a tensor with the given shape and device.
        For CPU tensors, a NumPy array is allocated and initialized with ones.
        For CUDA tensors, device memory is allocated and filled using a native
        CUDA kernel, with a host-to-device memcpy fallback if needed.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the output tensor.
        device : Device
            Target device placement (CPU or CUDA).
        requires_grad : bool, optional
            Whether the tensor should track gradients for autograd.

        Returns
        -------
        Tensor
            Newly created tensor filled with ones.

        Notes
        -----
        - The dtype is currently fixed to `float32`.
        - Zero-sized tensors are valid and return immediately without invoking
        CUDA kernels.
        - The CUDA path prioritizes correctness and may fall back to a slower
        initialization strategy if the native fill kernel fails.
        """
        import numpy as np

        dtype = np.float32
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)

        if device.is_cpu():
            out._data = np.ones(shape, dtype=dtype)
            return out

        out._ensure_cuda_alloc(dtype=dtype)
        numel = out._numel_from_shape()
        if numel == 0:
            return out

        from ..ops.fill_cuda import ones_cuda

        lib = out._get_cuda_lib()
        ones_cuda(lib, y_dev=int(out._data), numel=numel, dtype=dtype, sync=True)
        return out

    # ----------------------------
    # Transpose (2D)
    # ----------------------------
    def transpose(self) -> "Tensor":
        """
        2D transpose: out[i, j] = self[j, i].

        Requirements
        ------------
        - input must be 2D
        - CPU and CUDA supported

        Backward
        --------
        If out = A^T, then dL/dA = (dL/dout)^T
        """
        if len(self.shape) != 2:
            raise ValueError(f"transpose requires a 2D tensor, got shape={self.shape}")

        r, c = self.shape
        req = self.requires_grad

        # -----------------------
        # CUDA path
        # -----------------------
        if self.device.is_cuda():
            import numpy as np
            from ..ops.transpose_cuda import transpose2d_cuda  # ops-layer wrapper

            dtype = np.dtype(self.dtype)

            # Input must have allocated device memory
            x_dev = int(self.data)
            if x_dev == 0:
                raise RuntimeError(
                    "CUDA transpose requires an allocated input device buffer (data != 0)"
                )

            # Allocate output buffer if needed
            out = Tensor(
                shape=(c, r),
                device=self.device,
                requires_grad=req,
                ctx=None,
                dtype=dtype,
            )
            out._ensure_cuda_alloc(dtype=dtype)
            y_dev = int(out.data)
            if y_dev == 0:
                raise RuntimeError(
                    "CUDA transpose failed to allocate output device buffer"
                )

            lib = self._get_cuda_lib()

            # Forward kernel
            transpose2d_cuda(
                lib,
                x_dev=x_dev,
                y_dev=y_dev,
                rows=int(r),
                cols=int(c),
                dtype=dtype,
                sync=True,
            )

            if req:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA transpose backward"
                        )
                    if grad_out.shape != (c, r):
                        raise ValueError(
                            f"grad_out shape mismatch: expected {(c, r)}, got {grad_out.shape}"
                        )

                    go_dev = int(grad_out.data)
                    if go_dev == 0:
                        raise RuntimeError(
                            "grad_out CUDA tensor has no allocated devptr (data == 0)"
                        )

                    grad = Tensor(
                        shape=(r, c),
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                        dtype=dtype,
                    )
                    grad._ensure_cuda_alloc(dtype=dtype)
                    gx_dev = int(grad.data)
                    if gx_dev == 0:
                        raise RuntimeError(
                            "CUDA transpose backward failed to allocate grad buffer"
                        )

                    # grad_x = grad_out^T
                    transpose2d_cuda(
                        lib,
                        x_dev=go_dev,
                        y_dev=gx_dev,
                        rows=int(c),
                        cols=int(r),
                        dtype=dtype,
                        sync=True,
                    )
                    return (grad,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                out._set_ctx(ctx)

            return out

        # -----------------------
        # CPU path (unchanged)
        # -----------------------
        if not self.device.is_cpu():
            self._raise_device_not_supported("transpose")

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
                g_np = grad_out.to_numpy().T
                grad_parent = Tensor(
                    shape=self.shape, device=self.device, requires_grad=False, ctx=None
                )
                grad_parent.copy_from_numpy(g_np)
                return (grad_parent,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            out._set_ctx(ctx)

        return out

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
        if self._device.is_cpu():
            self._data.fill(value)
            return

        if self._device.is_cuda():
            # ensure device buffer exists before calling native fill
            self._ensure_cuda_alloc(dtype=self.dtype)

            from ..ops.fill_cuda_ext import fill_ as _fill_cuda_

            _fill_cuda_(
                self, float(value), device=int(self._device.index or 0), sync=True
            )
            return

        self._raise_device_not_supported("fill")

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

        # ----------------------------
        # CPU path (backward compatible)
        # ----------------------------
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

        # ----------------------------
        # CUDA path (device-pointer, no to_numpy)
        # ----------------------------
        if self._device.is_cuda() and other_t.device.is_cuda():
            self._binary_op_shape_check(self, other_t)

            # Enforce dtype policy consistent with CUDA kernels (f32/f64 only, same dtype)
            import numpy as np

            dt_a = np.dtype(self.dtype)
            dt_b = np.dtype(other_t.dtype)
            if dt_a not in (np.float32, np.float64) or dt_b not in (
                np.float32,
                np.float64,
            ):
                raise TypeError(
                    f"CUDA truediv supports float32/float64 only; got self.dtype={dt_a}, other.dtype={dt_b}"
                )
            if dt_a != dt_b:
                raise TypeError(
                    f"CUDA truediv requires matching dtypes; got {dt_a} vs {dt_b}"
                )

            # Require same CUDA device placement
            if self.device != other_t.device and str(self.device) != str(
                other_t.device
            ):
                raise ValueError(
                    f"device mismatch: self.device={self.device} vs other.device={other_t.device}"
                )

            from ..ops.tensor_arithmetic_cuda_ext import div as _cuda_div

            device_index = int(self._device.index or 0)
            req = self._result_requires_grad(self, other_t)

            out = _cuda_div(self, other_t, device=device_index)
            out.requires_grad = bool(req)

            if req:
                # grad_a = grad_out / b
                # grad_b = -(grad_out * a) / (b*b)
                # Use existing Tensor ops; for CUDA these should route to CUDA-friendly paths.
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
    # Python 2 legacy division alias
    # ----------------------------
    def __div__(self, other: Union["Tensor", Number]) -> "Tensor":
        """
        Elementwise division (legacy alias for true division).

        Notes
        -----
        Python 3 uses `__truediv__` for `/`. This method exists for compatibility
        with code that still calls `__div__` explicitly.
        """
        return self.__truediv__(other)

    def __rdiv__(self, other: Number) -> "Tensor":
        """
        Right-hand division (legacy alias for right true division).

        Notes
        -----
        Python 3 uses `__rtruediv__`. This method exists for compatibility with
        code that still calls `__rdiv__` explicitly.
        """
        return self.__rtruediv__(other)

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

        # -----------------------------
        # CPU path (existing)
        # -----------------------------
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

        # -----------------------------
        # CUDA path (device-pointer elementwise)
        # -----------------------------
        if self._device.is_cuda() and other_t.device.is_cuda():
            self._binary_op_shape_check(self, other_t)

            # dtype must match for our CUDA kernels
            if np.dtype(self.dtype) != np.dtype(other_t.dtype):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
                )

            req = self._result_requires_grad(self, other_t)

            # Use your CUDA ext wrapper (allocates output and returns CUDA Tensor)
            from ..ops.tensor_arithmetic_cuda_ext import add as _cuda_add

            # Prefer the tensor's device index if available; otherwise default 0
            device_index = int(getattr(self._device, "index", 0) or 0)

            out = _cuda_add(self, other_t, device=device_index)
            out.requires_grad = bool(req)  # ensure flag matches CPU behavior

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

        # -----------------------------
        # CPU path (existing)
        # -----------------------------
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

        # -----------------------------
        # CUDA path (device-pointer elementwise)
        # -----------------------------
        if self._device.is_cuda() and other_t.device.is_cuda():
            self._binary_op_shape_check(self, other_t)

            if np.dtype(self.dtype) != np.dtype(other_t.dtype):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
                )

            req = self._result_requires_grad(self, other_t)

            from ..ops.tensor_arithmetic_cuda_ext import sub as _cuda_sub

            device_index = int(getattr(self._device, "index", 0) or 0)

            out = _cuda_sub(self, other_t, device=device_index)
            out.requires_grad = bool(req)

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

        # -----------------------------
        # CPU path
        # -----------------------------
        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            out = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            out.copy_from_numpy(
                (self.to_numpy() > other_t.to_numpy()).astype(np.float32)
            )
            return out

        # -----------------------------
        # CUDA path
        # -----------------------------
        if self._device.is_cuda() and other_t.device.is_cuda():
            self._binary_op_shape_check(self, other_t)

            # dtype must match for CUDA comparison kernels
            if np.dtype(self.dtype) != np.dtype(other_t.dtype):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(self.dtype)} vs other.dtype={np.dtype(other_t.dtype)}"
                )

            from ..ops.tensor_arithmetic_cuda_ext import gt as _cuda_gt

            device_index = int(getattr(self._device, "index", 0) or 0)
            out = _cuda_gt(self, other_t, device=device_index)

            # Explicitly ensure no gradients
            out.requires_grad = False
            return out

        self._raise_device_not_supported("gt")

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
        # -----------------------------
        # CPU path
        # -----------------------------
        if self._device.is_cpu():
            out = Tensor(
                shape=self.shape,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            out.copy_from_numpy(-self.to_numpy())

            if self.requires_grad:
                ctx = Context(
                    parents=(self,),
                    backward_fn=lambda grad_out: (-(grad_out),),
                )
                out._set_ctx(ctx)

            return out

        # -----------------------------
        # CUDA path
        # -----------------------------
        if self._device.is_cuda():
            from ..ops.tensor_arithmetic_cuda_ext import neg as _cuda_neg

            device_index = int(getattr(self._device, "index", 0) or 0)
            out = _cuda_neg(self, device=device_index)

            # Preserve autograd semantics
            out.requires_grad = bool(self.requires_grad)

            if self.requires_grad:
                ctx = Context(
                    parents=(self,),
                    backward_fn=lambda grad_out: (-(grad_out),),
                )
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("neg")

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __ge__(self, other: Union["Tensor", Number]) -> "Tensor":
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
        other_t = self._as_tensor_like(other, self)

        # Ensure same shape (and implicitly same device support path via __gt__)
        self._binary_op_shape_check(self, other_t)

        # ge = 1 - (other > self)
        gt = other_t.__gt__(self)  # float32 mask
        one = self._as_tensor_like(1.0, gt)
        out = one - gt

        out.requires_grad = False
        return out

    def __lt__(self, other: Union["Tensor", Number]) -> "Tensor":
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
        other_t = self._as_tensor_like(other, self)
        self._binary_op_shape_check(self, other_t)

        out = other_t.__gt__(self)
        out.requires_grad = False
        return out

    def __le__(self, other: Union["Tensor", Number]) -> "Tensor":
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
        other_t = self._as_tensor_like(other, self)
        self._binary_op_shape_check(self, other_t)

        # le = 1 - (self > other)
        gt = self.__gt__(other_t)  # float32 mask
        one = self._as_tensor_like(1.0, gt)
        out = one - gt

        out.requires_grad = False
        return out

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

        # -----------------------------
        # CPU path (existing)
        # -----------------------------
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

        # -----------------------------
        # CUDA path
        # -----------------------------
        if self._device.is_cuda() and other_t.device.is_cuda():
            self._binary_op_shape_check(self, other_t)

            # Enforce same dtype semantics as other CUDA arithmetic ops ext
            import numpy as np

            dt_self = np.dtype(self.dtype)
            dt_other = np.dtype(other_t.dtype)
            if dt_self != dt_other:
                raise TypeError(
                    f"dtype mismatch: self.dtype={dt_self} vs other.dtype={dt_other}"
                )

            from ..ops.mul_cuda_ext import mul_forward as _cuda_mul

            device_index = int(getattr(self.device, "index", 0) or 0)
            out = _cuda_mul(self, other_t, device=device_index, sync=True)

            # Autograd (same rule as CPU)
            req = self._result_requires_grad(self, other_t)
            out.requires_grad = bool(req)  # ensure flag matches graph needs

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

    def copy_from_numpy(self, arr) -> None:
        """
        Copy data from a NumPy array (or array-like / scalar) into this tensor.

        Backward compatibility (CPU)
        ----------------------------
        The original implementation accepted any array-like input (including NumPy
        scalars like np.float32) by calling `np.asarray(arr, dtype=np.float32)`.
        This method preserves that behavior.

        CUDA
        ----
        - Accepts any array-like / scalar.
        - Casts to `self.dtype`, makes it C-contiguous, then HtoD memcpy into the
        tensor's existing (or newly allocated) device buffer.

        Parameters
        ----------
        arr : Any
            Array-like object accepted by `np.asarray`, including NumPy scalars.

        Raises
        ------
        ValueError
            If shape mismatch.
        RuntimeError
            If device unsupported or CUDA copy fails.
        """
        import numpy as np

        # IMPORTANT: preserve legacy behavior:
        # - accept np.float32, python float, lists, etc.
        # - materialize as ndarray
        dt = np.dtype(self.dtype)  # default float32 in your codebase
        arr_nd = np.asarray(arr, dtype=dt)

        # Preserve original strict shape check (including scalar shape == ())
        if arr_nd.shape != self._shape:
            raise ValueError(
                f"Shape mismatch: tensor {self._shape} vs array {arr_nd.shape}"
            )

        # -----------------------
        # CPU path (original semantics)
        # -----------------------
        if self._device.is_cpu():
            self._data[...] = arr_nd
            return

        # -----------------------
        # CUDA path
        # -----------------------
        if self._device.is_cuda():
            # Ensure device buffer exists for destination
            if int(self.data) == 0:
                self._ensure_cuda_alloc(dtype=dt)

            dst_dev = int(self.data)
            if dst_dev == 0 and arr_nd.nbytes != 0:
                raise RuntimeError(
                    "CUDA copy_from_numpy: destination devptr is 0 after allocation"
                )

            # Make sure host buffer is contiguous
            x_c = np.ascontiguousarray(arr_nd)
            nbytes = int(x_c.nbytes)
            if nbytes == 0:
                return

            # Use ops-layer memcpy (HtoD)
            from ..ops.pool2d_cuda import cuda_set_device
            from ..ops.memcpy_cuda import memcpy_htod as _memcpy_htod

            lib = self._get_cuda_lib()
            cuda_set_device(lib, int(self._device.index or 0))

            _memcpy_htod(
                lib,
                dst_dev=int(dst_dev),
                src_host=x_c,
                nbytes=nbytes,
                sync=True,
            )
            return

        raise RuntimeError(f"Unsupported device type: {self._device!r}")

    def copy_from(self, other: "Tensor", *, allow_cross_device: bool = False) -> None:
        """
        Copy data from another tensor into this tensor (in-place).

        Parameters
        ----------
        other : Tensor
            Source tensor.
        allow_cross_device : bool, default False
            If False (backward-compatible), require `self.device` and `other.device`
            to match exactly (string compare) and perform same-device copies only.

            If True, allow:
            - CPU -> CPU (same as before)
            - CUDA -> CUDA (D2D memcpy; same device index is recommended)
            - CPU -> CUDA (HtoD memcpy)
            - CUDA -> CPU (DtoH memcpy)

            Notes
            -----
            - Shape must match.
            - dtype must match (no implicit casting).
            - Cross-GPU copies (cuda:0 -> cuda:1) are not handled here; they may work
            only if your memcpy wrapper supports peer copies. By default this
            method raises for different CUDA device indices.

        Raises
        ------
        TypeError, ValueError, RuntimeError
        """
        import numpy as np

        if not isinstance(other, Tensor):
            raise TypeError(f"copy_from expects a Tensor, got {type(other)!r}")

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        dt_self = np.dtype(self.dtype)
        dt_other = np.dtype(other.dtype)
        if dt_self != dt_other:
            raise TypeError(f"dtype mismatch in copy_from: {dt_self} vs {dt_other}")

        # Backward-compatible default: exact device match required.
        if not allow_cross_device:
            if str(self.device) != str(other.device):
                raise RuntimeError(
                    f"Device mismatch in copy_from: {self.device} vs {other.device}"
                )
            # Same-device copy paths below will handle CPU or CUDA.
        else:
            # If both are CUDA, be conservative about cross-GPU unless you explicitly support it.
            if self.device.is_cuda() and other.device.is_cuda():
                if int(self.device.index or 0) != int(other.device.index or 0):
                    raise RuntimeError(
                        f"Cross-GPU copy_from not supported: {other.device} -> {self.device}"
                    )

        # -----------------------
        # CPU <- CPU
        # -----------------------
        if self.device.is_cpu() and other.device.is_cpu():
            self._data[...] = other._data
            return

        # Common sizes
        nbytes = int(self.numel()) * int(dt_self.itemsize)
        if nbytes == 0:
            return

        # -----------------------
        # CUDA <- CUDA (D2D)
        # -----------------------
        if self.device.is_cuda() and other.device.is_cuda():
            src_dev = int(other.data)
            if src_dev == 0 and other.numel() != 0:
                raise RuntimeError(
                    "CUDA copy_from: source tensor has no allocated devptr (data == 0)"
                )

            if int(self.data) == 0:
                self._ensure_cuda_alloc(dtype=dt_self)

            dst_dev = int(self.data)
            if dst_dev == 0 and self.numel() != 0:
                raise RuntimeError(
                    "CUDA copy_from: destination tensor has no allocated devptr (data == 0)"
                )

            from ..ops.pool2d_cuda import cuda_set_device
            from ..ops.memcpy_cuda import memcpy_dtod as _memcpy_dtod

            lib = self._get_cuda_lib()
            cuda_set_device(lib, int(self.device.index or 0))

            _memcpy_dtod(
                lib,
                dst_dev=int(dst_dev),
                src_dev=int(src_dev),
                nbytes=int(nbytes),
                sync=True,
            )
            return

        # -----------------------
        # CUDA <- CPU (HtoD)
        # -----------------------
        if self.device.is_cuda() and other.device.is_cpu():
            if not allow_cross_device:
                # Should be unreachable because we already enforced device match,
                # but keep it explicit and clear.
                raise RuntimeError(
                    f"Device mismatch in copy_from: {other.device} -> {self.device}"
                )

            if int(self.data) == 0:
                self._ensure_cuda_alloc(dtype=dt_self)

            dst_dev = int(self.data)
            if dst_dev == 0 and self.numel() != 0:
                raise RuntimeError(
                    "CUDA copy_from (HtoD): destination tensor has no allocated devptr (data == 0)"
                )

            # Make host buffer contiguous for memcpy
            x_host = np.ascontiguousarray(other._data, dtype=dt_self)

            from ..ops.pool2d_cuda import cuda_set_device
            from ..ops.memcpy_cuda import memcpy_htod as _memcpy_htod

            lib = self._get_cuda_lib()
            cuda_set_device(lib, int(self.device.index or 0))

            _memcpy_htod(
                lib,
                dst_dev=int(dst_dev),
                src_host=x_host,
                nbytes=int(nbytes),
                sync=True,
            )
            return

        # -----------------------
        # CPU <- CUDA (DtoH)
        # -----------------------
        if self.device.is_cpu() and other.device.is_cuda():
            if not allow_cross_device:
                raise RuntimeError(
                    f"Device mismatch in copy_from: {other.device} -> {self.device}"
                )

            src_dev = int(other.data)
            if src_dev == 0 and other.numel() != 0:
                raise RuntimeError(
                    "CUDA copy_from (DtoH): source tensor has no allocated devptr (data == 0)"
                )

            # Ensure destination host buffer exists (it should on CPU tensors).
            # If your Tensor can be CPU with _data None, allocate here:
            if getattr(self, "_data", None) is None:
                self._data = np.empty(self.shape, dtype=dt_self)

            from ..ops.pool2d_cuda import cuda_set_device
            from ..ops.memcpy_cuda import memcpy_dtoh as _memcpy_dtoh

            lib = self._get_cuda_lib()
            cuda_set_device(lib, int(other.device.index or 0))

            _memcpy_dtoh(
                lib,
                dst_host=self._data,
                src_dev=int(src_dev),
                nbytes=int(nbytes),
                sync=True,
            )
            return

        raise RuntimeError(f"Unsupported device copy: {other.device} -> {self.device}")

    @staticmethod
    def full(
        shape: tuple,
        fill_value: float,
        *,
        device: Device,
        requires_grad: bool = False,
    ) -> "Tensor":
        """
        Create a tensor filled with a constant value.

        Backward compatibility
        ----------------------
        - CPU path preserves the original logic exactly:
        uses NumPy to create a filled array and calls copy_from_numpy().
        - Default dtype remains float32, matching the original implementation.
        - The returned tensor has ctx=None.

        CUDA
        ----
        - Allocates a CUDA tensor (no host staging) and fills via Tensor.fill(),
        which dispatches to the CUDA fill kernel after ensuring allocation.

        Parameters
        ----------
        shape
            Desired tensor shape. May be any shape accepted by NumPy (including
            `()` for a scalar tensor).
        fill_value : float
            Constant value to write into every element.
        device
            Target device placement (CPU or CUDA).
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
            If `device` is not a supported device type.
        """
        import numpy as np

        # -----------------------
        # CPU path (unchanged)
        # -----------------------
        if device.is_cpu():
            arr = np.full(shape, fill_value, dtype=np.float32)
            t = Tensor(
                shape=arr.shape,
                device=device,
                requires_grad=requires_grad,
                ctx=None,
                dtype=np.float32,
            )
            t.copy_from_numpy(arr)
            return t

        # -----------------------
        # CUDA path
        # -----------------------
        if device.is_cuda():
            # Normalize shape using NumPy semantics (so shape=() becomes scalar)
            arr_shape = np.empty(shape, dtype=np.float32).shape

            t = Tensor(
                shape=arr_shape,
                device=device,
                requires_grad=requires_grad,
                ctx=None,
                dtype=np.float32,
            )
            # Uses your already refactored fill(): ensures alloc + CUDA kernel fill
            t.fill(float(fill_value))
            return t

        raise RuntimeError(f"full is not supported for device={device!r}")

    def reshape(self, new_shape: tuple[int, ...]) -> "Tensor":
        """
        Return a reshaped view of this tensor.

        Notes
        -----
        - CPU: uses NumPy reshape (view when possible).
        - CUDA: no kernel needed; returns a metadata-only view that shares the same
        device pointer (no D2D copy).
        - Backward: reshapes grad_out back to the original tensor shape.
        - Validates reshape by preserving total number of elements, matching NumPy
        semantics (including -1 inference).
        """
        import numpy as np

        # ---------- helpers ----------
        def _numel(shape: tuple[int, ...]) -> int:
            n = 1
            for d in shape:
                n *= int(d)
            return int(n)

        def _normalize_shape(shape_like) -> tuple[int, ...]:
            if isinstance(shape_like, tuple):
                return tuple(int(x) for x in shape_like)
            # allow list/int etc. if you want to keep old behavior
            return tuple(int(x) for x in tuple(shape_like))

        new_shape = _normalize_shape(new_shape)

        # Validate reshape using NumPy semantics (supports -1)
        # We avoid allocating large arrays; reshape on a tiny dummy of same numel.
        src_numel = int(self.numel()) if hasattr(self, "numel") else _numel(self.shape)

        try:
            # For zero-numel tensors, NumPy reshape rules still apply.
            # Use a 1D dummy with the same number of elements.
            dummy = np.empty((src_numel,), dtype=np.int8)
            _ = dummy.reshape(new_shape)  # just to validate / infer -1
            # If -1 existed, reshape() above resolves it; get the resolved shape:
            resolved = dummy.reshape(new_shape).shape
            new_shape_resolved = tuple(int(x) for x in resolved)
        except Exception as e:
            raise ValueError(f"Invalid reshape from {self.shape} to {new_shape}") from e

        req = self.requires_grad

        # -----------------------
        # CPU path (preserve intent; now avoids unnecessary copy if you already store ndarray)
        # -----------------------
        if self.device.is_cpu():
            src_np = self.to_numpy()
            reshaped_np = src_np.reshape(new_shape_resolved)

            out = Tensor(
                shape=reshaped_np.shape,
                device=self.device,
                requires_grad=req,
                ctx=None,
                dtype=getattr(self, "dtype", np.float32),
            )
            # keep existing semantics (copy_from_numpy). If you have a true view mode,
            # you can switch to sharing storage, but for backward-compat keep copy.
            out.copy_from_numpy(reshaped_np)

            if req:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cpu():
                        raise RuntimeError(
                            "grad_out must be CPU in current implementation"
                        )

                    g_out_np = grad_out.to_numpy()
                    grad_parent_np = g_out_np.reshape(self.shape)

                    grad_parent = Tensor(
                        shape=self.shape,
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                        dtype=getattr(self, "dtype", np.float32),
                    )
                    grad_parent.copy_from_numpy(grad_parent_np)
                    return (grad_parent,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                out._set_ctx(ctx)

            return out

        # -----------------------
        # CUDA path (no kernel needed: pointer alias)
        # -----------------------
        if self.device.is_cuda():
            # Must have an allocated pointer if numel != 0.
            # (If numel == 0, allow data==0 and just propagate.)
            if int(self.data) == 0 and src_numel != 0:
                raise RuntimeError(
                    "CUDA reshape requires allocated device buffer (data == 0)."
                )

            # Create an alias tensor that shares the same devptr.
            # Prefer your existing internal constructor used in tests.
            if hasattr(Tensor, "_from_devptr"):
                out = Tensor._from_devptr(
                    dev_ptr=int(self.data),
                    shape=new_shape_resolved,
                    device=self.device,
                    requires_grad=req,
                    ctx=None,
                    dtype=np.dtype(self.dtype),
                )
            else:
                # Fallback: if you do not have _from_devptr, you need an equivalent API.
                raise RuntimeError(
                    "CUDA reshape requires Tensor._from_devptr (or equivalent)"
                )

            if req:

                def backward_fn(grad_out: "Tensor"):
                    # Reshape grad_out back to parent shape without copying
                    if str(grad_out.device) != str(self.device):
                        raise ValueError(
                            "reshape backward expects grad_out on same device"
                        )

                    # grad_out might be unallocated for empty tensors; keep consistent.
                    if int(grad_out.data) == 0 and src_numel != 0:
                        raise RuntimeError(
                            "grad_out has no allocated devptr (data == 0)"
                        )

                    if hasattr(Tensor, "_from_devptr"):
                        grad_parent = Tensor._from_devptr(
                            dev_ptr=int(grad_out.data),
                            shape=self.shape,
                            device=self.device,
                            requires_grad=False,
                            ctx=None,
                            dtype=np.dtype(self.dtype),
                        )
                    else:
                        raise RuntimeError(
                            "CUDA reshape backward requires Tensor._from_devptr (or equivalent)"
                        )
                    return (grad_parent,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("reshape")

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
        CPU behavior
        ------------
        - Uses NumPy for the forward kernel on CPU tensors.

        CUDA behavior (workaround)
        --------------------------
        - For CUDA tensors, this method currently performs a CPU round-trip:
        device -> host (to_numpy) -> NumPy log -> device (copy_from_numpy).
        - This preserves correctness and autograd semantics, but is not optimized.

        Autograd
        --------
        If `self.requires_grad` is True, the returned tensor participates in
        autograd with the backward rule:

            d(log(x)) / dx = 1 / x

        The behavior for non-positive input values follows NumPy semantics
        (e.g., `-inf` or `nan`).

        TODO
        ----
        Implement a native CUDA kernel for log (and a fused backward) to avoid
        device<->host transfers.
        """
        # ============================================================
        # CPU path (KEEP EXACT SEMANTICS)
        # ============================================================
        if self.device.is_cpu():
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

        # ============================================================
        # CUDA path (CPU workaround)
        # ============================================================
        if self.device.is_cuda():
            # Forward: D2H -> NumPy -> H2D
            x_np = self.to_numpy()  # should be a CPU ndarray even if self is CUDA
            y_np = np.log(x_np).astype(np.float32, copy=False)

            out = Tensor(
                shape=self.shape,
                device=self.device,
                requires_grad=self.requires_grad,
                ctx=None,
            )
            # Ensure output has a device buffer; dtype should match your tensor dtype.
            out._ensure_cuda_alloc(dtype=np.dtype(getattr(self, "dtype", np.float32)))
            out.copy_from_numpy(y_np)

            if self.requires_grad:

                def backward_fn(grad_out: "Tensor"):
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA log backward"
                        )
                    if str(grad_out.device) != str(self.device):
                        raise RuntimeError(
                            f"grad_out device mismatch: expected {self.device!r}, got {grad_out.device!r}"
                        )
                    # d/dx log(x) = 1/x
                    return (grad_out / self,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                ctx.saved_meta["cuda_workaround"] = True
                ctx.saved_meta["op"] = "log"
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("log")
        raise RuntimeError("Unreachable")

    def sqrt(self) -> "Tensor":
        """
        Elementwise square root.

        Returns
        -------
        Tensor
            A tensor with the same shape as `self`, containing sqrt(self) elementwise.

        Notes
        -----
        CPU behavior
        ------------
        - Uses NumPy for the forward kernel on CPU tensors.

        CUDA behavior (workaround)
        --------------------------
        - For CUDA tensors, this method currently performs a CPU round-trip:
        device -> host (to_numpy) -> NumPy sqrt -> device (copy_from_numpy).
        - This preserves correctness and autograd semantics, but is not optimized.

        Autograd
        --------
        If `self.requires_grad` is True, attaches a Context with parent (self,).

        TODO
        ----
        Implement a native CUDA kernel for sqrt (and optionally a fused backward) to
        avoid device<->host transfers.
        """
        # ============================================================
        # CPU path (KEEP EXACT SEMANTICS)
        # ============================================================
        if self.device.is_cpu():
            x_np = self.to_numpy()
            y_np = np.sqrt(x_np).astype(np.float32, copy=False)

            out = Tensor(
                shape=self.shape, device=self.device, requires_grad=self.requires_grad
            )
            out.copy_from_numpy(y_np)

            if self.requires_grad:

                def backward_fn(grad_out: "Tensor") -> Sequence[Optional["Tensor"]]:
                    if not grad_out.device.is_cpu():
                        grad_out._raise_device_not_supported("sqrt_backward")

                    go = grad_out.to_numpy().astype(np.float32, copy=False)
                    y = out.to_numpy().astype(np.float32, copy=False)

                    gx_np = (go * (0.5 / y)).astype(np.float32, copy=False)

                    gx = Tensor(
                        shape=self.shape, device=self.device, requires_grad=False
                    )
                    gx.copy_from_numpy(gx_np)
                    return (gx,)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                ctx.save_for_backward(out)
                out._set_ctx(ctx)

            return out

        # ============================================================
        # CUDA path (CPU workaround)
        # ============================================================
        if self.device.is_cuda():
            # Forward: D2H -> NumPy -> H2D
            x_np = self.to_numpy()
            y_np = np.sqrt(x_np).astype(np.float32, copy=False)

            out = Tensor(
                shape=self.shape,
                device=self.device,
                requires_grad=self.requires_grad,
                ctx=None,
            )
            out._ensure_cuda_alloc(dtype=np.dtype(getattr(self, "dtype", np.float32)))
            out.copy_from_numpy(y_np)

            if self.requires_grad:

                def backward_fn(grad_out: "Tensor") -> Sequence[Optional["Tensor"]]:
                    if not grad_out.device.is_cuda():
                        raise RuntimeError(
                            "grad_out must be CUDA for CUDA sqrt backward"
                        )
                    if str(grad_out.device) != str(self.device):
                        raise RuntimeError(
                            f"grad_out device mismatch: expected {self.device!r}, got {grad_out.device!r}"
                        )

                    # d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / out
                    return (grad_out * (0.5 / out),)

                ctx = Context(parents=(self,), backward_fn=backward_fn)
                ctx.saved_meta["cuda_workaround"] = True
                ctx.saved_meta["op"] = "sqrt"
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("sqrt")
        raise RuntimeError("Unreachable")

    def to(self, device, *, copy: bool = False) -> "Tensor":
        import numpy as np
        from src.keydnn.domain.device._device import Device

        if isinstance(device, str):
            device = Device(device)

        # Same-device shortcut
        if str(device) == str(self.device):
            return self if not copy else self.clone()

        dtype = np.dtype(getattr(self, "dtype", np.float32))
        numel = self._numel_from_shape()
        nbytes = int(numel) * int(dtype.itemsize)

        # CPU -> CUDA
        if self.device.is_cpu() and device.is_cuda():
            from ..native_cuda.python.ops import memcpy_ctypes as mc
            import numpy as np

            out = Tensor(shape=self.shape, device=device, requires_grad=False, ctx=None)
            out._ensure_cuda_alloc(dtype=dtype)

            lib = out._get_cuda_lib()

            # IMPORTANT: ensure contiguous host buffer before raw memcpy
            host = np.ascontiguousarray(self.to_numpy(), dtype=dtype)

            nbytes = int(host.size) * int(host.dtype.itemsize)
            if nbytes > 0:
                mc.memcpy_htod(
                    lib,
                    dst_dev=int(out.data),
                    src_host=host,
                    nbytes=nbytes,
                    sync=True,
                )
            return out

        # CUDA -> CPU
        if self.device.is_cuda() and device.is_cpu():
            from ..native_cuda.python.ops import memcpy_ctypes as mc

            out = Tensor(shape=self.shape, device=device, requires_grad=False, ctx=None)
            host = np.empty(self.shape, dtype=dtype)  # contiguous
            lib = self._get_cuda_lib()

            self._ensure_cuda_alloc(dtype=dtype)
            if nbytes > 0:
                mc.memcpy_dtoh(
                    lib, dst_host=host, src_dev=int(self.data), nbytes=nbytes, sync=True
                )

            out.copy_from_numpy(host)
            return out

        # CUDA -> CUDA (different device index)
        if self.device.is_cuda() and device.is_cuda():
            # simplest safe path for now: D2H then H2D
            return self.to("cpu", copy=True).to(device, copy=True)

        self._raise_device_not_supported("to")

    def clone(self) -> "Tensor":
        """
        Deep copy of tensor data into a new Tensor.

        - CPU: copies the underlying NumPy array.
        - CUDA: allocates a new device buffer and performs D2D memcpy.

        Notes
        -----
        - clone() returns a tensor with requires_grad=False and no ctx by default
        (typical framework behavior for cloning raw storage). If you want to
        preserve requires_grad, you can add a flag later.
        """
        out = Tensor(
            shape=self.shape, device=self.device, requires_grad=False, ctx=None
        )

        # -------------------------
        # CPU path
        # -------------------------
        if self.device.is_cpu():
            out.copy_from_numpy(self.to_numpy().copy())
            return out

        # -------------------------
        # CUDA path
        # -------------------------
        if self.device.is_cuda():
            from ..native_cuda.python.ops import memcpy_ctypes as mc

            y = Tensor(
                shape=self.shape, device=self.device, requires_grad=False, ctx=None
            )
            y._ensure_cuda_alloc(dtype=getattr(self, "dtype", np.float32))

            self._ensure_cuda_alloc(dtype=getattr(self, "dtype", np.float32))

            lib = self._get_cuda_lib()

            dtype = getattr(self, "dtype", np.float32)
            itemsize = int(np.dtype(dtype).itemsize)

            numel = 1
            for d in self.shape:
                numel *= int(d)

            nbytes = int(numel * itemsize)

            if nbytes > 0:
                mc.memcpy_dtod(
                    lib,
                    dst_dev=int(y.data),
                    src_dev=int(self.data),
                    nbytes=nbytes,
                    sync=True,
                )

            return y

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

        CUDA behavior
        -------------
        - Uses the native CUDA unary exp kernel via `unary_cuda_ext.exp_forward`.
        - Operates directly on device pointers (no NumPy round-trip).
        """
        import numpy as np

        # -------------------------
        # CPU path (NumPy)
        # -------------------------
        if self.device.is_cpu():
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

        # -------------------------
        # CUDA path (native kernel)
        # -------------------------
        if self.device.is_cuda():
            # exp_forward returns requires_grad=False by design; we attach ctx below if needed.
            from ..ops.unary_cuda_ext import exp_forward as _exp_forward

            out = _exp_forward(
                self, device=self.device.index if hasattr(self.device, "index") else 0
            )

            # Preserve autograd participation
            out.requires_grad = (
                self.requires_grad
            )  # if your Tensor allows attribute set

            if self.requires_grad:
                ctx = Context(
                    parents=(self,),
                    backward_fn=lambda grad_out: (grad_out * out,),
                )
                out._set_ctx(ctx)

            return out

        self._raise_device_not_supported("exp")
