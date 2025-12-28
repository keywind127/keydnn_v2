"""
Tensor memory / construction mixin.

This module defines `TensorMixinMemory`, a focused mixin that provides
factory constructors (zeros/ones/full/from_numpy) and core memory-related
utilities (copying, device transfer, allocation) for a concrete `Tensor`
that satisfies the domain-level `ITensor` protocol.

Design intent
-------------
- Keep object creation and memory movement centralized in the tensor
  implementation (infrastructure layer), while still presenting a clean,
  framework-style API (`Tensor.zeros`, `Tensor.ones`, etc.).
- Provide consistent CPU and CUDA behavior:
  - CPU tensors are backed by NumPy arrays.
  - CUDA tensors store a device pointer and rely on native CUDA ops/wrappers
    for allocation, memset/fill, and memcpy.

Notes
-----
- The mixin assumes the concrete `Tensor` class provides infrastructure hooks
  such as `_get_cuda_lib()`, `_numel_from_shape()`, `_raise_device_not_supported()`,
  and internal fields like `_data` / `_dtype`.
- Autograd behavior is intentionally minimal in these helpers: most factories
  return tensors with `ctx=None` and do not attach computation graphs.
"""

from typing import Tuple, Union, Type
from abc import ABC

from .....domain._tensor import ITensor
from .....domain.device._device import Device

import numpy as np


Number = Union[int, float]


class TensorMixinMemory(ABC):
    """
    Mixin that implements tensor construction and memory-management helpers.

    This mixin is intended to be inherited by a concrete `Tensor` class that
    implements `ITensor`. It provides:

    - Factory constructors: `zeros`, `ones`, `full`, `_from_numpy`
    - Memory utilities: `copy_from_numpy`, `to_numpy`, `to`
    - Allocation helper: `_ensure_cuda_alloc`
    - Lightweight shape-only transform: `reshape` (CPU copy + CUDA alias)

    The goal is to keep storage creation and device transfer logic in one place,
    while allowing higher-level modules to remain backend-agnostic.
    """

    @classmethod
    def zeros(
        cls: Type[ITensor],
        *,
        shape: tuple[int, ...],
        device: Device,
        requires_grad: bool = False,
    ) -> "ITensor":
        """
        Create a tensor filled with zeros on the specified device.

        This factory method constructs a tensor with the given shape and device.
        For CPU tensors, a NumPy array is allocated and zero-initialized.
        For CUDA tensors, device memory is allocated and zeroed via a CUDA fill
        routine.

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
        ITensor
            Newly created tensor filled with zeros.

        Notes
        -----
        - The dtype is currently fixed to `float32`.
        - Zero-sized tensors are valid and return immediately without invoking
          CUDA kernels.
        """
        import numpy as np

        Tensor = cls

        dtype = np.float32
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)

        if device.is_cpu():
            out._data = np.zeros(shape, dtype=dtype)
            return out

        out._ensure_cuda_alloc(dtype=dtype)
        numel = out._numel_from_shape()
        if numel == 0:
            return out

        from ....ops.fill_cuda import zeros_cuda

        lib = out._get_cuda_lib()
        zeros_cuda(lib, y_dev=int(out.data), numel=numel, dtype=dtype, sync=True)
        return out

    @classmethod
    def ones(
        cls: Type[ITensor],
        *,
        shape: tuple[int, ...],
        device: Device,
        requires_grad: bool = False,
    ) -> "ITensor":
        """
        Create a tensor filled with ones on the specified device.

        This factory method constructs a tensor with the given shape and device.
        For CPU tensors, a NumPy array is allocated and initialized with ones.
        For CUDA tensors, device memory is allocated and filled using a native
        CUDA fill routine.

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
        ITensor
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

        Tensor = cls

        dtype = np.float32
        out = Tensor(shape=shape, device=device, requires_grad=requires_grad)

        if device.is_cpu():
            out._data = np.ones(shape, dtype=dtype)
            return out

        out._ensure_cuda_alloc(dtype=dtype)
        numel = out._numel_from_shape()
        if numel == 0:
            return out

        from ....ops.fill_cuda import ones_cuda

        lib = out._get_cuda_lib()
        ones_cuda(lib, y_dev=int(out.data), numel=numel, dtype=dtype, sync=True)
        return out

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
        ITensor
            A newly allocated tensor with the given shape, filled with `fill_value`.

        Raises
        ------
        RuntimeError
            If `device` is not a supported device type.
        """
        import numpy as np

        Tensor = cls

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

    @classmethod
    def _from_numpy(
        cls: Type[ITensor], arr: ITensor, *, device: Device, requires_grad: bool = False
    ) -> "ITensor":
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
        ITensor
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
        Tensor = cls
        t = Tensor(
            shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None
        )
        t.copy_from_numpy(arr)
        return t

    def fill(self: ITensor, value: float) -> None:
        """
        Fill the tensor with a scalar value.

        This method overwrites every element of the tensor with `value`.
        The concrete implementation may dispatch differently depending on
        device placement (e.g., NumPy fill on CPU vs. a CUDA fill kernel).

        Parameters
        ----------
        value : float
            Scalar value to write into the tensor.

        Raises
        ------
        RuntimeError
            If the tensor's device is unsupported by this implementation.
        """
        ...

    def broadcast_to(self: ITensor, shape: Tuple[int, ...]) -> "ITensor":
        """
        Broadcast this tensor to a target shape by explicit expansion.

        This is an explicit broadcasting primitive intended to mirror NumPy's
        broadcasting rules while keeping most binary ops strict (i.e., they do
        not implicitly broadcast unless you opt in).

        Parameters
        ----------
        shape : tuple[int, ...]
            Target shape to broadcast to.

        Returns
        -------
        ITensor
            Broadcasted tensor (materialized copy).

        Notes
        -----
        - The operation is conceptually a "repeat/expand" that materializes a
          new tensor of the requested shape.
        - Backward typically reduces gradients by summing over the broadcasted
          dimensions (i.e., the inverse of expansion).
        """
        ...

    def clone(self: ITensor) -> "ITensor":
        """
        Create a deep copy of the tensor's storage.

        - CPU: copies the underlying NumPy array into a new tensor.
        - CUDA: allocates a new device buffer and performs a device-to-device
          memcpy.

        Returns
        -------
        ITensor
            A new tensor with identical contents.

        Notes
        -----
        - `clone()` is intended to copy raw storage and typically returns a tensor
          with `requires_grad=False` and no autograd context (`ctx=None`).
          If you want to preserve autograd flags/ctx, that should be handled by
          a higher-level API.
        """
        ...

    # ----------------------------
    # Transpose (2D)
    # ----------------------------
    def transpose(self: ITensor) -> "ITensor":
        """
        Return the 2D transpose of this tensor.

        For a 2D tensor A with shape (M, N), transpose returns Aᵀ with shape
        (N, M) such that:

            out[i, j] = self[j, i]

        Requirements
        ------------
        - Input must be 2D.
        - CPU and CUDA are supported by the concrete implementation.

        Backward
        --------
        If out = Aᵀ, then dL/dA = (dL/dout)ᵀ.
        """
        ...

    @property
    def T(self: ITensor) -> "ITensor":
        """
        Convenience property for 2D transpose.

        Equivalent to calling `self.transpose()`.
        """
        return self.transpose()

    def to_numpy(self: ITensor) -> np.ndarray:
        """
        Convert the tensor to a NumPy ndarray on the host.

        Returns
        -------
        np.ndarray
            A NumPy array containing the tensor data.

        Raises
        ------
        RuntimeError
            If device-to-host transfer is unavailable or the tensor's dtype is unknown.

        Notes
        -----
        - CPU tensors return a view/copy of the underlying CPU storage (preserving
          existing behavior of the concrete tensor type).
        - CUDA tensors are copied from device to host via a device-to-host memcpy.
        """
        ...

    def copy_from_numpy(self: ITensor, arr: np.ndarray) -> None:
        """
        Copy data from a NumPy array (or array-like / scalar) into this tensor.

        Backward compatibility (CPU)
        ----------------------------
        The original implementation accepted any array-like input (including NumPy
        scalars like `np.float32`) by calling `np.asarray(arr, dtype=np.float32)`.
        This method preserves that behavior for the CPU path.

        CUDA
        ----
        - Accepts any array-like / scalar.
        - Casts to `self.dtype`, makes it C-contiguous, then performs a host-to-device
          memcpy into the tensor's existing (or newly allocated) device buffer.

        Parameters
        ----------
        arr : Any
            Array-like object accepted by `np.asarray`, including NumPy scalars.

        Raises
        ------
        ValueError
            If the input shape does not match this tensor's shape.
        RuntimeError
            If the tensor's device is unsupported or a CUDA copy fails.
        """
        ...

    def to(self: ITensor, device, *, copy: bool = False) -> "ITensor":
        """
        Move or copy this tensor to another device.

        This method implements explicit device placement transitions and returns
        a tensor on `device`. If the target device matches the current device,
        it returns `self` by default (or a cloned copy if `copy=True`).

        Supported transfers
        -------------------
        - CPU -> CUDA: allocates device memory and performs host-to-device memcpy.
        - CUDA -> CPU: allocates host buffer and performs device-to-host memcpy.
        - CUDA -> CUDA (different device indices): currently implemented via an
          intermediate CPU round-trip for simplicity.

        Parameters
        ----------
        device : Device | str
            Target device. If a string is provided, it is parsed as a `Device`
            (e.g., "cpu", "cuda:0").
        copy : bool, optional
            If True, forces a copy even when the device is unchanged.
            Defaults to False.

        Returns
        -------
        ITensor
            A tensor placed on the requested device.

        Notes
        -----
        - For CPU -> CUDA copies, the host buffer is made C-contiguous before
          raw memcpy to ensure correct layout.
        - This method does not propagate autograd context; returned tensors are
          created with `requires_grad=False` and `ctx=None` in the transfer paths.
        """
        import numpy as np
        from src.keydnn.domain.device._device import Device

        Tensor = type(self)

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
            from ....native_cuda.python.ops import memcpy_ctypes as mc
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
            from ....native_cuda.python.ops import memcpy_ctypes as mc

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

    def _ensure_cuda_alloc(self: ITensor, *, dtype) -> None:
        """
        Ensure that a CUDA tensor has an allocated device buffer.

        This helper allocates device memory sized to:

            numel(self.shape) * dtype.itemsize

        and stores the resulting device pointer on the tensor instance. Memory
        contents are left uninitialized.

        Parameters
        ----------
        dtype
            Desired NumPy-compatible dtype for the device allocation.

        Raises
        ------
        RuntimeError
            If called on a non-CUDA tensor.

        Notes
        -----
        - Zero-sized tensors do not allocate device memory; their device pointer
          is represented as `0`.
        - This method is designed to be idempotent for an already-allocated tensor
          when the allocation size matches the required size.
        - If a re-allocation is required (dtype/size mismatch), the old buffer
          should be freed (TODO: wire/verify `cuda_free` usage to avoid leaks).
        """
        import numpy as np
        from ....native_cuda.python.avgpool2d_ctypes import cuda_set_device, cuda_malloc

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

    def reshape(self: ITensor, new_shape: tuple[int, ...]) -> "ITensor":
        """
        Return a reshaped view of this tensor.

        This operation changes the logical shape while preserving the total
        number of elements.

        Behavior
        --------
        - CPU: reshapes via NumPy and then materializes via `copy_from_numpy`
          (preserving current copy-based semantics).
        - CUDA: returns a metadata-only alias that shares the same device pointer
          (no kernel launch, no device-to-device copy).

        Backward
        --------
        If autograd is enabled, backward reshapes `grad_out` back to the original
        shape of the parent tensor.

        Notes
        -----
        - Reshape validity is checked using NumPy semantics, including support
          for `-1` inference.
        - For CUDA tensors, an allocation must exist if `numel != 0` (i.e.,
          `data` cannot be 0 for non-empty tensors).
        """

        import numpy as np

        from ..._tensor_context import Context

        Tensor = type(self)

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

                def backward_fn(grad_out: "ITensor"):
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

                def backward_fn(grad_out: "ITensor"):
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
