"""
Device-specific implementations of Tensor.sum using control-path dispatch.

This module defines CUDA and CPU implementations of the Tensor.sum reduction,
registered via the tensor_control_path_manager. At runtime, the appropriate
implementation is selected based on the tensor's device.

The implementations preserve the public Tensor.sum API while specializing
the forward and backward computation strategies for each backend.
"""

from typing import Optional

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinReduction as TMR


@tensor_control_path_manager(TMR, TMR.sum, Device("cuda:0"))
def sum(self: ITensor, axis: Optional[int] = None, keepdims: bool = False) -> "ITensor":
    """
    CUDA implementation of Tensor.sum.

    Computes the sum of elements in a CUDA tensor, either across all elements
    or along a specified axis. The forward and backward passes are implemented
    using native CUDA reduction kernels.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on a CUDA device.
    axis : int or None, optional
        Axis along which to compute the sum. If None, all elements are reduced
        to a scalar. Defaults to None.
    keepdims : bool, optional
        Whether to retain reduced dimensions in the output tensor.
        Defaults to False.

    Returns
    -------
    ITensor
        Tensor containing the summed values. The shape depends on `axis`
        and `keepdims`.

    Raises
    ------
    TypeError
        If `axis` is not an integer or None.
    ValueError
        If `axis` is out of bounds for the input tensor.
    NotImplementedError
        If the requested reduction mode is not yet supported on CUDA
        (e.g., non-2D axis reductions, or missing reshape support).

    Notes
    -----
    - Full-tensor reduction (`axis=None`) produces a scalar CUDA tensor.
    - Axis-based reduction is currently implemented for 2D tensors only,
      matching the available CUDA kernels.
    - Backward propagation relies on CUDA kernels that either fill gradients
      uniformly or scatter gradients back along the reduced axis.
    """
    import numpy as np

    Tensor = type(self)

    # -----------------------
    # CUDA path
    # -----------------------

    # NOTE: keep current dtype policy (float32) until Tensor tracks dtype end-to-end
    dtype = np.float32  # TODO: replace with self.dtype if you track it

    # Normalize/validate axis
    axis_ = None
    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError("axis must be int or None")
        ndim = len(self.shape)
        axis_ = axis if axis >= 0 else ndim + axis
        if axis_ < 0 or axis_ >= ndim:
            raise ValueError(f"axis {axis} out of bounds for ndim {ndim}")

    # Tensor-boundary CUDA reduce wrappers (no direct ctypes usage here)
    from ....ops.reduce_cuda_ext import (
        sum_all_forward as _sum_all_forward,
        sum_backward_fill_forward as _sum_backward_fill_forward,
        sum_axis2d_forward as _sum_axis2d_forward,
        sum_axis2d_backward as _sum_axis2d_backward,
    )

    # -----------------------
    # forward
    # -----------------------
    if axis_ is None:
        # returns a CUDA scalar tensor (shape=())
        out = _sum_all_forward(self, device=0, sync=True)

        if keepdims:
            # best-effort reshape to (1,1,...) to match CPU semantics
            new_shape = tuple(1 for _ in self.shape)
            if hasattr(out, "reshape") and callable(getattr(out, "reshape")):
                out = out.reshape(new_shape)  # type: ignore[call-arg]
            else:
                # If reshape isn't implemented for CUDA tensors in this repo yet,
                # keepdims=True cannot be represented without a metadata-only view.
                raise NotImplementedError(
                    "keepdims=True for CUDA sum_all requires Tensor.reshape support"
                )

        if self.requires_grad:

            def backward_fn(grad_out: "ITensor"):
                """
                Backward function for full-tensor CUDA sum.

                Propagates the upstream scalar gradient uniformly to all input
                elements by filling a flat gradient buffer and reshaping it
                to the input tensor's shape.

                Parameters
                ----------
                grad_out : ITensor
                    Gradient of the output tensor. Must be a CUDA tensor.

                Returns
                -------
                tuple[ITensor]
                    A single-element tuple containing the gradient with respect
                    to the input tensor.
                """
                if not grad_out.device.is_cuda():
                    raise RuntimeError(
                        "grad_out must be CUDA for CUDA Tensor.sum backward"
                    )

                # grad_out is scalar (or keepdims scalar-view) on CUDA.
                # Fill a flat grad buffer then reshape to input shape.
                numel = int(self.numel())
                gx_flat = _sum_backward_fill_forward(
                    grad_out, numel=numel, device=0, sync=True
                )

                if hasattr(gx_flat, "reshape") and callable(
                    getattr(gx_flat, "reshape")
                ):
                    gx = gx_flat.reshape(self.shape)  # type: ignore[call-arg]
                else:
                    raise NotImplementedError(
                        "CUDA sum backward requires Tensor.reshape support"
                    )

                return (gx,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            ctx.saved_meta["axis"] = axis
            ctx.saved_meta["keepdims"] = keepdims
            out._set_ctx(ctx)

        return out

    # axis != None: currently support 2D only (matches native CUDA reduce kernels)
    if len(self.shape) != 2:
        raise NotImplementedError(
            "Tensor.sum(axis=...) for CUDA is currently implemented for 2D tensors only."
        )

    rows, cols = int(self.shape[0]), int(self.shape[1])

    # forward output as a 1D CUDA tensor: (cols,) for axis=0, (rows,) for axis=1
    out_base = _sum_axis2d_forward(self, axis=int(axis_), device=0, sync=True)

    # keepdims reshape
    if keepdims:
        kd_shape = (1, cols) if axis_ == 0 else (rows, 1)
        if hasattr(out_base, "reshape") and callable(getattr(out_base, "reshape")):
            out = out_base.reshape(kd_shape)  # type: ignore[call-arg]
        else:
            raise NotImplementedError(
                "keepdims=True for CUDA sum(axis=...) requires Tensor.reshape support"
            )
    else:
        out = out_base

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor"):
            """
            Backward function for axis-based CUDA sum on 2D tensors.

            Scatters upstream gradients back to the input tensor along the
            reduced axis, handling keepdims semantics and device-to-device
            memory copies when necessary.

            Parameters
            ----------
            grad_out : ITensor
                Gradient of the output tensor. Must be a CUDA tensor.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect
                to the input tensor.
            """
            device_index: int = grad_out.device.index
            if not grad_out.device.is_cuda():
                raise RuntimeError("grad_out must be CUDA for CUDA Tensor.sum backward")

            go = grad_out

            # For keepdims=True, upstream grad arrives as (1, cols) or (rows, 1),
            # but reduce_cuda_ext.sum_axis2d_backward expects (cols,) / (rows,).
            if keepdims:
                if axis_ == 0:
                    expected = (1, cols)
                    squeezed_shape = (cols,)
                    n = cols
                else:
                    expected = (rows, 1)
                    squeezed_shape = (rows,)
                    n = rows

                if tuple(int(d) for d in go.shape) != expected:
                    raise ValueError(
                        f"grad_out shape mismatch: expected {expected} for axis={axis_}, "
                        f"got {tuple(go.shape)}"
                    )

                # Allocate a 1D CUDA buffer and memcpy device-to-device
                import ctypes

                from ....ops.pool2d_cuda import (
                    _load_cuda_lib,
                    cuda_set_device,
                    cuda_malloc,
                )

                lib = _load_cuda_lib()
                cuda_set_device(lib, 0)

                nbytes = int(n) * int(np.dtype(dtype).itemsize)
                go_dev = int(cuda_malloc(lib, int(nbytes)))

                from ....tensor._cuda_storage import _CudaStorage

                storage_gd = _CudaStorage(
                    lib=lib,
                    device_index=device_index,
                    dev_ptr=int(go_dev),
                    nbytes=int(nbytes),
                    dtype=np.dtype(dtype),
                )

                # Prefer dedicated wrapper if present; else bind DLL symbol.
                try:
                    from ....native_cuda.python.ops import memcpy_ctypes as mc

                    if hasattr(mc, "cuda_memcpy_d2d"):
                        mc.cuda_memcpy_d2d(
                            lib,
                            dst_dev=int(go_dev),
                            src_dev=int(go.data),
                            nbytes=int(nbytes),
                        )
                    else:
                        raise AttributeError
                except Exception:
                    if not hasattr(lib, "keydnn_cuda_memcpy_d2d"):
                        raise RuntimeError(
                            "Missing device-to-device memcpy: expected "
                            "`memcpy_ctypes.cuda_memcpy_d2d` or DLL symbol "
                            "`keydnn_cuda_memcpy_d2d`."
                        )

                    fn = lib.keydnn_cuda_memcpy_d2d
                    fn.argtypes = [
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                        ctypes.c_size_t,
                    ]
                    fn.restype = ctypes.c_int

                    st = int(
                        fn(
                            ctypes.c_uint64(int(go_dev)),
                            ctypes.c_uint64(int(go.data)),
                            ctypes.c_size_t(int(nbytes)),
                        )
                    )
                    if st != 0:
                        raise RuntimeError(
                            f"keydnn_cuda_memcpy_d2d failed with status={st}"
                        )

                go = Tensor._from_storage(
                    storage_gd,
                    shape=squeezed_shape,
                    dtype=np.dtype(dtype),
                    device=self.device,
                    requires_grad=False,
                )

            gx = _sum_axis2d_backward(
                go,
                rows=rows,
                cols=cols,
                axis=int(axis_),
                device=0,
                sync=True,
            )
            return (gx,)

        ctx = Context(parents=(self,), backward_fn=backward_fn)
        ctx.saved_meta["axis"] = axis
        ctx.saved_meta["keepdims"] = keepdims
        out._set_ctx(ctx)

    return out


@tensor_control_path_manager(TMR, TMR.sum, Device("cpu"))
def sum(self: ITensor, axis: Optional[int] = None, keepdims: bool = False) -> "ITensor":
    """
    CPU implementation of Tensor.sum.

    Computes the sum of elements in a tensor using NumPy, either across all
    elements or along a specified axis. This implementation mirrors standard
    NumPy semantics and serves as the reference behavior for Tensor.sum.

    Parameters
    ----------
    self : ITensor
        Input tensor residing on the CPU.
    axis : int or None, optional
        Axis along which to compute the sum. If None, all elements are reduced
        to a scalar. Defaults to None.
    keepdims : bool, optional
        Whether to retain reduced dimensions in the output tensor.
        Defaults to False.

    Returns
    -------
    ITensor
        Tensor containing the summed values.

    Raises
    ------
    TypeError
        If `axis` is not an integer or None.
    ValueError
        If `axis` is out of bounds for the input tensor.

    Notes
    -----
    - The forward pass delegates to NumPy's `np.sum`.
    - The backward pass propagates gradients by broadcasting the upstream
      gradient to the input tensor's shape.
    """
    import numpy as np

    Tensor = type(self)

    # -----------------------
    # CPU path (unchanged)
    # -----------------------

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

    out = Tensor(shape=out_shape, device=self.device, requires_grad=self.requires_grad)
    out.copy_from_numpy(np.asarray(value, dtype=np.float32))

    if self.requires_grad:

        def backward_fn(grad_out: "ITensor"):
            """
            Backward function for CPU Tensor.sum.

            Propagates the upstream gradient to all input elements by
            broadcasting it to the input tensor's shape.

            Parameters
            ----------
            grad_out : ITensor
                Gradient of the output tensor. Must be a CPU tensor.

            Returns
            -------
            tuple[ITensor]
                A single-element tuple containing the gradient with respect
                to the input tensor.
            """
            if not grad_out.device.is_cpu():
                raise RuntimeError("grad_out must be CPU in current implementation")

            g = np.asarray(grad_out.to_numpy(), dtype=np.float32)

            if axis is None:
                grad_np = np.ones(self.shape, dtype=np.float32) * float(np.asarray(g))
            else:
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
