"""
2D transpose implementations (CPU and CUDA) for KeyDNN tensors.

This module provides backend-specific implementations of `Tensor.transpose()`
and registers them via `tensor_control_path_manager`:

- `tensor_transpose_cpu`: computes a 2D transpose using NumPy and materializes
  the result as a new CPU tensor.
- `tensor_transpose_gpu`: computes a 2D transpose using a CUDA kernel through
  the ops-layer wrapper `transpose2d_cuda`.

Autograd
--------
If `self.requires_grad` is True, both implementations attach a `Context` with a
backward rule that transposes the upstream gradient:

    out = xᵀ  =>  dL/dx = (dL/dout)ᵀ

Notes
-----
- Transpose is restricted to 2D tensors in these implementations.
- The CUDA path requires that the input tensor already has an allocated device
  buffer (`self.data != 0` for non-empty tensors) and allocates the output and
  gradient buffers explicitly.
"""

from ..._tensor_builder import tensor_control_path_manager
from ..._tensor_context import Context

from .....domain.device._device import Device
from .....domain._tensor import ITensor

from ._base import TensorMixinMemory as TMM
from typing import Union, Type


Number = Union[int, float]
"""Scalar types accepted by Tensor arithmetic operators."""


@tensor_control_path_manager(TMM, TMM.transpose, Device("cuda:0"))
def tensor_transpose_gpu(self: ITensor) -> "ITensor":
    """
    Transpose a 2D CUDA tensor using a native CUDA kernel.

    This function implements `Tensor.transpose()` for CUDA tensors registered on
    `"cuda:0"`. It allocates a new output tensor of shape `(cols, rows)` and
    launches a CUDA transpose kernel.

    Returns
    -------
    ITensor
        A new CUDA tensor containing the transposed data.

    Raises
    ------
    ValueError
        If the input tensor is not 2D.
    RuntimeError
        If the input has no allocated device buffer, or if output/gradient device
        allocations fail.

    Notes
    -----
    - Input must be contiguous in the sense expected by the underlying kernel.
    - When `requires_grad=True`, the backward rule computes `grad_x = grad_outᵀ`
      by launching the same transpose kernel with swapped dimensions.
    """
    Tensor = type(self)

    if len(self.shape) != 2:
        raise ValueError(f"transpose requires a 2D tensor, got shape={self.shape}")

    r, c = self.shape
    req = self.requires_grad

    # -----------------------
    # CUDA path
    # -----------------------
    import numpy as np
    from ....ops.transpose_cuda import transpose2d_cuda  # ops-layer wrapper

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
        raise RuntimeError("CUDA transpose failed to allocate output device buffer")

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

        def backward_fn(grad_out: "ITensor"):
            """
            Backprop for transpose: grad_x = grad_outᵀ (CUDA).

            Parameters
            ----------
            grad_out : ITensor
                Upstream gradient w.r.t. the transposed output. Must be a CUDA
                tensor with shape `(c, r)`.

            Returns
            -------
            tuple[ITensor]
                A one-tuple containing the gradient w.r.t. the input tensor,
                with shape `(r, c)`.

            Raises
            ------
            RuntimeError
                If `grad_out` is not CUDA or lacks an allocated device buffer.
            ValueError
                If `grad_out.shape` does not match `(c, r)`.
            """
            if not grad_out.device.is_cuda():
                raise RuntimeError("grad_out must be CUDA for CUDA transpose backward")
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


@tensor_control_path_manager(TMM, TMM.transpose, Device("cpu"))
def tensor_transpose_cpu(self: ITensor) -> "ITensor":
    """
    Transpose a 2D CPU tensor using NumPy and materialize the result.

    This function implements `Tensor.transpose()` for CPU tensors. It uses NumPy's
    `.T` to compute the transpose and copies the result into a newly allocated
    output tensor.

    Returns
    -------
    ITensor
        A new CPU tensor containing the transposed data.

    Raises
    ------
    ValueError
        If the input tensor is not 2D.

    Notes
    -----
    - Current CPU implementation materializes a copy (not a view).
    - When `requires_grad=True`, the backward rule computes `grad_x = grad_outᵀ`
      using NumPy transpose on the upstream gradient.
    """
    Tensor = type(self)

    if len(self._shape) != 2:
        raise ValueError(f"transpose requires a 2D tensor, got shape={self.shape}")

    r, c = self.shape
    req = self.requires_grad

    # -----------------------
    # CPU path (unchanged)
    # -----------------------
    out = Tensor(shape=(c, r), device=self.device, requires_grad=req, ctx=None)
    out.copy_from_numpy(self.to_numpy().T)

    if req:

        def backward_fn(grad_out: "ITensor"):
            """
            Backprop for transpose: grad_x = grad_outᵀ (CPU).

            Parameters
            ----------
            grad_out : ITensor
                Upstream gradient w.r.t. the transposed output. Must be a CPU
                tensor with shape `(c, r)`.

            Returns
            -------
            tuple[ITensor]
                A one-tuple containing the gradient w.r.t. the input tensor,
                with shape `(r, c)`.

            Raises
            ------
            RuntimeError
                If `grad_out` is not on CPU in the current implementation.
            ValueError
                If `grad_out.shape` does not match `(c, r)`.
            """
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
