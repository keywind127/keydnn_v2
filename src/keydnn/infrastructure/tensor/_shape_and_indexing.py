"""
Tensor shape, indexing, and structural ops mixin (NumPy CPU backend).

This module defines `TensorShapeAndIndexingMixin`, a cohesive mixin that
implements shape-transforming and indexing-related Tensor methods for the
NumPy-backed concrete Tensor implementation.

Design notes
------------
- This mixin is intended to be inherited by the concrete `Tensor` class.
- To avoid circular imports, the implementation does not import `Tensor`
  directly; instead it constructs new tensors via `self.__class__` (instance
  methods) or `first.__class__` (staticmethods).
- Operations are CPU-only in the current backend; CUDA paths raise
  `DeviceNotSupportedError` via the host Tensor's `_raise_device_not_supported`.
- Autograd wiring is expressed by attaching a `Context` to output tensors, and
  storing backward rules as `backward_fn`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ...domain._tensor import ITensor
from ._tensor_context import Context


class TensorShapeAndIndexingMixin(ITensor):
    """
    Shape and indexing operations for the concrete Tensor implementation.

    This mixin groups together Tensor methods that primarily:
    - change the logical shape/structure of tensors (e.g., reshape, transpose),
    - compose tensors structurally (e.g., stack, concat, broadcast_to), and/or
    - provide indexing/slicing semantics (__getitem__).

    Notes
    -----
    - Methods assume the host class provides:
        - `.shape`, `.device`, `.requires_grad`
        - `.to_numpy()`, `.copy_from_numpy(...)`
        - `._set_ctx(...)`, `._raise_device_not_supported(...)`
        - a `Context` type compatible with the one shown in your snippet
    - New tensors are constructed via the host class (e.g., `self.__class__`).
    """

    def __getitem__(self, key: Any) -> "ITensor":
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

        TensorClass = self.__class__

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
        out = TensorClass(shape=out_shape, device=self.device, requires_grad=req)
        out.copy_from_numpy(sliced_arr)

        if req:

            def backward_fn(grad_out: "ITensor"):
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

                grad_parent = TensorClass(
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
    def stack(tensors: Sequence["ITensor"], axis: int = 0) -> "ITensor":
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

        TensorClass = first.__class__

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
        out = TensorClass(shape=stacked.shape, device=dev, requires_grad=req, ctx=None)
        out.copy_from_numpy(stacked)

        # ---- Backward ----
        if req:

            def backward_fn(grad_out: "ITensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy()  # shape: stacked.shape

                grads: list[Optional["ITensor"]] = []
                for i, t in enumerate(tensors):
                    if not t.requires_grad:
                        grads.append(None)
                        continue

                    # Select the i-th slice along the stacked axis.
                    # Use take() to avoid view complexities; it returns an array.
                    gi_np = np.take(g, indices=i, axis=axis).astype(
                        np.float32, copy=False
                    )

                    gi = TensorClass(
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

    def reshape(self, new_shape: tuple[int, ...]) -> "ITensor":
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

        TensorClass = self.__class__

        # NumPy reshape (may be a view)
        src_np = self.to_numpy()
        try:
            reshaped_np = src_np.reshape(new_shape)
        except Exception as e:
            raise ValueError(f"Invalid reshape from {self.shape} to {new_shape}") from e

        req = self.requires_grad
        out = TensorClass(
            shape=reshaped_np.shape,
            device=self.device,
            requires_grad=req,
            ctx=None,
        )
        out.copy_from_numpy(reshaped_np)

        if req:

            def backward_fn(grad_out: "ITensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g_out_np = grad_out.to_numpy()

                # Reshape gradient back to input shape
                grad_parent_np = g_out_np.reshape(self.shape)

                grad_parent = TensorClass(
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
    def concat(tensors: Sequence["ITensor"], axis: int = 0) -> "ITensor":
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

        TensorClass = first.__class__

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
        out = TensorClass(shape=out_np.shape, device=dev, requires_grad=req, ctx=None)
        out.copy_from_numpy(out_np)

        # Backward
        if req:
            # Build slice boundaries along axis
            # e.g. sizes [2,3,1] -> offsets [0,2,5,6]
            offsets = [0]
            for s in sizes_along_axis:
                offsets.append(offsets[-1] + int(s))

            def backward_fn(grad_out: "ITensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")

                g = grad_out.to_numpy()
                grads: list[Optional["ITensor"]] = []

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

                    gi = TensorClass(
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

    def broadcast_to(self, shape: tuple[int, ...]) -> "ITensor":
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

        TensorClass = self.__class__

        src = self.to_numpy()
        try:
            out_np = np.broadcast_to(src, shape).astype(np.float32, copy=False)
        except Exception as e:
            raise ValueError(f"Cannot broadcast shape {self.shape} to {shape}") from e

        req = self.requires_grad
        out = TensorClass(shape=shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(out_np)

        if req:
            src_shape = self.shape

            def backward_fn(grad_out: "ITensor"):
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

                grad = TensorClass(
                    shape=src_shape, device=self.device, requires_grad=False
                )
                grad.copy_from_numpy(g.astype(np.float32, copy=False))
                return (grad,)

            ctx = Context(parents=(self,), backward_fn=backward_fn)
            ctx.saved_meta["broadcast_from"] = src_shape
            ctx.saved_meta["broadcast_to"] = shape
            out._set_ctx(ctx)

        return out

    # ----------------------------
    # Transpose (2D)
    # ----------------------------
    def transpose(self) -> "ITensor":
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

        TensorClass = self.__class__

        r, c = self.shape
        req = self.requires_grad

        out = TensorClass(shape=(c, r), device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(self.to_numpy().T)

        if req:

            def backward_fn(grad_out: "ITensor"):
                if not grad_out.device.is_cpu():
                    raise RuntimeError("grad_out must be CPU in current implementation")
                if grad_out.shape != (c, r):
                    raise ValueError(
                        f"grad_out shape mismatch: expected {(c, r)}, got {grad_out.shape}"
                    )

                g_np = grad_out.to_numpy().T  # back to (r, c)
                grad_parent = TensorClass(
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
    def T(self) -> "ITensor":
        """
        Convenience property for 2D transpose.
        """
        return self.transpose()
