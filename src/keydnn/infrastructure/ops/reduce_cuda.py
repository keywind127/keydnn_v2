# infrastructure/ops/reduce_cuda.py
"""
CUDA reduce operation wrappers for KeyDNN (infrastructure layer).

This module provides small, Python-level wrapper functions around KeyDNN's
native CUDA reduction kernels exposed via ctypes. Each wrapper:

- validates arguments (dtype, axis) at the Python boundary,
- normalizes pointer/shape inputs to plain `int` for ctypes calls,
- optionally performs post-call synchronization via `cuda_synchronize(lib)`.

Implemented ops
--------------
- sum_all_cuda / mean_all_cuda:
    Reduce a contiguous 1D device buffer into a device scalar.
- sum_backward_fill_cuda / mean_backward_fill_cuda:
    Backward "fill" kernels that broadcast a scalar grad_out into grad_x.
- max_axis2d_forward_cuda / max_axis2d_backward_cuda:
    2D max reduction with argmax indices along axis {0, 1}, plus backward scatter.
- sum_axis2d_forward_cuda / sum_axis2d_backward_cuda:
    2D sum reduction along axis {0, 1}, plus backward broadcast.
- sum_to_shape_cuda:
    General "unbroadcast" reduction that reduces a flat input buffer of shape
    `in_shape` into an output buffer of shape `out_shape` by summing along
    broadcasted axes (out dim == 1 and in dim > 1).

Notes
-----
- These wrappers intentionally avoid changing kernel semantics; they only add
  boundary checks and optional synchronization.
- Some underlying ctypes functions do not accept a `sync=` keyword; this module
  enforces a consistent `sync` API at the wrapper layer where applicable.
"""

from __future__ import annotations

import numpy as np

from ..native_cuda.python.global_avgpool2d_ctypes import cuda_synchronize
from ..native_cuda.python.maxpool2d_ctypes import cuda_memset


class _CudaReduceOps:
    """
    Namespace marker for CUDA reduce wrappers.

    This class is not intended to be instantiated. It exists to document and
    group the reduce-related CUDA wrapper functions in this module. All actual
    APIs are provided as module-level functions for minimal call overhead and
    compatibility with existing imports.
    """

    pass


def sum_all_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Sum all elements of a 1D contiguous device buffer into a device scalar.

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    x_dev : int
        Device pointer to the input buffer of length `numel` (contiguous 1D).
    y_dev : int
        Device pointer to the output scalar buffer (1 element).
    numel : int
        Number of elements in the input buffer.
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.

    Notes
    -----
    - The underlying ctypes wrapper `reduce_ctypes.sum_all_cuda` does not accept
      a `sync=` keyword, so synchronization is handled here conditionally.
    """
    from ..native_cuda.python.reduce_ctypes import sum_all_cuda as _sum

    if dtype not in (np.float32, np.float64):
        raise TypeError(f"sum_all_cuda supports float32/float64 only, got {dtype}")

    # NOTE: reduce_ctypes.sum_all_cuda does NOT accept sync=
    _sum(lib, x_dev=int(x_dev), y_dev=int(y_dev), numel=int(numel), dtype=dtype)

    if sync:
        cuda_synchronize(lib)


def mean_all_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Mean-reduce all elements of a 1D contiguous device buffer into a device scalar.

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    x_dev : int
        Device pointer to the input buffer of length `numel` (contiguous 1D).
    y_dev : int
        Device pointer to the output scalar buffer (1 element).
    numel : int
        Number of elements in the input buffer.
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import mean_all_cuda as _mean

    if dtype not in (np.float32, np.float64):
        raise TypeError(f"mean_all_cuda supports float32/float64 only, got {dtype}")

    _mean(lib, x_dev=int(x_dev), y_dev=int(y_dev), numel=int(numel), dtype=dtype)

    if sync:
        cuda_synchronize(lib)


def sum_backward_fill_cuda(
    lib,
    *,
    grad_out_dev: int,
    grad_x_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Backward fill kernel for `sum_all_cuda`: broadcast scalar grad into grad_x.

    This computes:

        grad_x[i] = grad_out   for i in [0, numel)

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    grad_out_dev : int
        Device pointer to upstream gradient scalar (1 element).
    grad_x_dev : int
        Device pointer to output gradient buffer for x (length `numel`).
    numel : int
        Number of elements in `grad_x`.
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import sum_backward_fill_cuda as _fill

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"sum_backward_fill_cuda supports float32/float64 only, got {dtype}"
        )

    _fill(
        lib,
        grad_out_dev=int(grad_out_dev),
        grad_x_dev=int(grad_x_dev),
        numel=int(numel),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def mean_backward_fill_cuda(
    lib,
    *,
    grad_out_dev: int,
    grad_x_dev: int,
    numel: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Backward fill kernel for `mean_all_cuda`: broadcast scaled scalar into grad_x.

    This computes:

        grad_x[i] = grad_out / numel   for i in [0, numel)

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    grad_out_dev : int
        Device pointer to upstream gradient scalar (1 element).
    grad_x_dev : int
        Device pointer to output gradient buffer for x (length `numel`).
    numel : int
        Number of elements in `grad_x` (used to scale the gradient).
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import mean_backward_fill_cuda as _fill

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"mean_backward_fill_cuda supports float32/float64 only, got {dtype}"
        )

    _fill(
        lib,
        grad_out_dev=int(grad_out_dev),
        grad_x_dev=int(grad_x_dev),
        numel=int(numel),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def max_axis2d_forward_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    idx_dev: int,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Compute 2D max reduction and argmax indices along a specified axis.

    Given an input matrix `x` with shape (rows, cols), this computes:

    - `y`: the maximum values along `axis`
    - `idx`: the argmax indices along `axis`

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    x_dev : int
        Device pointer to input matrix buffer (rows * cols elements).
    y_dev : int
        Device pointer to output buffer for max values.
        - If axis == 0: shape (cols,)
        - If axis == 1: shape (rows,)
    idx_dev : int
        Device pointer to output buffer for argmax indices (same shape as `y`).
        Indices are along the reduced dimension.
    rows : int
        Number of rows in the input matrix.
    cols : int
        Number of columns in the input matrix.
    axis : int
        Reduction axis. Must be 0 (reduce rows) or 1 (reduce cols).
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    ValueError
        If `axis` is not 0 or 1.
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import max_axis2d_forward_cuda as _fwd

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"max_axis2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    _fwd(
        lib,
        x_dev=int(x_dev),
        y_dev=int(y_dev),
        idx_dev=int(idx_dev),
        rows=int(rows),
        cols=int(cols),
        axis=int(axis),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def max_axis2d_backward_cuda(
    lib,
    *,
    grad_out_dev: int,
    idx_dev: int,
    grad_x_dev: int,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
    zero_grad_x: bool = False,
    sync: bool = True,
) -> None:
    """
    Backward scatter for `max_axis2d_forward_cuda` using stored argmax indices.

    This scatters `grad_out` back into `grad_x` at the argmax locations recorded
    in `idx`. The native kernel uses **accumulating scatter** semantics:

        grad_x[argmax_pos] += grad_out_element

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    grad_out_dev : int
        Device pointer to upstream gradient buffer (same shape as the forward `y`):
        - If axis == 0: shape (cols,)
        - If axis == 1: shape (rows,)
    idx_dev : int
        Device pointer to argmax indices produced by the forward pass.
    grad_x_dev : int
        Device pointer to output gradient buffer for x (rows * cols elements).
    rows : int
        Number of rows in the original input matrix.
    cols : int
        Number of columns in the original input matrix.
    axis : int
        Reduction axis used in the forward pass. Must be 0 or 1.
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    zero_grad_x : bool, optional
        If True, zeroes `grad_x` (byte-wise memset) before scattering, effectively
        providing overwrite-like behavior for a fresh gradient buffer. Defaults to False.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    ValueError
        If `axis` is not 0 or 1.
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.

    Notes
    -----
    - The memset performed when `zero_grad_x=True` matches the byte-wise pattern
      used elsewhere in the CUDA pool ops wrappers.
    """
    from ..native_cuda.python.reduce_ctypes import max_axis2d_backward_cuda as _bwd

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"max_axis2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    if zero_grad_x:
        # byte-wise memset, same pattern as pool ops
        cuda_memset(
            lib, int(grad_x_dev), 0, int(rows) * int(cols) * np.dtype(dtype).itemsize
        )

    _bwd(
        lib,
        grad_out_dev=int(grad_out_dev),
        idx_dev=int(idx_dev),
        grad_x_dev=int(grad_x_dev),
        rows=int(rows),
        cols=int(cols),
        axis=int(axis),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def sum_axis2d_forward_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Compute 2D sum reduction along a specified axis.

    Given an input matrix `x` with shape (rows, cols), this computes:

    - axis == 1: y[r] = sum_c x[r, c]  -> y shape (rows,)
    - axis == 0: y[c] = sum_r x[r, c]  -> y shape (cols,)

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    x_dev : int
        Device pointer to input matrix buffer (rows * cols elements).
    y_dev : int
        Device pointer to output buffer for reduced sums:
        - If axis == 0: shape (cols,)
        - If axis == 1: shape (rows,)
    rows : int
        Number of rows in the input matrix.
    cols : int
        Number of columns in the input matrix.
    axis : int
        Reduction axis. Must be 0 (reduce rows) or 1 (reduce cols).
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    ValueError
        If `axis` is not 0 or 1.
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import sum_axis2d_forward_cuda as _fwd

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"sum_axis2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    _fwd(
        lib,
        x_dev=int(x_dev),
        y_dev=int(y_dev),
        rows=int(rows),
        cols=int(cols),
        axis=int(axis),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


def sum_axis2d_backward_cuda(
    lib,
    *,
    grad_out_dev: int,
    grad_x_dev: int,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
    sync: bool = True,
) -> None:
    """
    Backward broadcast for `sum_axis2d_forward_cuda`.

    This broadcasts `grad_out` back into `grad_x`:

    - axis == 1: grad_x[r, c] = grad_out[r]
    - axis == 0: grad_x[r, c] = grad_out[c]

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    grad_out_dev : int
        Device pointer to upstream gradient buffer (shape depends on axis):
        - If axis == 0: shape (cols,)
        - If axis == 1: shape (rows,)
    grad_x_dev : int
        Device pointer to output gradient buffer for x (rows * cols elements).
    rows : int
        Number of rows in the original input matrix.
    cols : int
        Number of columns in the original input matrix.
    axis : int
        Reduction axis used in the forward pass. Must be 0 or 1.
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call by invoking
        `cuda_synchronize(lib)`. Defaults to True.

    Raises
    ------
    ValueError
        If `axis` is not 0 or 1.
    TypeError
        If `dtype` is not `np.float32` or `np.float64`.
    """
    from ..native_cuda.python.reduce_ctypes import sum_axis2d_backward_cuda as _bwd

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"sum_axis2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    _bwd(
        lib,
        grad_out_dev=int(grad_out_dev),
        grad_x_dev=int(grad_x_dev),
        rows=int(rows),
        cols=int(cols),
        axis=int(axis),
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


# -----------------------------------------------------------------------------
# NEW: sum_to_shape
# -----------------------------------------------------------------------------


def sum_to_shape_cuda(
    lib,
    *,
    x_dev: int,
    y_dev: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: np.dtype,
    sync: bool = True,
    zero_y: bool = True,
) -> None:
    """
    Reduce `x` from `in_shape` into `y` of `out_shape` by summing over broadcasted axes.

    This is the inverse of broadcast: it sums `x` over any axis where:
        out_shape[d] == 1 and in_shape[d] != 1

    Parameters
    ----------
    lib : object
        Loaded native CUDA library handle passed through to ctypes wrappers.
    x_dev : int
        Device pointer to the input buffer of length prod(in_shape) (flattened).
    y_dev : int
        Device pointer to the output buffer of length prod(out_shape) (flattened).
    in_shape : tuple[int, ...]
        Logical input shape (rank R). Must match how `x_dev` was produced.
    out_shape : tuple[int, ...]
        Target reduced shape (rank R). For each dim d:
            out_shape[d] == in_shape[d] OR out_shape[d] == 1
    dtype : np.dtype
        Element dtype. Only `np.float32` and `np.float64` are supported.
    sync : bool, optional
        If True, synchronizes the CUDA device after the kernel call. Defaults True.
    zero_y : bool, optional
        If True, memset `y` to zero before calling the kernel. This is recommended
        if the kernel uses atomic accumulation into `y` (common implementation).
        Defaults True.

    Raises
    ------
    ValueError
        If rank mismatches or shape is incompatible with sum-to-shape semantics.
    TypeError
        If dtype is unsupported.
    """
    from ..native_cuda.python.reduce_ctypes import sum_to_shape_cuda as _sum_to_shape

    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError(f"sum_to_shape_cuda supports float32/float64 only, got {dtype}")

    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)

    if len(in_shape) != len(out_shape):
        raise ValueError(
            f"rank mismatch: in_shape rank={len(in_shape)} out_shape rank={len(out_shape)}"
        )

    # Validate compatibility (out dim must be 1 or equal to in dim)
    for i, (id_, od_) in enumerate(zip(in_shape, out_shape)):
        if id_ < 0 or od_ < 0:
            raise ValueError(f"negative dim at axis {i}: in={id_} out={od_}")
        if od_ != 1 and od_ != id_:
            raise ValueError(
                f"incompatible shapes at axis {i}: in_shape={in_shape} out_shape={out_shape}"
            )

    if zero_y:
        nbytes_y = int(np.prod(out_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        # allow empty outputs: memset(0 bytes) is fine, but your wrapper might not like it.
        if nbytes_y > 0:
            cuda_memset(lib, int(y_dev), 0, int(nbytes_y))

    # NOTE: reduce_ctypes.sum_to_shape_cuda does NOT accept sync=; handled here
    _sum_to_shape(
        lib,
        x_dev=int(x_dev),
        y_dev=int(y_dev),
        in_shape=in_shape,
        out_shape=out_shape,
        dtype=dtype,
    )

    if sync:
        cuda_synchronize(lib)


__all__ = [
    "sum_all_cuda",
    "mean_all_cuda",
    "sum_backward_fill_cuda",
    "mean_backward_fill_cuda",
    "max_axis2d_forward_cuda",
    "max_axis2d_backward_cuda",
    "sum_axis2d_forward_cuda",
    "sum_axis2d_backward_cuda",
    "sum_to_shape_cuda",
]
