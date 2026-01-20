"""
infrastructure/ops/conv2d_cuda.py

Ops-layer CUDA Conv2D wrapper for KeyDNN (NumPy-facing API).

This module provides a higher-level, NumPy-friendly interface for running
KeyDNN's CUDA Conv2D kernels. It mirrors the CPU conv2d wrapper semantics while
delegating the heavy compute to native CUDA kernels via `ctypes`.

Architecture
------------
- Ops-layer responsibilities:
  - Validate shapes/dtypes and compute output shapes.
  - Perform CPU-side padding to match the CUDA kernel contract (expects x_pad).
  - Allocate and free temporary CUDA device buffers.
  - Transfer data H2D/D2H using existing memcpy wrappers.
  - Invoke low-level `ctypes` bindings for forward/backward kernels.
  - Optionally synchronize for accurate benchmarking / correctness boundaries.

Native bindings
---------------
- Forward:  `keydnn_cuda_conv2d_forward_f32/f64`
- Backward: `keydnn_cuda_conv2d_backward_f32/f64`

Conventions
-----------
- Input/Output layout:
  - x, y, grad_out, grad_x: NCHW  (N, C, H, W)
  - w, grad_w:             OIHW  (C_out, C_in, K_h, K_w)
- Dtypes:
  - Supported: float32, float64
  - Unsupported dtypes raise TypeError.

Notes
-----
- Padding is currently done on CPU (`np.pad`) and the padded tensor is copied to
  GPU. The CUDA kernel expects already-padded input.
- Backward computes grad_b on CPU (sum over N, H_out, W_out) to match the CPU
  semantics and keep the native kernel focused on grad_x_pad/grad_w.
- Device-range probing is a best-effort guard to catch obvious pointer/size
  mismatches when a `memcpy_d2h` export exists.

This module does not change global CUDA state beyond optionally setting the
device index via `cuda_set_device`.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional, Tuple

import numpy as np

from ..native_cuda.python.avgpool2d_ctypes import (
    cuda_malloc,
    cuda_free,
    cudaMemcpyHtoD,
    cudaMemcpyDtoH,
    cuda_synchronize,
    cuda_set_device,
)

from ..native_cuda.python.ops.conv2d_ctypes import (
    conv2d_forward_cuda as _conv2d_forward_ctypes,
    conv2d_backward_cuda as _conv2d_backward_ctypes,
)

from ..native_cuda.python.pad2d_cuda_ctypes import (
    pad2d_cuda,
    crop2d_cuda,
)


def _is_cdll(obj: object) -> bool:
    """
    Return True if `obj` is a `ctypes.CDLL` instance.

    This helper supports a pattern used across the ops layer where wrappers may
    be accessed as plain functions or as attributes on another object (bound
    method injection). In the bound case, an extra `self` argument may appear
    before the actual `ctypes.CDLL`.

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
        True iff `obj` is a `ctypes.CDLL`.
    """
    return isinstance(obj, ctypes.CDLL)


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    """
    Normalize an int-or-pair into a (h, w) tuple of ints.

    Parameters
    ----------
    v : int | (int, int)
        Scalar or pair value.

    Returns
    -------
    (int, int)
        A 2-tuple of ints.
    """
    return v if isinstance(v, tuple) else (int(v), int(v))


def _dtype_itemsize(dtype: np.dtype) -> int:
    """
    Return the byte size of a supported dtype.

    Parameters
    ----------
    dtype : np.dtype
        Requested dtype.

    Returns
    -------
    int
        Size in bytes of the dtype.

    Raises
    ------
    TypeError
        If dtype is not float32/float64.
    """
    dt = np.dtype(dtype)
    if dt not in (np.float32, np.float64):
        raise TypeError(f"conv2d_cuda supports float32/float64 only, got {dt}")
    return int(dt.itemsize)


def _probe_dev_range(lib: ctypes.CDLL, base_dev: int, nbytes_required: int) -> None:
    """
    Best-effort guard to validate a device pointer range looks readable.

    This helper attempts to detect obvious allocation/pointer mismatches by
    probing the last byte in the requested range using a device-to-host memcpy
    symbol, if it exists on the loaded library.

    Behavior
    --------
    - If `nbytes_required <= 0`, returns immediately.
    - If the library does not export `keydnn_cuda_memcpy_d2h`, this is a no-op.
    - Otherwise, it attempts to copy 1 byte from `base_dev + nbytes_required - 1`
      into a temporary host buffer.
    - A non-zero status raises a RuntimeError.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded KeyDNN CUDA native library.
    base_dev : int
        Base device pointer address.
    nbytes_required : int
        Number of bytes that must be readable starting at `base_dev`.

    Raises
    ------
    RuntimeError
        If the probe fails (non-zero status from the memcpy export).

    Notes
    -----
    This mirrors the behavior used by other ops wrappers (e.g., matmul) to
    provide a clearer early failure mode when device buffers are undersized or
    invalid.
    """
    if nbytes_required <= 0:
        return
    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        return

    fn = lib.keydnn_cuda_memcpy_d2h
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    fn.restype = ctypes.c_int

    tmp = (ctypes.c_ubyte * 1)()
    last_addr = int(base_dev) + int(nbytes_required) - 1
    st = int(
        fn(
            ctypes.cast(tmp, ctypes.c_void_p),
            ctypes.c_uint64(last_addr),
            ctypes.c_size_t(1),
        )
    )
    if st != 0:
        raise RuntimeError(
            f"device buffer too small or invalid: probe failed at 0x{last_addr:x} (status={st})"
        )


def conv2d_forward_cuda(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Run CUDA Conv2D forward using NumPy inputs (NCHW/OIHW) and return a NumPy output.

    This is the ops-layer wrapper around the low-level ctypes binding. It follows
    the CPU wrapper "spirit" (validate inputs, compute shapes, return NumPy
    output) while using the CUDA kernel for compute.

    Expected usage
    --------------
    y = conv2d_forward_cuda(
        lib,
        x=x,
        w=w,
        b=b,                    # optional
        stride=1,               # int or (int,int)
        padding=0,              # int or (int,int)
        device_index=0,         # optional
        sync=True,              # optional
        dtype=None,             # optional, inferred from x
    )

    Parameters (keywords)
    ---------------------
    lib : ctypes.CDLL
        CUDA native DLL handle. Passed positionally as the first argument
        (or second when accessed as a bound method).
    x : np.ndarray
        Input tensor of shape (N, C_in, H, W).
    w : np.ndarray
        Weight tensor of shape (C_out, C_in, K_h, K_w).
    b : Optional[np.ndarray]
        Bias vector of shape (C_out,) or None for no bias.
    stride : int | (int, int)
        Stride along height/width.
    padding : int | (int, int)
        Zero-padding applied symmetrically along height/width.
    device_index : int, optional
        CUDA device index to set before allocations/transfers.
        Alias: `device`.
    sync : bool, optional
        If True, synchronizes the device after kernel launch for correctness and
        benchmarking.
    dtype : np.dtype, optional
        Dtype to use. If None, inferred from `x.dtype`.

    Returns
    -------
    np.ndarray
        Output tensor `y` of shape (N, C_out, H_out, W_out).

    Raises
    ------
    TypeError
        For missing/extra kwargs, wrong types, or unsupported dtypes.
    ValueError
        For shape mismatches or invalid output dimension computation.
    RuntimeError
        For CUDA allocation/probe failures or non-zero kernel status.

    Notes
    -----
    - CPU padding is performed via `np.pad`; the CUDA kernel expects x_pad.
    - Temporary device buffers are allocated and freed internally.
    - Uses existing cudaMemcpy wrappers for transfers.
    """
    if not args:
        raise TypeError("conv2d_forward_cuda expected at least a lib argument")

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "conv2d_forward_cuda expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]
    if rest:
        raise TypeError(
            "conv2d_forward_cuda does not accept positional args beyond lib"
        )

    x = kwargs.pop("x", None)
    w = kwargs.pop("w", None)
    b = kwargs.pop("b", None)

    stride = kwargs.pop("stride", 1)
    padding = kwargs.pop("padding", 0)

    dtype = kwargs.pop("dtype", None)  # optional: infer from x if missing
    sync = kwargs.pop("sync", True)

    device_index = kwargs.pop("device_index", None)
    if device_index is None:
        device_index = kwargs.pop("device", None)

    if kwargs:
        # keep strict to avoid silently ignoring typos
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"conv2d_forward_cuda got unexpected kwargs: {extra}")

    if x is None or w is None:
        raise TypeError("conv2d_forward_cuda requires x and w")
    if not isinstance(x, np.ndarray) or not isinstance(w, np.ndarray):
        raise TypeError("x and w must be numpy arrays")

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    if dtype is None:
        dtype = x.dtype
    dtype = np.dtype(dtype)

    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"conv2d_forward_cuda supports float32/float64 only, got {dtype}"
        )

    x = x.astype(dtype, copy=False)
    w = w.astype(dtype, copy=False)
    b_arr: Optional[np.ndarray]
    if b is None:
        b_arr = None
    else:
        if not isinstance(b, np.ndarray):
            raise TypeError("b must be a numpy array or None")
        b_arr = b.astype(dtype, copy=False)

    x = np.ascontiguousarray(x)
    w = np.ascontiguousarray(w)
    if b_arr is not None:
        b_arr = np.ascontiguousarray(b_arr)

    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {x.shape}")
    if w.ndim != 4:
        raise ValueError(f"w must be 4D OIHW, got shape {w.shape}")

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")
    if b_arr is not None:
        if b_arr.ndim != 1 or b_arr.shape[0] != C_out:
            raise ValueError(
                f"bias shape mismatch: expected ({C_out},), got {b_arr.shape}"
            )

    H_out = (H + 2 * p_h - K_h) // s_h + 1
    W_out = (W + 2 * p_w - K_w) // s_w + 1
    if H_out < 0 or W_out < 0:
        raise ValueError(f"invalid output size: H_out={H_out}, W_out={W_out}")

    # Padded input dimensions (padding is done on GPU)
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    # ----------------------------
    # Debug asserts (shape invariants)
    # ----------------------------
    assert H_pad == H + 2 * p_h, "H_pad invariant violated"
    assert W_pad == W + 2 * p_w, "W_pad invariant violated"
    assert H_out == (H_pad - K_h) // s_h + 1, (
        f"H_out mismatch: got {H_out}, expected {(H_pad - K_h) // s_h + 1} "
        f"(H_pad={H_pad}, K_h={K_h}, s_h={s_h})"
    )
    assert W_out == (W_pad - K_w) // s_w + 1, (
        f"W_out mismatch: got {W_out}, expected {(W_pad - K_w) // s_w + 1} "
        f"(W_pad={W_pad}, K_w={K_w}, s_w={s_w})"
    )

    # Set device (optional)
    if device_index is not None:
        cuda_set_device(lib, int(device_index))

    # Allocate device buffers
    nbytes_x = int(x.nbytes)  # unpadded x
    nbytes_xpad = int(N * C_in * H_pad * W_pad * x.itemsize)
    nbytes_w = int(w.nbytes)
    nbytes_b = int(0 if b_arr is None else b_arr.nbytes)
    y = np.empty((N, C_out, H_out, W_out), dtype=dtype)
    nbytes_y = int(y.nbytes)

    x_unpad_dev = int(cuda_malloc(lib, nbytes_x if nbytes_x > 0 else 1))
    x_pad_dev = int(cuda_malloc(lib, nbytes_xpad if nbytes_xpad > 0 else 1))
    w_dev = int(cuda_malloc(lib, nbytes_w if nbytes_w > 0 else 1))
    y_dev = int(cuda_malloc(lib, nbytes_y if nbytes_y > 0 else 1))
    b_dev = 0

    try:
        # H2D copies (unpadded input)
        cudaMemcpyHtoD(lib, x_unpad_dev, x, nbytes_x)
        cudaMemcpyHtoD(lib, w_dev, w, nbytes_w)

        # Create padded x on device via pad2d CUDA kernel
        pad2d_cuda(
            lib,
            x_dev=x_unpad_dev,
            y_pad_dev=x_pad_dev,
            N=N,
            C=C_in,
            H=H,
            W=W,
            p_h=p_h,
            p_w=p_w,
            pad_value=0.0,
            dtype=dtype,
            device=(int(device_index) if device_index is not None else 0),
            sync=False,
        )

        # ----------------------------
        # Debug assert: pad2d matches NumPy exactly
        # (only do this when padding is non-trivial; avoids unnecessary overhead)
        # ----------------------------
        if (p_h > 0 or p_w > 0) and (N * C_in * H_pad * W_pad) > 0:
            x_pad_host = np.empty((N, C_in, H_pad, W_pad), dtype=dtype)
            cudaMemcpyDtoH(lib, x_pad_host, x_pad_dev, int(x_pad_host.nbytes))
            x_pad_np = np.pad(
                x,
                pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode="constant",
                constant_values=0.0,
            )
            assert x_pad_host.shape == x_pad_np.shape
            assert np.array_equal(
                x_pad_host, x_pad_np
            ), "pad2d_cuda mismatch vs np.pad (expected exact match)"

        if b_arr is not None:
            b_dev = int(cuda_malloc(lib, nbytes_b if nbytes_b > 0 else 1))
            cudaMemcpyHtoD(lib, b_dev, b_arr, nbytes_b)

        # Optional probes (catch obvious mismatched allocations)
        itemsize = _dtype_itemsize(dtype)
        _probe_dev_range(lib, x_pad_dev, N * C_in * H_pad * W_pad * itemsize)
        _probe_dev_range(lib, w_dev, C_out * C_in * K_h * K_w * itemsize)
        _probe_dev_range(
            lib, y_dev, N * C_out * max(H_out, 0) * max(W_out, 0) * itemsize
        )
        if b_arr is not None:
            _probe_dev_range(lib, b_dev, C_out * itemsize)

        _conv2d_forward_ctypes(
            lib,
            x_pad_dev=x_pad_dev,
            w_dev=w_dev,
            b_dev=(b_dev if b_arr is not None else None),
            y_dev=y_dev,
            N=N,
            C_in=C_in,
            H_pad=H_pad,
            W_pad=W_pad,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            dtype=dtype,
        )

        if sync:
            cuda_synchronize(lib)

        cudaMemcpyDtoH(lib, y, y_dev, nbytes_y)

        # ----------------------------
        # Debug assert: output is finite
        # ----------------------------
        assert np.isfinite(
            y
        ).all(), "conv2d_forward_cuda produced non-finite output (NaN/Inf)"

        return y

    finally:
        cuda_free(lib, x_unpad_dev)
        cuda_free(lib, x_pad_dev)
        cuda_free(lib, w_dev)
        cuda_free(lib, y_dev)
        if b_dev:
            cuda_free(lib, b_dev)


def conv2d_backward_cuda(
    *args: Any, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Run CUDA Conv2D backward and return (grad_x, grad_w, grad_b).

    This ops-layer wrapper computes gradients w.r.t. the unpadded input `x`,
    weights `w`, and optionally bias `b`. It delegates computation of grad_x_pad
    and grad_w to the native CUDA kernel, then unpads grad_x_pad to produce
    grad_x. Bias gradient is computed on CPU to match the CPU semantics.

    Expected usage
    --------------
    grad_x, grad_w, grad_b = conv2d_backward_cuda(
        lib,
        x=x,
        w=w,
        b=b,                    # optional
        grad_out=grad_out,
        stride=1,               # int or (int,int)
        padding=0,              # int or (int,int)
        device_index=0,         # optional
        sync=True,              # optional
        dtype=None,             # optional, inferred from x
    )

    Returns
    -------
    (grad_x, grad_w, grad_b)
        grad_x : np.ndarray
            Gradient w.r.t. input x, shape (N, C_in, H, W).
        grad_w : np.ndarray
            Gradient w.r.t. weights, shape (C_out, C_in, K_h, K_w).
        grad_b : Optional[np.ndarray]
            Gradient w.r.t. bias, shape (C_out,) if `b` is not None, else None.

    Raises
    ------
    TypeError
        For missing/extra kwargs, wrong types, or unsupported dtypes.
    ValueError
        For shape mismatches between x/w/grad_out.
    RuntimeError
        For CUDA allocation/probe failures or non-zero kernel status.

    Notes
    -----
    - The native backward kernel accumulates into grad_x_pad and grad_w, so we
      allocate zero-initialized host buffers and copy them to the device before
      launching the kernel.
    - grad_b is computed on CPU as `grad_out.sum(axis=(0,2,3))`.
    - Temporary device buffers are allocated and freed internally.
    """
    if not args:
        raise TypeError("conv2d_backward_cuda expected at least a lib argument")

    # Handle bound-method injection
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "conv2d_backward_cuda expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]
    if rest:
        raise TypeError(
            "conv2d_backward_cuda does not accept positional args beyond lib"
        )

    x = kwargs.pop("x", None)
    w = kwargs.pop("w", None)
    b = kwargs.pop("b", None)
    grad_out = kwargs.pop("grad_out", None)

    stride = kwargs.pop("stride", 1)
    padding = kwargs.pop("padding", 0)

    dtype = kwargs.pop("dtype", None)
    sync = kwargs.pop("sync", True)

    device_index = kwargs.pop("device_index", None)
    if device_index is None:
        device_index = kwargs.pop("device", None)

    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"conv2d_backward_cuda got unexpected kwargs: {extra}")

    if x is None or w is None or grad_out is None:
        raise TypeError("conv2d_backward_cuda requires x, w, grad_out")
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(w, np.ndarray)
        or not isinstance(grad_out, np.ndarray)
    ):
        raise TypeError("x, w, grad_out must be numpy arrays")

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    if dtype is None:
        dtype = x.dtype
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError(
            f"conv2d_backward_cuda supports float32/float64 only, got {dtype}"
        )

    x = x.astype(dtype, copy=False)
    w = w.astype(dtype, copy=False)
    grad_out = grad_out.astype(dtype, copy=False)

    b_arr: Optional[np.ndarray]
    if b is None:
        b_arr = None
    else:
        if not isinstance(b, np.ndarray):
            raise TypeError("b must be a numpy array or None")
        b_arr = b.astype(dtype, copy=False)

    x = np.ascontiguousarray(x)
    w = np.ascontiguousarray(w)
    grad_out = np.ascontiguousarray(grad_out)

    assert x.flags["C_CONTIGUOUS"]
    assert w.flags["C_CONTIGUOUS"]
    assert grad_out.flags["C_CONTIGUOUS"]  # backward only

    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {x.shape}")
    if w.ndim != 4:
        raise ValueError(f"w must be 4D OIHW, got shape {w.shape}")
    if grad_out.ndim != 4:
        raise ValueError(f"grad_out must be 4D NCHW, got shape {grad_out.shape}")

    N, C_in, H, W = x.shape
    C_out, C_in2, K_h, K_w = w.shape
    if C_in != C_in2:
        raise ValueError(f"in_channels mismatch: x has {C_in}, weight has {C_in2}")

    # Output dims come from grad_out
    N2, C_out2, H_out, W_out = grad_out.shape
    if N2 != N or C_out2 != C_out:
        raise ValueError(
            f"grad_out shape mismatch: expected (N={N}, C_out={C_out}, H_out, W_out), got {grad_out.shape}"
        )

    # Prepare padded input dimensions (same as forward), but padding is done on GPU.
    H_pad = H + 2 * p_h
    W_pad = W + 2 * p_w

    # ----------------------------
    # Debug asserts (shape invariants)
    # ----------------------------
    assert H_pad == H + 2 * p_h, "H_pad invariant violated"
    assert W_pad == W + 2 * p_w, "W_pad invariant violated"
    assert H_out == (H_pad - K_h) // s_h + 1, (
        f"H_out mismatch: grad_out has H_out={H_out}, expected {(H_pad - K_h) // s_h + 1} "
        f"(H_pad={H_pad}, K_h={K_h}, s_h={s_h})"
    )
    assert W_out == (W_pad - K_w) // s_w + 1, (
        f"W_out mismatch: grad_out has W_out={W_out}, expected {(W_pad - K_w) // s_w + 1} "
        f"(W_pad={W_pad}, K_w={K_w}, s_w={s_w})"
    )

    assert np.isfinite(grad_out).all(), "grad_out already contains NaN/Inf before conv2d backward"

    # Bias grad (CPU path) matches your CPU kernel semantics
    grad_b = None
    if b_arr is not None:
        grad_b = grad_out.sum(axis=(0, 2, 3)).astype(dtype, copy=False)

    # Host outputs
    grad_x = np.empty_like(x)
    grad_w = np.zeros_like(w)

    # Set device (optional)
    if device_index is not None:
        cuda_set_device(lib, int(device_index))

    # Allocate device buffers:
    # - x_unpad_dev: unpadded input x (H2D)
    # - x_pad_dev: padded input created via pad2d CUDA kernel
    # - grad_x_pad_dev: accumulated by native backward kernel
    # - grad_x_dev: cropped unpadded grad_x created via crop2d CUDA kernel
    x_unpad_dev = int(cuda_malloc(lib, int(x.nbytes) if int(x.nbytes) > 0 else 1))
    x_pad_bytes = int(N * C_in * H_pad * W_pad * x.itemsize)
    gx_pad_bytes = int(N * C_in * H_pad * W_pad * x.itemsize)

    x_pad_dev = int(cuda_malloc(lib, x_pad_bytes if x_pad_bytes > 0 else 1))
    w_dev = int(cuda_malloc(lib, int(w.nbytes) if int(w.nbytes) > 0 else 1))
    go_dev = int(
        cuda_malloc(lib, int(grad_out.nbytes) if int(grad_out.nbytes) > 0 else 1)
    )
    gx_pad_dev = int(cuda_malloc(lib, gx_pad_bytes if gx_pad_bytes > 0 else 1))
    gx_dev = int(cuda_malloc(lib, int(grad_x.nbytes) if int(grad_x.nbytes) > 0 else 1))
    gw_dev = int(cuda_malloc(lib, int(grad_w.nbytes) if int(grad_w.nbytes) > 0 else 1))

    try:
        # Copy unpadded x to device (no CPU np.pad)
        cudaMemcpyHtoD(lib, x_unpad_dev, x, int(x.nbytes))
        cudaMemcpyHtoD(lib, w_dev, w, int(w.nbytes))
        cudaMemcpyHtoD(lib, go_dev, grad_out, int(grad_out.nbytes))

        # Create padded x on device via pad2d CUDA kernel
        pad2d_cuda(
            lib,
            x_dev=x_unpad_dev,
            y_pad_dev=x_pad_dev,
            N=N,
            C=C_in,
            H=H,
            W=W,
            p_h=p_h,
            p_w=p_w,
            pad_value=0.0,
            dtype=dtype,
            device=(int(device_index) if device_index is not None else 0),
            sync=False,  # defer sync to the existing sync flag below
        )

        # ----------------------------
        # Debug assert: pad2d matches NumPy exactly
        # ----------------------------
        if (p_h > 0 or p_w > 0) and (N * C_in * H_pad * W_pad) > 0:
            x_pad_host = np.empty((N, C_in, H_pad, W_pad), dtype=dtype)
            cudaMemcpyDtoH(lib, x_pad_host, x_pad_dev, int(x_pad_host.nbytes))
            x_pad_np = np.pad(
                x,
                pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode="constant",
                constant_values=0.0,
            )
            assert x_pad_host.shape == x_pad_np.shape
            assert np.array_equal(
                x_pad_host, x_pad_np
            ), "pad2d_cuda mismatch vs np.pad (expected exact match)"

        # Zero-init grad_x_pad and grad_w on device (native kernel accumulates into them).
        grad_x_pad_zeros = np.zeros((N, C_in, H_pad, W_pad), dtype=dtype)
        assert (
            int(grad_x_pad_zeros.nbytes) == gx_pad_bytes
        ), f"gx_pad zero-init bytes mismatch: host={int(grad_x_pad_zeros.nbytes)} vs dev={gx_pad_bytes}"
        cudaMemcpyHtoD(lib, gx_pad_dev, grad_x_pad_zeros, int(grad_x_pad_zeros.nbytes))
        cudaMemcpyHtoD(lib, gw_dev, grad_w, int(grad_w.nbytes))  # zeros

        itemsize = _dtype_itemsize(dtype)
        _probe_dev_range(lib, x_pad_dev, N * C_in * H_pad * W_pad * itemsize)
        _probe_dev_range(lib, w_dev, C_out * C_in * K_h * K_w * itemsize)
        _probe_dev_range(lib, go_dev, N * C_out * H_out * W_out * itemsize)
        _probe_dev_range(lib, gx_pad_dev, N * C_in * H_pad * W_pad * itemsize)
        _probe_dev_range(lib, gw_dev, C_out * C_in * K_h * K_w * itemsize)

        _conv2d_backward_ctypes(
            lib,
            x_pad_dev=x_pad_dev,
            w_dev=w_dev,
            grad_out_dev=go_dev,
            grad_x_pad_dev=gx_pad_dev,
            grad_w_dev=gw_dev,
            N=N,
            C_in=C_in,
            H_pad=H_pad,
            W_pad=W_pad,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            dtype=dtype,
        )

        # Crop grad_x from grad_x_pad on device (no CPU slicing)
        crop2d_cuda(
            lib,
            x_pad_dev=gx_pad_dev,
            y_dev=gx_dev,
            N=N,
            C=C_in,
            H_pad=H_pad,
            W_pad=W_pad,
            p_h=p_h,
            p_w=p_w,
            H=H,
            W=W,
            dtype=dtype,
            device=(int(device_index) if device_index is not None else 0),
            sync=False,
        )

        if sync:
            cuda_synchronize(lib)

        # Copy results back to host
        cudaMemcpyDtoH(lib, grad_x, gx_dev, int(grad_x.nbytes))
        cudaMemcpyDtoH(lib, grad_w, gw_dev, int(grad_w.nbytes))

        # ----------------------------
        # Debug asserts: grads are finite
        # ----------------------------
        assert np.isfinite(
            grad_w
        ).all(), "conv2d_backward_cuda produced non-finite grad_w (NaN/Inf)"
        assert np.isfinite(
            grad_x
        ).all(), "conv2d_backward_cuda produced non-finite grad_x (NaN/Inf)"

        return grad_x, grad_w, grad_b

    finally:
        cuda_free(lib, x_unpad_dev)
        cuda_free(lib, x_pad_dev)
        cuda_free(lib, w_dev)
        cuda_free(lib, go_dev)
        cuda_free(lib, gx_pad_dev)
        cuda_free(lib, gx_dev)
        cuda_free(lib, gw_dev)


class _Conv2dCudaAliases:
    """
    Public alias names exposed by this ops-layer module.

    This class is for documentation/organization only and is not used at runtime.
    It exists to make it clear that this module intentionally exposes multiple
    entry points with identical behavior to mirror patterns used elsewhere in
    the codebase (e.g., matmul wrappers).

    Attributes
    ----------
    conv2d_cuda : callable
        Alias for `conv2d_forward_cuda`.
    conv2d_forward : callable
        Alias for `conv2d_forward_cuda`.
    conv2d_backward : callable
        Alias for `conv2d_backward_cuda`.
    """

    conv2d_cuda = conv2d_forward_cuda
    conv2d_forward = conv2d_forward_cuda
    conv2d_backward = conv2d_backward_cuda


# Aliases (similar to your matmul wrapper)
conv2d_cuda = conv2d_forward_cuda
conv2d_forward = conv2d_forward_cuda
conv2d_backward = conv2d_backward_cuda

__all__ = [
    "conv2d_forward_cuda",
    "conv2d_backward_cuda",
    "conv2d_cuda",
    "conv2d_forward",
    "conv2d_backward",
]
