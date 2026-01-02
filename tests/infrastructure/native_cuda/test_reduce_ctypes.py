"""
Unit tests for KeyDNN v2 CUDA reduction ctypes bindings (reduce_ctypes).

These tests validate correctness of CUDA reductions implemented in:
    keydnn/infrastructure/native_cuda/python/reduce_ctypes.py

Covered ops
-----------
- sum_all_cuda + sum_backward_fill_cuda
- mean_all_cuda + mean_backward_fill_cuda
- max_axis2d_forward_cuda + max_axis2d_backward_cuda (2D only, axis in {0,1})

Assumptions
-----------
- The CUDA native DLL has been built and is loadable.
- The reduce ctypes wrapper is available at:
    src.keydnn.infrastructure.native_cuda.python.reduce_ctypes
- We rely on the same CUDA utils exported by the DLL:
    cuda_set_device, cuda_malloc, cuda_free, cuda_from_host, cuda_memcpy_d2h, cuda_memset, cuda_synchronize

Notes
-----
- Tests are skipped cleanly if the DLL cannot be loaded or CUDA is unavailable.
- We use practical tolerances for float32 and strict tolerances for float64.
- sum_all/mean_all use atomic reductions on GPU (non-deterministic accumulation order),
  so float32 tolerances are relaxed for large numel.
- For max backward, we validate gradient routing to a *single* argmax position per reduced slice,
  which matches the typical "argmax index" backward semantics (not tie-splitting by mask).
"""

from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_from_host,
    cuda_memcpy_d2h,
    cuda_memset,
    cuda_synchronize,
)

from src.keydnn.infrastructure.native_cuda.python.reduce_ctypes import (
    sum_all_cuda,
    mean_all_cuda,
    sum_backward_fill_cuda,
    mean_backward_fill_cuda,
    max_axis2d_forward_cuda,
    max_axis2d_backward_cuda,
    sum_to_shape_cuda,
)


# -----------------------------------------------------------------------------
# NumPy references
# -----------------------------------------------------------------------------


def sum_all_cpu(x: np.ndarray) -> np.ndarray:
    # Keep reference dtype consistent with input dtype
    return np.array(x.sum(dtype=x.dtype), dtype=x.dtype)


def mean_all_cpu(x: np.ndarray) -> np.ndarray:
    return np.array(x.mean(dtype=x.dtype), dtype=x.dtype)


def sum_backward_fill_cpu(grad_out_scalar: np.ndarray, *, numel: int) -> np.ndarray:
    # grad_out_scalar is scalar array (shape ())
    return (np.ones((numel,), dtype=grad_out_scalar.dtype) * grad_out_scalar).astype(
        grad_out_scalar.dtype, copy=False
    )


def mean_backward_fill_cpu(grad_out_scalar: np.ndarray, *, numel: int) -> np.ndarray:
    return (
        np.ones((numel,), dtype=grad_out_scalar.dtype)
        * (grad_out_scalar / np.array(float(numel), dtype=grad_out_scalar.dtype))
    ).astype(grad_out_scalar.dtype, copy=False)


def max_axis2d_forward_cpu(
    x: np.ndarray, *, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    2D argmax-based max forward reference.

    Returns
    -------
    y : np.ndarray
        Reduced values.
    idx : np.ndarray (int64)
        Argmax indices along reduced axis.
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if axis == 1:
        idx = np.argmax(x, axis=1).astype(np.int64, copy=False)
        y = x[np.arange(x.shape[0]), idx].astype(x.dtype, copy=False)
        return y, idx
    if axis == 0:
        idx = np.argmax(x, axis=0).astype(np.int64, copy=False)
        y = x[idx, np.arange(x.shape[1])].astype(x.dtype, copy=False)
        return y, idx
    raise ValueError("axis must be 0 or 1")


def max_axis2d_backward_cpu(
    grad_out: np.ndarray,
    idx: np.ndarray,
    *,
    rows: int,
    cols: int,
    axis: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Scatter grad_out to argmax locations in grad_x. Ties handled by argmax choice.
    """
    grad_x = np.zeros((rows, cols), dtype=dtype)
    if axis == 1:
        # grad_out shape: (rows,)
        for r in range(rows):
            c = int(idx[r])
            grad_x[r, c] += grad_out[r]
        return grad_x
    if axis == 0:
        # grad_out shape: (cols,)
        for c in range(cols):
            r = int(idx[c])
            grad_x[r, c] += grad_out[c]
        return grad_x
    raise ValueError("axis must be 0 or 1")


def sum_to_shape_cpu(
    x: np.ndarray, *, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """
    NumPy reference for sum_to_shape ("unbroadcast sum").

    Assumes:
    - x is contiguous flattened buffer with numel == prod(in_shape)
    - in_shape and out_shape have the same rank
    - for each dim d: out_shape[d] == in_shape[d] or out_shape[d] == 1
    """
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    if len(in_shape) != len(out_shape):
        raise ValueError("rank mismatch")
    x_view = x.reshape(in_shape)

    reduce_axes = tuple(
        i
        for i, (id_, od_) in enumerate(zip(in_shape, out_shape))
        if (od_ == 1 and id_ != 1)
    )
    if reduce_axes:
        y = x_view.sum(axis=reduce_axes, keepdims=True, dtype=x.dtype)
    else:
        y = x_view.astype(x.dtype, copy=False)

    # keepdims=True already matches out_shape (same rank), but be strict:
    y = y.reshape(out_shape)
    return y


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestReduceCudaCtypes(unittest.TestCase):
    """
    Correctness tests for CUDA reduction kernels via ctypes.

    Strategy:
    - Allocate device buffers with cuda_malloc / cuda_from_host
    - Run reduce kernels (sum/mean/max forward/backward)
    - Copy results back via cuda_memcpy_d2h
    - Compare to NumPy references
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = load_keydnn_cuda_native()
        except (OSError, FileNotFoundError) as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        try:
            cuda_set_device(cls.lib, 0)
        except RuntimeError as e:
            raise unittest.SkipTest(f"CUDA device unavailable: {e}")

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(24680)

    def _tols(self, dtype: np.dtype) -> tuple[float, float]:
        if dtype == np.float32:
            return (1e-5, 1e-6)
        return (1e-12, 1e-12)

    def _tols_sum(self, dtype: np.dtype, numel: int) -> tuple[float, float]:
        """
        Tolerances for sum/mean reductions.

        CUDA sum/mean use atomic reductions: float32 results can differ slightly
        from CPU due to non-deterministic accumulation order, especially for
        large numel.
        """
        if dtype == np.float32:
            if numel >= 100_000:
                # relaxed for large reductions
                return (2e-5, 1e-4)
            return (1e-5, 1e-6)
        return (1e-12, 1e-12)

    def _cuda_from_host_i64(self, x: np.ndarray) -> int:
        """
        Allocate device buffer and copy an int64 NumPy array to GPU.

        We intentionally do NOT reuse cuda_from_host from global_avgpool2d_ctypes
        because that helper is float-only.
        """
        if x.dtype != np.int64:
            raise TypeError(f"_cuda_from_host_i64 expects int64, got {x.dtype}")
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        dev = cuda_malloc(self.lib, x.nbytes)
        try:
            # cuda_memcpy_h2d signature: (dst_dev, src_host_void_p, nbytes)
            self.lib.keydnn_cuda_memcpy_h2d(
                np.uint64(int(dev)),
                int(x.ctypes.data),  # treated as void*
                int(x.nbytes),
            )
            # If the DLL binding isn't set on the lib object in your environment,
            # fallback to the bound wrapper by importing and calling it.
        except Exception:
            cuda_free(self.lib, dev)
            raise
        return dev

    # -------------------------
    # sum_all forward
    # -------------------------

    def _run_sum_all_forward_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        x = rng.standard_normal((numel,)).astype(dtype, copy=False)

        y_ref = sum_all_cpu(x)  # scalar array ()

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=dtype)  # device scalar copied into len-1
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            sum_all_cuda(self.lib, x_dev=x_dev, y_dev=y_dev, numel=numel, dtype=dtype)
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        rtol, atol = self._tols_sum(dtype, numel)
        np.testing.assert_allclose(y_host[0], y_ref.item(), rtol=rtol, atol=atol)

    def test_sum_all_forward_f32_small(self) -> None:
        self._run_sum_all_forward_case(np.float32, numel=128)

    def test_sum_all_forward_f64_small(self) -> None:
        self._run_sum_all_forward_case(np.float64, numel=128)

    def test_sum_all_forward_f32_large(self) -> None:
        self._run_sum_all_forward_case(np.float32, numel=100_003)

    def test_sum_all_forward_f64_large(self) -> None:
        self._run_sum_all_forward_case(np.float64, numel=100_003)

    # -------------------------
    # mean_all forward
    # -------------------------

    def _run_mean_all_forward_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        x = rng.standard_normal((numel,)).astype(dtype, copy=False)

        y_ref = mean_all_cpu(x)  # scalar

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            mean_all_cuda(self.lib, x_dev=x_dev, y_dev=y_dev, numel=numel, dtype=dtype)
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        rtol, atol = self._tols_sum(dtype, numel)
        np.testing.assert_allclose(y_host[0], y_ref.item(), rtol=rtol, atol=atol)

    def test_mean_all_forward_f32_small(self) -> None:
        self._run_mean_all_forward_case(np.float32, numel=257)

    def test_mean_all_forward_f64_small(self) -> None:
        self._run_mean_all_forward_case(np.float64, numel=257)

    def test_mean_all_forward_f32_large(self) -> None:
        self._run_mean_all_forward_case(np.float32, numel=200_000)

    def test_mean_all_forward_f64_large(self) -> None:
        self._run_mean_all_forward_case(np.float64, numel=200_000)

    # -------------------------
    # sum backward fill
    # -------------------------

    def _run_sum_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        grad_out = np.array(rng.standard_normal(), dtype=dtype)  # scalar ()

        grad_x_ref = sum_backward_fill_cpu(grad_out, numel=numel)

        # device scalar grad_out
        go_host = np.array([grad_out.item()], dtype=dtype)
        go_dev = cuda_from_host(self.lib, go_host)

        grad_x_host = np.empty((numel,), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            sum_backward_fill_cuda(
                self.lib,
                grad_out_dev=go_dev,
                grad_x_dev=grad_x_dev,
                numel=numel,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_sum_backward_fill_f32(self) -> None:
        self._run_sum_backward_fill_case(np.float32, numel=10_000)

    def test_sum_backward_fill_f64(self) -> None:
        self._run_sum_backward_fill_case(np.float64, numel=10_000)

    # -------------------------
    # mean backward fill
    # -------------------------

    def _run_mean_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        grad_out = np.array(rng.standard_normal(), dtype=dtype)  # scalar ()

        grad_x_ref = mean_backward_fill_cpu(grad_out, numel=numel)

        go_host = np.array([grad_out.item()], dtype=dtype)
        go_dev = cuda_from_host(self.lib, go_host)

        grad_x_host = np.empty((numel,), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            mean_backward_fill_cuda(
                self.lib,
                grad_out_dev=go_dev,
                grad_x_dev=grad_x_dev,
                numel=numel,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_mean_backward_fill_f32(self) -> None:
        self._run_mean_backward_fill_case(np.float32, numel=12_345)

    def test_mean_backward_fill_f64(self) -> None:
        self._run_mean_backward_fill_case(np.float64, numel=12_345)

    # -------------------------
    # max axis2d forward/backward
    # -------------------------

    def _run_max_axis2d_forward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = self._rng()
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)

        y_ref, idx_ref = max_axis2d_forward_cpu(x, axis=axis)

        x_dev = cuda_from_host(self.lib, x)

        y_host = np.empty((rows if axis == 1 else cols,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        idx_host = np.empty((rows if axis == 1 else cols,), dtype=np.int64)
        idx_dev = cuda_malloc(self.lib, idx_host.nbytes)

        try:
            max_axis2d_forward_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                idx_dev=idx_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
            )

            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_memcpy_d2h(self.lib, idx_host, idx_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)
        np.testing.assert_array_equal(idx_host, idx_ref)

    def _run_max_axis2d_backward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = self._rng()
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)

        # Forward ref to get idx (argmax)
        _, idx_ref = max_axis2d_forward_cpu(x, axis=axis)

        # grad_out shaped as reduced output
        go_shape = (rows,) if axis == 1 else (cols,)
        grad_out = rng.standard_normal(go_shape).astype(dtype, copy=False)

        grad_x_ref = max_axis2d_backward_cpu(
            grad_out, idx_ref, rows=rows, cols=cols, axis=axis, dtype=dtype
        )

        # Allocate device buffers
        grad_out_dev = cuda_from_host(self.lib, grad_out)

        # idx is int64 => allocate + copy manually (cuda_from_host is float-only)
        idx_dev = self._cuda_from_host_i64(idx_ref.astype(np.int64, copy=False))

        grad_x_host = np.empty((rows, cols), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            # important: grad_x must be zeroed before scatter-add
            cuda_memset(self.lib, grad_x_dev, 0, grad_x_host.nbytes)

            max_axis2d_backward_cuda(
                self.lib,
                grad_out_dev=grad_out_dev,
                idx_dev=idx_dev,
                grad_x_dev=grad_x_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
            )

            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, grad_out_dev)
            cuda_free(self.lib, idx_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = self._tols(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_max_axis2d_forward_f32_axis1(self) -> None:
        self._run_max_axis2d_forward_case(np.float32, rows=17, cols=31, axis=1)

    def test_max_axis2d_forward_f64_axis1(self) -> None:
        self._run_max_axis2d_forward_case(np.float64, rows=17, cols=31, axis=1)

    def test_max_axis2d_forward_f32_axis0(self) -> None:
        self._run_max_axis2d_forward_case(np.float32, rows=19, cols=29, axis=0)

    def test_max_axis2d_forward_f64_axis0(self) -> None:
        self._run_max_axis2d_forward_case(np.float64, rows=19, cols=29, axis=0)

    def test_max_axis2d_backward_f32_axis1(self) -> None:
        self._run_max_axis2d_backward_case(np.float32, rows=23, cols=11, axis=1)

    def test_max_axis2d_backward_f64_axis1(self) -> None:
        self._run_max_axis2d_backward_case(np.float64, rows=23, cols=11, axis=1)

    def test_max_axis2d_backward_f32_axis0(self) -> None:
        self._run_max_axis2d_backward_case(np.float32, rows=13, cols=37, axis=0)

    def test_max_axis2d_backward_f64_axis0(self) -> None:
        self._run_max_axis2d_backward_case(np.float64, rows=13, cols=37, axis=0)

    # -------------------------
    # Edge / determinism tests
    # -------------------------

    def test_max_axis2d_ties_choose_first_argmax_like_numpy(self) -> None:
        """
        Construct a tie case and ensure idx matches NumPy argmax behavior.
        """
        dtype = np.float32
        x = np.array(
            [
                [1.0, 2.0, 2.0, 0.0],  # tie at cols 1 and 2 -> argmax=1
                [3.0, 3.0, -1.0, 3.0],  # tie at cols 0,1,3 -> argmax=0
            ],
            dtype=dtype,
        )
        rows, cols = x.shape
        axis = 1

        y_ref, idx_ref = max_axis2d_forward_cpu(x, axis=axis)

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((rows,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)
        idx_host = np.empty((rows,), dtype=np.int64)
        idx_dev = cuda_malloc(self.lib, idx_host.nbytes)

        try:
            max_axis2d_forward_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                idx_dev=idx_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_memcpy_d2h(self.lib, idx_host, idx_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

        np.testing.assert_allclose(y_host, y_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(idx_host, idx_ref)

    def test_sum_mean_rejects_unsupported_dtype(self) -> None:
        """
        reduce_ctypes should raise TypeError for unsupported dtype.
        """
        x = np.arange(10, dtype=np.int32)
        x_dev = cuda_from_host(
            self.lib, x.astype(np.float32)
        )  # allocate something valid
        y_host = np.empty((1,), dtype=np.float32)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)
        try:
            with self.assertRaises(TypeError):
                sum_all_cuda(
                    self.lib, x_dev=x_dev, y_dev=y_dev, numel=10, dtype=np.int32
                )
            with self.assertRaises(TypeError):
                mean_all_cuda(
                    self.lib, x_dev=x_dev, y_dev=y_dev, numel=10, dtype=np.int32
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

    def test_max_axis2d_rejects_bad_axis(self) -> None:
        dtype = np.float32
        x = self._rng().standard_normal((3, 4)).astype(dtype, copy=False)
        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((3,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)
        idx_host = np.empty((3,), dtype=np.int64)
        idx_dev = cuda_malloc(self.lib, idx_host.nbytes)

        try:
            with self.assertRaises(ValueError):
                max_axis2d_forward_cuda(
                    self.lib,
                    x_dev=x_dev,
                    y_dev=y_dev,
                    idx_dev=idx_dev,
                    rows=3,
                    cols=4,
                    axis=2,  # invalid
                    dtype=dtype,
                )
            with self.assertRaises(ValueError):
                max_axis2d_backward_cuda(
                    self.lib,
                    grad_out_dev=x_dev,  # dummy
                    idx_dev=idx_dev,
                    grad_x_dev=y_dev,  # dummy
                    rows=3,
                    cols=4,
                    axis=-1,  # invalid for this wrapper
                    dtype=dtype,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

        # -------------------------

    # sum_to_shape (general unbroadcast reduction)
    # -------------------------

    def _tols_sum_to_shape(self, dtype: np.dtype, numel_in: int) -> tuple[float, float]:
        """
        Tolerances for sum_to_shape.

        Uses atomicAdd into multiple outputs; float32 can differ slightly from NumPy.
        """
        if dtype == np.float32:
            if numel_in >= 100_000:
                return (2e-5, 1e-4)
            return (1e-5, 1e-6)
        return (1e-12, 1e-12)

    def _run_sum_to_shape_case(
        self,
        dtype: np.dtype,
        *,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        dtype = np.dtype(dtype)

        in_shape = tuple(int(d) for d in in_shape)
        out_shape = tuple(int(d) for d in out_shape)

        # Rank must match (caller pads in real use)
        self.assertEqual(len(in_shape), len(out_shape))

        numel_in = int(np.prod(in_shape, dtype=np.int64))
        # Allocate host input as flat buffer (matches kernel assumption)
        x = rng.standard_normal((numel_in,)).astype(dtype, copy=False)

        y_ref = sum_to_shape_cpu(x, in_shape=in_shape, out_shape=out_shape)

        # Device allocations
        x_dev = cuda_from_host(self.lib, x)

        y_host = np.empty(out_shape, dtype=dtype)
        y_dev = cuda_malloc(self.lib, int(y_host.nbytes))

        try:
            sum_to_shape_cuda(
                self.lib,
                x_dev=int(x_dev),
                y_dev=int(y_dev),
                in_shape=in_shape,
                out_shape=out_shape,
                dtype=dtype,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        rtol, atol = self._tols_sum_to_shape(dtype, numel_in)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def test_sum_to_shape_f32_reduce_two_axes(self) -> None:
        # (2,3,4) -> (1,3,1): reduce axes 0 and 2
        self._run_sum_to_shape_case(
            np.float32,
            in_shape=(2, 3, 4),
            out_shape=(1, 3, 1),
            seed=10,
        )

    def test_sum_to_shape_f64_reduce_two_axes(self) -> None:
        self._run_sum_to_shape_case(
            np.float64,
            in_shape=(2, 3, 4),
            out_shape=(1, 3, 1),
            seed=11,
        )

    def test_sum_to_shape_f32_no_reduction_identity(self) -> None:
        # out_shape == in_shape => should match x reshaped
        self._run_sum_to_shape_case(
            np.float32,
            in_shape=(4, 5),
            out_shape=(4, 5),
            seed=12,
        )

    def test_sum_to_shape_f32_reduce_last_dim(self) -> None:
        # (7,9,8) -> (7,9,1): reduce axis 2
        self._run_sum_to_shape_case(
            np.float32,
            in_shape=(7, 9, 8),
            out_shape=(7, 9, 1),
            seed=13,
        )

    def test_sum_to_shape_f32_reduce_middle_dim(self) -> None:
        # (5,6,7) -> (5,1,7): reduce axis 1
        self._run_sum_to_shape_case(
            np.float32,
            in_shape=(5, 6, 7),
            out_shape=(5, 1, 7),
            seed=14,
        )

    def test_sum_to_shape_f32_large(self) -> None:
        # Large-ish to exercise atomic accumulation differences
        self._run_sum_to_shape_case(
            np.float32,
            in_shape=(64, 33, 7),  # numel ~ 14784 (not huge but enough)
            out_shape=(1, 33, 1),
            seed=15,
        )

    def test_sum_to_shape_rejects_unsupported_dtype(self) -> None:
        # allocate a valid float buffer for x_dev
        x = np.arange(12, dtype=np.float32)
        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=np.float32)
        y_dev = cuda_malloc(self.lib, int(y_host.nbytes))
        try:
            with self.assertRaises(TypeError):
                sum_to_shape_cuda(
                    self.lib,
                    x_dev=int(x_dev),
                    y_dev=int(y_dev),
                    in_shape=(12,),
                    out_shape=(1,),
                    dtype=np.int32,  # unsupported
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

    def test_sum_to_shape_rank_mismatch_raises(self) -> None:
        x = np.arange(6, dtype=np.float32)
        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=np.float32)
        y_dev = cuda_malloc(self.lib, int(y_host.nbytes))
        try:
            with self.assertRaises(ValueError):
                sum_to_shape_cuda(
                    self.lib,
                    x_dev=int(x_dev),
                    y_dev=int(y_dev),
                    in_shape=(2, 3),
                    out_shape=(1, 1, 1),  # rank mismatch
                    dtype=np.float32,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

    def test_sum_to_shape_incompatible_shapes_raises_runtime(self) -> None:
        """
        Out shape must be 1 or equal to input dim per axis.
        Example: (2,3,4) -> (2,2,4) is invalid (2 != 3 and not 1).
        """
        dtype = np.float32
        x = (
            np.random.default_rng(0)
            .standard_normal((2 * 3 * 4,))
            .astype(dtype, copy=False)
        )

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((2, 2, 4), dtype=dtype)
        y_dev = cuda_malloc(self.lib, int(y_host.nbytes))
        try:
            with self.assertRaises(RuntimeError):
                sum_to_shape_cuda(
                    self.lib,
                    x_dev=int(x_dev),
                    y_dev=int(y_dev),
                    in_shape=(2, 3, 4),
                    out_shape=(2, 2, 4),  # invalid
                    dtype=dtype,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)


if __name__ == "__main__":
    unittest.main()
