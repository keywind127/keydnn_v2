"""
Unit tests for KeyDNN v2 CUDA reduction ops module (ops/reduce_cuda.py).

These tests validate that the *ops-level* device-pointer wrappers in:
    src.keydnn.infrastructure.ops.reduce_cuda

correctly dispatch to the underlying ctypes bindings and produce numerically
correct results compared to NumPy references.

Covered ops
-----------
- sum_all_cuda
- mean_all_cuda
- sum_backward_fill_cuda
- mean_backward_fill_cuda
- max_axis2d_forward_cuda
- max_axis2d_backward_cuda (with zero_grad_x=True)
- sum_axis2d_forward_cuda
- sum_axis2d_backward_cuda

Assumptions
-----------
- The CUDA native DLL has been built and is loadable.
- The exported CUDA utils exist (malloc/free/memcpy/memset/sync).
- The reduce_cuda ops module is available at:
    src.keydnn.infrastructure.ops.reduce_cuda

Notes
-----
- Tests are skipped cleanly if the DLL cannot be loaded or CUDA is unavailable.
- float32 tolerances are relaxed for large reductions due to atomicAdd order.
- For max backward, we validate argmax-scatter semantics (single index per slice).
"""

from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_from_host,
    cuda_from_host_any,  # supports int64 buffers (argmax idx)
    cuda_malloc,
    cuda_free,
    cuda_memcpy_d2h,
    cuda_synchronize,
)

from src.keydnn.infrastructure.ops.reduce_cuda import (
    sum_all_cuda,
    mean_all_cuda,
    sum_backward_fill_cuda,
    mean_backward_fill_cuda,
    max_axis2d_forward_cuda,
    max_axis2d_backward_cuda,
    sum_axis2d_forward_cuda,
    sum_axis2d_backward_cuda,
)


# -----------------------------------------------------------------------------
# NumPy references
# -----------------------------------------------------------------------------


def sum_all_cpu(x: np.ndarray) -> np.ndarray:
    return np.array(x.sum(), dtype=x.dtype)


def mean_all_cpu(x: np.ndarray) -> np.ndarray:
    return np.array(x.mean(), dtype=x.dtype)


def sum_backward_fill_cpu(grad_out_scalar: np.ndarray, *, numel: int) -> np.ndarray:
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
    grad_x = np.zeros((rows, cols), dtype=dtype)
    if axis == 1:
        for r in range(rows):
            c = int(idx[r])
            grad_x[r, c] += grad_out[r]
        return grad_x
    if axis == 0:
        for c in range(cols):
            r = int(idx[c])
            grad_x[r, c] += grad_out[c]
        return grad_x
    raise ValueError("axis must be 0 or 1")


def sum_axis2d_forward_cpu(x: np.ndarray, *, axis: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    return x.sum(axis=axis).astype(x.dtype, copy=False)


def sum_axis2d_backward_cpu(
    grad_out: np.ndarray, *, rows: int, cols: int, axis: int, dtype: np.dtype
) -> np.ndarray:
    if axis == 1:
        # grad_out shape: (rows,)
        return np.broadcast_to(grad_out.reshape(rows, 1), (rows, cols)).astype(
            dtype, copy=False
        )
    if axis == 0:
        # grad_out shape: (cols,)
        return np.broadcast_to(grad_out.reshape(1, cols), (rows, cols)).astype(
            dtype, copy=False
        )
    raise ValueError("axis must be 0 or 1")


def _tols_sum(dtype: np.dtype, numel: int) -> tuple[float, float]:
    if dtype == np.float32:
        # atomicAdd reduction order is nondeterministic; error grows with numel
        if numel >= 100_000:
            return (2e-5, 1e-4)
        return (1e-5, 1e-6)
    return (1e-12, 1e-12)


def _tols_default(dtype: np.dtype) -> tuple[float, float]:
    if dtype == np.float32:
        return (1e-5, 1e-6)
    return (1e-12, 1e-12)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestReduceCudaOps(unittest.TestCase):
    """
    End-to-end correctness tests for ops/reduce_cuda.py.

    Strategy:
    - Allocate device buffers with cuda_malloc / cuda_from_host / cuda_from_host_any
    - Call ops-layer functions (reduce_cuda.py)
    - Copy results back to host and compare to NumPy references
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
        return np.random.default_rng(13579)

    # -------------------------
    # sum_all forward
    # -------------------------

    def _run_sum_all_forward_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        x = rng.standard_normal((numel,)).astype(dtype, copy=False)
        y_ref = sum_all_cpu(x)

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            sum_all_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                numel=numel,
                dtype=dtype,
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        rtol, atol = _tols_sum(dtype, numel)
        np.testing.assert_allclose(y_host[0], y_ref.item(), rtol=rtol, atol=atol)

    def test_sum_all_forward_f32_small(self) -> None:
        self._run_sum_all_forward_case(np.float32, numel=257)

    def test_sum_all_forward_f64_small(self) -> None:
        self._run_sum_all_forward_case(np.float64, numel=257)

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
        y_ref = mean_all_cpu(x)

        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            mean_all_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                numel=numel,
                dtype=dtype,
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(y_host[0], y_ref.item(), rtol=rtol, atol=atol)

    def test_mean_all_forward_f32_small(self) -> None:
        self._run_mean_all_forward_case(np.float32, numel=1024)

    def test_mean_all_forward_f64_small(self) -> None:
        self._run_mean_all_forward_case(np.float64, numel=1024)

    def test_mean_all_forward_f32_large(self) -> None:
        self._run_mean_all_forward_case(np.float32, numel=200_000)

    def test_mean_all_forward_f64_large(self) -> None:
        self._run_mean_all_forward_case(np.float64, numel=200_000)

    # -------------------------
    # sum backward fill
    # -------------------------

    def _run_sum_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        grad_out = np.array(rng.standard_normal(), dtype=dtype)

        grad_x_ref = sum_backward_fill_cpu(grad_out, numel=numel)

        # device scalar grad_out: pass as a 1-element buffer
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
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_sum_backward_fill_f32(self) -> None:
        self._run_sum_backward_fill_case(np.float32, numel=12_345)

    def test_sum_backward_fill_f64(self) -> None:
        self._run_sum_backward_fill_case(np.float64, numel=12_345)

    # -------------------------
    # mean backward fill
    # -------------------------

    def _run_mean_backward_fill_case(self, dtype: np.dtype, *, numel: int) -> None:
        rng = self._rng()
        grad_out = np.array(rng.standard_normal(), dtype=dtype)

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
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, go_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_mean_backward_fill_f32(self) -> None:
        self._run_mean_backward_fill_case(np.float32, numel=65_537)

    def test_mean_backward_fill_f64(self) -> None:
        self._run_mean_backward_fill_case(np.float64, numel=65_537)

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
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_memcpy_d2h(self.lib, idx_host, idx_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)
        np.testing.assert_array_equal(idx_host, idx_ref)

    def _run_max_axis2d_backward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = self._rng()
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)

        _, idx_ref = max_axis2d_forward_cpu(x, axis=axis)

        go_shape = (rows,) if axis == 1 else (cols,)
        grad_out = rng.standard_normal(go_shape).astype(dtype, copy=False)

        grad_x_ref = max_axis2d_backward_cpu(
            grad_out, idx_ref, rows=rows, cols=cols, axis=axis, dtype=dtype
        )

        grad_out_dev = cuda_from_host(self.lib, grad_out)
        idx_dev = cuda_from_host_any(self.lib, idx_ref.astype(np.int64, copy=False))

        grad_x_host = np.empty((rows, cols), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            max_axis2d_backward_cuda(
                self.lib,
                grad_out_dev=grad_out_dev,
                idx_dev=idx_dev,
                grad_x_dev=grad_x_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
                zero_grad_x=True,  # exercise ops-level zeroing path
                sync=True,
            )

            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, grad_out_dev)
            cuda_free(self.lib, idx_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = _tols_default(dtype)
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
    # sum axis2d forward/backward
    # -------------------------

    def _run_sum_axis2d_forward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = self._rng()
        x = rng.standard_normal((rows, cols)).astype(dtype, copy=False)
        y_ref = sum_axis2d_forward_cpu(x, axis=axis)

        x_dev = cuda_from_host(self.lib, x)

        y_len = rows if axis == 1 else cols
        y_host = np.empty((y_len,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            sum_axis2d_forward_cuda(
                self.lib,
                x_dev=x_dev,
                y_dev=y_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, y_host, y_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)

        # Forward here is deterministic (single thread per row/col), so default tol is fine.
        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def _run_sum_axis2d_backward_case(
        self, dtype: np.dtype, *, rows: int, cols: int, axis: int
    ) -> None:
        rng = self._rng()

        go_shape = (rows,) if axis == 1 else (cols,)
        grad_out = rng.standard_normal(go_shape).astype(dtype, copy=False)

        grad_x_ref = sum_axis2d_backward_cpu(
            grad_out, rows=rows, cols=cols, axis=axis, dtype=dtype
        )

        grad_out_dev = cuda_from_host(self.lib, grad_out)

        grad_x_host = np.empty((rows, cols), dtype=dtype)
        grad_x_dev = cuda_malloc(self.lib, grad_x_host.nbytes)

        try:
            sum_axis2d_backward_cuda(
                self.lib,
                grad_out_dev=grad_out_dev,
                grad_x_dev=grad_x_dev,
                rows=rows,
                cols=cols,
                axis=axis,
                dtype=dtype,
                sync=True,
            )
            cuda_memcpy_d2h(self.lib, grad_x_host, grad_x_dev)
            cuda_synchronize(self.lib)
        finally:
            cuda_free(self.lib, grad_out_dev)
            cuda_free(self.lib, grad_x_dev)

        rtol, atol = _tols_default(dtype)
        np.testing.assert_allclose(grad_x_host, grad_x_ref, rtol=rtol, atol=atol)

    def test_sum_axis2d_forward_f32_axis1(self) -> None:
        self._run_sum_axis2d_forward_case(np.float32, rows=23, cols=11, axis=1)

    def test_sum_axis2d_forward_f64_axis1(self) -> None:
        self._run_sum_axis2d_forward_case(np.float64, rows=23, cols=11, axis=1)

    def test_sum_axis2d_forward_f32_axis0(self) -> None:
        self._run_sum_axis2d_forward_case(np.float32, rows=13, cols=37, axis=0)

    def test_sum_axis2d_forward_f64_axis0(self) -> None:
        self._run_sum_axis2d_forward_case(np.float64, rows=13, cols=37, axis=0)

    def test_sum_axis2d_backward_f32_axis1(self) -> None:
        self._run_sum_axis2d_backward_case(np.float32, rows=17, cols=29, axis=1)

    def test_sum_axis2d_backward_f64_axis1(self) -> None:
        self._run_sum_axis2d_backward_case(np.float64, rows=17, cols=29, axis=1)

    def test_sum_axis2d_backward_f32_axis0(self) -> None:
        self._run_sum_axis2d_backward_case(np.float32, rows=19, cols=31, axis=0)

    def test_sum_axis2d_backward_f64_axis0(self) -> None:
        self._run_sum_axis2d_backward_case(np.float64, rows=19, cols=31, axis=0)

    # -------------------------
    # Edge / validation tests
    # -------------------------

    def test_max_axis2d_ties_choose_first_argmax_like_numpy(self) -> None:
        dtype = np.float32
        x = np.array(
            [
                [1.0, 2.0, 2.0, 0.0],  # tie at 1 and 2 -> argmax=1
                [3.0, 3.0, -1.0, 3.0],  # tie at 0,1,3 -> argmax=0
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
                sync=True,
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

    def test_ops_rejects_unsupported_dtype(self) -> None:
        # Allocate dummy float buffers
        x = self._rng().standard_normal((16,)).astype(np.float32, copy=False)
        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((1,), dtype=np.float32)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            with self.assertRaises(TypeError):
                sum_all_cuda(
                    self.lib, x_dev=x_dev, y_dev=y_dev, numel=16, dtype=np.int32
                )
            with self.assertRaises(TypeError):
                mean_all_cuda(
                    self.lib, x_dev=x_dev, y_dev=y_dev, numel=16, dtype=np.int32
                )

            # exercise new ops dtype validation too (use same dummy buffers)
            with self.assertRaises(TypeError):
                sum_axis2d_forward_cuda(
                    self.lib,
                    x_dev=x_dev,
                    y_dev=y_dev,  # dummy
                    rows=4,
                    cols=4,
                    axis=1,
                    dtype=np.int32,
                    sync=True,
                )
            with self.assertRaises(TypeError):
                sum_axis2d_backward_cuda(
                    self.lib,
                    grad_out_dev=x_dev,  # dummy
                    grad_x_dev=y_dev,  # dummy
                    rows=4,
                    cols=4,
                    axis=1,
                    dtype=np.int32,
                    sync=True,
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
                    sync=True,
                )
            with self.assertRaises(ValueError):
                max_axis2d_backward_cuda(
                    self.lib,
                    grad_out_dev=x_dev,  # dummy
                    idx_dev=idx_dev,
                    grad_x_dev=y_dev,  # dummy
                    rows=3,
                    cols=4,
                    axis=-1,  # invalid
                    dtype=dtype,
                    zero_grad_x=False,
                    sync=True,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)
            cuda_free(self.lib, idx_dev)

    def test_sum_axis2d_rejects_bad_axis(self) -> None:
        dtype = np.float32
        x = self._rng().standard_normal((3, 4)).astype(dtype, copy=False)
        x_dev = cuda_from_host(self.lib, x)
        y_host = np.empty((3,), dtype=dtype)
        y_dev = cuda_malloc(self.lib, y_host.nbytes)

        try:
            with self.assertRaises(ValueError):
                sum_axis2d_forward_cuda(
                    self.lib,
                    x_dev=x_dev,
                    y_dev=y_dev,
                    rows=3,
                    cols=4,
                    axis=2,  # invalid
                    dtype=dtype,
                    sync=True,
                )
            with self.assertRaises(ValueError):
                sum_axis2d_backward_cuda(
                    self.lib,
                    grad_out_dev=x_dev,  # dummy
                    grad_x_dev=y_dev,  # dummy
                    rows=3,
                    cols=4,
                    axis=-1,  # invalid
                    dtype=dtype,
                    sync=True,
                )
        finally:
            cuda_free(self.lib, x_dev)
            cuda_free(self.lib, y_dev)


if __name__ == "__main__":
    unittest.main()
