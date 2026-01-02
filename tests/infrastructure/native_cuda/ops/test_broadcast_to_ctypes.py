from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
    cuda_synchronize,
    cudaMemcpyHtoD,
    cudaMemcpyDtoH,
)

from src.keydnn.infrastructure.native_cuda.python.ops.broadcast_ctypes import (
    broadcast_to_cuda,
)


class _CudaTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.lib = load_keydnn_cuda_native()
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            cls.lib = None
            cls._skip_reason = f"CUDA native library not available: {e!r}"

    def setUp(self) -> None:
        if getattr(self, "lib", None) is None:
            self.skipTest(getattr(self, "_skip_reason", "CUDA not available"))


class TestBroadcastToCtypes(_CudaTestCase):
    def _alloc_and_copy_in(self, x) -> int:
        """
        Allocate device memory and copy host -> device.

        Accepts np.ndarray OR numpy scalar; always converts to a C-contiguous ndarray
        because cudaMemcpyHtoD requires np.ndarray in this codebase.
        """
        lib = self.lib

        # Ensure ndarray (handles numpy scalars, Python scalars, lists, etc.)
        x_arr = np.asarray(x)
        if not x_arr.flags["C_CONTIGUOUS"]:
            x_arr = np.ascontiguousarray(x_arr)

        # Allocate at least 1 byte (cuda_malloc(0) may fail)
        nbytes = int(x_arr.nbytes)
        alloc_nbytes = max(nbytes, 1)

        x_dev = int(cuda_malloc(lib, alloc_nbytes))
        cudaMemcpyHtoD(lib, x_dev, x_arr, nbytes)  # nbytes can be 0 only if truly empty
        return x_dev

    def _alloc_out(
        self, dtype: np.dtype, out_shape: tuple[int, ...]
    ) -> tuple[int, int]:
        """Return (y_dev, y_nbytes_alloc). Allocate at least 1 byte to keep cuda_malloc happy."""
        lib = self.lib
        dtype = np.dtype(dtype)
        out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        out_nbytes = int(out_numel * dtype.itemsize)
        alloc_nbytes = max(out_nbytes, 1)
        y_dev = int(cuda_malloc(lib, alloc_nbytes))
        return y_dev, out_nbytes

    def _copy_out(
        self, dtype: np.dtype, out_shape: tuple[int, ...], y_dev: int
    ) -> np.ndarray:
        lib = self.lib
        dtype = np.dtype(dtype)
        y = np.empty(out_shape, dtype=dtype)
        nbytes = int(y.nbytes)
        if nbytes > 0:
            cudaMemcpyDtoH(lib, y, y_dev, nbytes)
        return y

    def _run_case(
        self,
        dtype: np.dtype,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        *,
        seed: int = 0,
    ) -> None:
        lib = self.lib
        dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        # deterministic + small
        x = (rng.random(in_shape, dtype=np.float64) - 0.5).astype(dtype, copy=False)
        y_ref = np.broadcast_to(x, out_shape).astype(dtype, copy=False)

        x_dev = 0
        y_dev = 0
        try:
            x_dev = self._alloc_and_copy_in(x)
            y_dev, y_nbytes = self._alloc_out(dtype, out_shape)

            broadcast_to_cuda(
                lib,
                x_dev=x_dev,
                y_dev=y_dev,
                in_shape=in_shape,
                out_shape=out_shape,
                dtype=dtype,
            )
            cuda_synchronize(lib)

            y = self._copy_out(dtype, out_shape, y_dev)

            if dtype == np.float32:
                np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_allclose(y, y_ref, rtol=1e-12, atol=1e-12)

        finally:
            if y_dev:
                cuda_free(lib, y_dev)
            if x_dev:
                cuda_free(lib, x_dev)

    # -------------------------
    # Correctness tests
    # -------------------------

    def test_broadcast_to_float32_basic_expand(self) -> None:
        # (3,1) -> (3,5)
        self._run_case(np.float32, (3, 1), (3, 5), seed=1)

    def test_broadcast_to_float32_rank_increase(self) -> None:
        # (4,) -> (2,4)
        self._run_case(np.float32, (4,), (2, 4), seed=2)

    def test_broadcast_to_float32_multi_axis(self) -> None:
        # (1,3,1) -> (2,3,4)
        self._run_case(np.float32, (1, 3, 1), (2, 3, 4), seed=3)

    def test_broadcast_to_float64_basic_expand(self) -> None:
        # (2,1,3) -> (2,7,3)
        self._run_case(np.float64, (2, 1, 3), (2, 7, 3), seed=4)

    def test_broadcast_to_float64_scalar_to_tensor(self) -> None:
        # () -> (2,3,4)
        self._run_case(np.float64, tuple(), (2, 3, 4), seed=5)

    # -------------------------
    # Error handling
    # -------------------------

    def test_broadcast_to_unsupported_dtype_raises(self) -> None:
        lib = self.lib
        with self.assertRaises(TypeError):
            broadcast_to_cuda(
                lib,
                x_dev=1,
                y_dev=2,
                in_shape=(1,),
                out_shape=(1,),
                dtype=np.int32,
            )

    def test_broadcast_to_incompatible_shapes_raises(self) -> None:
        # (2,3) cannot broadcast to (2,4)
        lib = self.lib
        dtype = np.float32
        x = (np.random.rand(2, 3) - 0.5).astype(dtype, copy=False)

        x_dev = 0
        y_dev = 0
        try:
            x_dev = self._alloc_and_copy_in(x)
            y_dev, _ = self._alloc_out(dtype, (2, 4))

            with self.assertRaises((ValueError, RuntimeError)):
                broadcast_to_cuda(
                    lib,
                    x_dev=x_dev,
                    y_dev=y_dev,
                    in_shape=(2, 3),
                    out_shape=(2, 4),
                    dtype=dtype,
                )
        finally:
            if y_dev:
                cuda_free(lib, y_dev)
            if x_dev:
                cuda_free(lib, x_dev)

    def test_broadcast_to_zero_numel_is_ok(self) -> None:
        # output has zero elements -> should no-op and not crash
        lib = self.lib
        dtype = np.float32

        x = np.array([1.0], dtype=dtype)
        x_dev = 0
        y_dev = 0
        try:
            x_dev = self._alloc_and_copy_in(x)

            # out_shape contains a zero dim => numel==0
            out_shape = (0, 5)
            y_dev, _ = self._alloc_out(dtype, out_shape)

            broadcast_to_cuda(
                lib,
                x_dev=x_dev,
                y_dev=y_dev,
                in_shape=(1,),
                out_shape=out_shape,
                dtype=dtype,
            )
            cuda_synchronize(lib)
        finally:
            if y_dev:
                cuda_free(lib, y_dev)
            if x_dev:
                cuda_free(lib, x_dev)


if __name__ == "__main__":
    unittest.main()
