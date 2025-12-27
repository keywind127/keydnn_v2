# tests/infrastructure/native_cuda/ops/test_tensor_arithmetic_ctypes.py
from __future__ import annotations

import unittest
from typing import Any, Callable, Tuple

import numpy as np


def _import_first(*candidates: Tuple[str, str]) -> Any:
    last = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last = e
    raise ImportError(f"Failed to import any of: {candidates}. Last error: {last}")


def _try_load_cuda_lib():
    try:
        load_keydnn_cuda_native = _import_first(
            (
                "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
                "load_keydnn_cuda_native",
            ),
            (
                "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
                "load_keydnn_cuda_native",
            ),
        )
        return load_keydnn_cuda_native()
    except Exception:
        return None


def _get_cuda_utils_wrappers():
    """
    Import module-level CUDA utils wrappers.

    These should have signatures like:
      - cuda_set_device(lib, device=0)
      - cuda_malloc(lib, nbytes) -> int
      - cuda_free(lib, ptr)
      - cudaMemcpyHtoD(lib, dst_dev, src_np, nbytes)
      - cudaMemcpyDtoH(lib, dst_np, src_dev, nbytes)
      - cuda_synchronize(lib)
    """
    cuda_set_device = _import_first(
        (
            "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cuda_set_device",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cuda_set_device",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_set_device",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cuda_set_device",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cuda_set_device",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_set_device",
        ),
    )
    cuda_malloc = _import_first(
        (
            "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cuda_malloc",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cuda_malloc",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_malloc",
        ),
        ("keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes", "cuda_malloc"),
        ("keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes", "cuda_malloc"),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_malloc",
        ),
    )
    cuda_free = _import_first(
        ("src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes", "cuda_free"),
        ("src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes", "cuda_free"),
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_free",
        ),
        ("keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes", "cuda_free"),
        ("keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes", "cuda_free"),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_free",
        ),
    )
    cudaMemcpyHtoD = _import_first(
        (
            "src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cudaMemcpyHtoD",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cudaMemcpyHtoD",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cudaMemcpyHtoD",
        ),
        ("keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes", "cudaMemcpyHtoD"),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cudaMemcpyHtoD",
        ),
        ("keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes", "cudaMemcpyHtoD"),
    )
    cudaMemcpyDtoH = _import_first(
        (
            "src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cudaMemcpyDtoH",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cudaMemcpyDtoH",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cudaMemcpyDtoH",
        ),
        ("keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes", "cudaMemcpyDtoH"),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cudaMemcpyDtoH",
        ),
        ("keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes", "cudaMemcpyDtoH"),
    )
    cuda_synchronize = _import_first(
        (
            "src.keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_synchronize",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cuda_synchronize",
        ),
        (
            "src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cuda_synchronize",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
            "cuda_synchronize",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
            "cuda_synchronize",
        ),
        (
            "keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
            "cuda_synchronize",
        ),
    )
    return (
        cuda_set_device,
        cuda_malloc,
        cuda_free,
        cudaMemcpyHtoD,
        cudaMemcpyDtoH,
        cuda_synchronize,
    )


class _CudaArithmeticCtypesBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _try_load_cuda_lib()
        if cls.lib is None:
            raise unittest.SkipTest(
                "CUDA native library not available / failed to load."
            )

        # IMPORTANT: wrap as staticmethod to avoid unittest binding them as methods
        (
            cuda_set_device,
            cuda_malloc,
            cuda_free,
            cudaMemcpyHtoD,
            cudaMemcpyDtoH,
            cuda_synchronize,
        ) = _get_cuda_utils_wrappers()

        cls.cuda_set_device = staticmethod(cuda_set_device)
        cls.cuda_malloc = staticmethod(cuda_malloc)
        cls.cuda_free = staticmethod(cuda_free)
        cls.cudaMemcpyHtoD = staticmethod(cudaMemcpyHtoD)
        cls.cudaMemcpyDtoH = staticmethod(cudaMemcpyDtoH)
        cls.cuda_synchronize = staticmethod(cuda_synchronize)

        # ops under test (ctypes dispatchers)
        cls.neg_cuda = staticmethod(
            _import_first(
                (
                    "src.keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "neg_cuda",
                ),
                (
                    "keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "neg_cuda",
                ),
            )
        )
        cls.add_cuda = staticmethod(
            _import_first(
                (
                    "src.keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "add_cuda",
                ),
                (
                    "keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "add_cuda",
                ),
            )
        )
        cls.sub_cuda = staticmethod(
            _import_first(
                (
                    "src.keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "sub_cuda",
                ),
                (
                    "keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "sub_cuda",
                ),
            )
        )
        cls.div_cuda = staticmethod(
            _import_first(
                (
                    "src.keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "div_cuda",
                ),
                (
                    "keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "div_cuda",
                ),
            )
        )
        cls.gt_cuda = staticmethod(
            _import_first(
                (
                    "src.keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "gt_cuda",
                ),
                (
                    "keydnn.infrastructure.native_cuda.python.ops.tensor_arithmetic_ctypes",
                    "gt_cuda",
                ),
            )
        )

        # Set device once for the process (safe even if device=0 already)
        cls.cuda_set_device(cls.lib, 0)

    # ---- tiny helpers ----
    def _alloc(self, nbytes: int) -> int:
        return int(type(self).cuda_malloc(self.lib, int(nbytes)))

    def _free(self, ptr: int) -> None:
        type(self).cuda_free(self.lib, int(ptr))

    def _htod(self, dst: int, src: np.ndarray) -> None:
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)
        type(self).cudaMemcpyHtoD(self.lib, int(dst), src, int(src.nbytes))

    def _dtoh(self, dst: np.ndarray, src: int) -> None:
        if not dst.flags["C_CONTIGUOUS"]:
            dst = np.ascontiguousarray(dst)
        type(self).cudaMemcpyDtoH(self.lib, dst, int(src), int(dst.nbytes))

    def _sync(self) -> None:
        type(self).cuda_synchronize(self.lib)


class TestTensorArithmeticCtypesF32(_CudaArithmeticCtypesBase):
    def test_neg_f32_matches_numpy(self):
        x = np.array([1.0, -2.5, 3.25, 0.0], dtype=np.float32)
        y = np.empty_like(x)

        x_dev = self._alloc(x.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(x_dev, x)
            type(self).neg_cuda(
                self.lib, x_dev=x_dev, y_dev=y_dev, n=x.size, dtype=x.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(x_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, -x, rtol=0, atol=0)

    def test_add_f32_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
        b = np.array([0.5, -2.0, 10.0, -1.5], dtype=np.float32)
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).add_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a + b, rtol=0, atol=0)

    def test_sub_f32_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
        b = np.array([0.5, -2.0, 10.0, -1.5], dtype=np.float32)
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).sub_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a - b, rtol=0, atol=0)

    def test_div_f32_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
        b = np.array([0.5, -2.0, 10.0, -1.6], dtype=np.float32)  # avoid 0
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).div_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a / b, rtol=1e-6, atol=1e-6)

    def test_gt_f32_outputs_float32_mask(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
        b = np.array([0.5, 10.0, -3.0, 3.0], dtype=np.float32)
        y = np.empty((a.size,), dtype=np.float32)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).gt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        ref = (a > b).astype(np.float32)
        np.testing.assert_allclose(y, ref, rtol=0, atol=0)
        self.assertEqual(y.dtype, np.float32)


class TestTensorArithmeticCtypesF64(_CudaArithmeticCtypesBase):
    def test_neg_f64_matches_numpy(self):
        x = np.array([1.0, -2.5, 3.25, 0.0], dtype=np.float64)
        y = np.empty_like(x)

        x_dev = self._alloc(x.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(x_dev, x)
            type(self).neg_cuda(
                self.lib, x_dev=x_dev, y_dev=y_dev, n=x.size, dtype=x.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(x_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, -x, rtol=0, atol=0)

    def test_add_f64_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        b = np.array([0.5, -2.0, 10.0, -1.5], dtype=np.float64)
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).add_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a + b, rtol=0, atol=0)

    def test_sub_f64_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        b = np.array([0.5, -2.0, 10.0, -1.5], dtype=np.float64)
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).sub_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a - b, rtol=0, atol=0)

    def test_div_f64_matches_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        b = np.array([0.5, -2.0, 10.0, -1.6], dtype=np.float64)  # avoid 0
        y = np.empty_like(a)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).div_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        np.testing.assert_allclose(y, a / b, rtol=1e-12, atol=1e-12)

    def test_gt_f64_outputs_float32_mask(self):
        a = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        b = np.array([0.5, 10.0, -3.0, 3.0], dtype=np.float64)
        y = np.empty((a.size,), dtype=np.float32)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            type(self).gt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=a.size, dtype=a.dtype
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        ref = (a > b).astype(np.float32)
        np.testing.assert_allclose(y, ref, rtol=0, atol=0)
        self.assertEqual(y.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
