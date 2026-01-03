# tests/infrastructure/native_cuda/ops/test_tensor_comparison_ctypes.py
from __future__ import annotations

import unittest
from typing import Any, Tuple

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


class _CudaComparisonCtypesBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _try_load_cuda_lib()
        if cls.lib is None:
            raise unittest.SkipTest(
                "CUDA native library not available / failed to load."
            )

        (
            cuda_set_device,
            cuda_malloc,
            cuda_free,
            cudaMemcpyHtoD,
            cudaMemcpyDtoH,
            cuda_synchronize,
        ) = _get_cuda_utils_wrappers()

        # IMPORTANT: wrap as staticmethod to avoid unittest binding them as methods
        cls.cuda_set_device = staticmethod(cuda_set_device)
        cls.cuda_malloc = staticmethod(cuda_malloc)
        cls.cuda_free = staticmethod(cuda_free)
        cls.cudaMemcpyHtoD = staticmethod(cudaMemcpyHtoD)
        cls.cudaMemcpyDtoH = staticmethod(cudaMemcpyDtoH)
        cls.cuda_synchronize = staticmethod(cuda_synchronize)

        # ops under test
        mod_candidates = (
            (
                "src.keydnn.infrastructure.native_cuda.python.ops.tensor_comparison_ctypes",
            ),
            ("keydnn.infrastructure.native_cuda.python.ops.tensor_comparison_ctypes",),
        )

        def _op(name: str):
            return staticmethod(
                _import_first(
                    (mod_candidates[0][0], name),
                    (mod_candidates[1][0], name),
                )
            )

        # elementwise compare
        cls.gt_cuda = _op("gt_cuda")
        cls.ge_cuda = _op("ge_cuda")
        cls.lt_cuda = _op("lt_cuda")
        cls.le_cuda = _op("le_cuda")
        cls.eq_cuda = _op("eq_cuda")
        cls.ne_cuda = _op("ne_cuda")

        # scalar compare
        cls.gt_scalar_cuda = _op("gt_scalar_cuda")
        cls.ge_scalar_cuda = _op("ge_scalar_cuda")
        cls.lt_scalar_cuda = _op("lt_scalar_cuda")
        cls.le_scalar_cuda = _op("le_scalar_cuda")
        cls.eq_scalar_cuda = _op("eq_scalar_cuda")
        cls.ne_scalar_cuda = _op("ne_scalar_cuda")

        cls.cuda_set_device(cls.lib, 0)

    # ---- tiny helpers ----
    def _alloc(self, nbytes: int) -> int:
        # IMPORTANT: some allocators reject nbytes==0; allocate 1 byte for "empty" tests.
        nbytes = int(nbytes)
        if nbytes <= 0:
            nbytes = 1
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

    # ---- shared compare helpers ----
    def _run_binary_cmp(self, op_name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        assert a.shape == b.shape
        y = np.empty((a.size,), dtype=np.float32)

        a_dev = self._alloc(a.nbytes)
        b_dev = self._alloc(b.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            self._htod(b_dev, b)
            getattr(type(self), op_name)(
                self.lib,
                a_dev=a_dev,
                b_dev=b_dev,
                y_dev=y_dev,
                n=a.size,
                dtype=a.dtype,
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)

        self.assertEqual(y.dtype, np.float32)
        return y.reshape(a.shape)

    def _run_scalar_cmp(self, op_name: str, a: np.ndarray, alpha: float) -> np.ndarray:
        y = np.empty((a.size,), dtype=np.float32)

        a_dev = self._alloc(a.nbytes)
        y_dev = self._alloc(y.nbytes)
        try:
            self._htod(a_dev, a)
            getattr(type(self), op_name)(
                self.lib,
                a_dev=a_dev,
                alpha=float(alpha),
                y_dev=y_dev,
                n=a.size,
                dtype=a.dtype,
            )
            self._sync()
            self._dtoh(y, y_dev)
        finally:
            self._free(a_dev)
            self._free(y_dev)

        self.assertEqual(y.dtype, np.float32)
        return y.reshape(a.shape)


class TestTensorComparisonCtypesF32(_CudaComparisonCtypesBase):
    def test_elementwise_all_ops_f32_match_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float32)
        b = np.array([0.5, 2.0, -10.0, 10.0, 0.0], dtype=np.float32)

        cases = [
            ("gt_cuda", (a > b).astype(np.float32)),
            ("ge_cuda", (a >= b).astype(np.float32)),
            ("lt_cuda", (a < b).astype(np.float32)),
            ("le_cuda", (a <= b).astype(np.float32)),
            ("eq_cuda", (a == b).astype(np.float32)),
            ("ne_cuda", (a != b).astype(np.float32)),
        ]

        for name, ref in cases:
            y = self._run_binary_cmp(name, a, b)
            np.testing.assert_allclose(y, ref, rtol=0, atol=0)

    def test_scalar_all_ops_f32_match_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float32)
        alpha = 2.0

        cases = [
            ("gt_scalar_cuda", (a > alpha).astype(np.float32)),
            ("ge_scalar_cuda", (a >= alpha).astype(np.float32)),
            ("lt_scalar_cuda", (a < alpha).astype(np.float32)),
            ("le_scalar_cuda", (a <= alpha).astype(np.float32)),
            ("eq_scalar_cuda", (a == alpha).astype(np.float32)),
            ("ne_scalar_cuda", (a != alpha).astype(np.float32)),
        ]

        for name, ref in cases:
            y = self._run_scalar_cmp(name, a, alpha)
            np.testing.assert_allclose(y, ref, rtol=0, atol=0)

    def test_non_contiguous_inputs_f32_are_ok(self):
        # Ensure our test harness makes contiguous copies before HtoD.
        base = np.array([[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]], dtype=np.float32)
        a = base[:, ::2]  # non-contiguous (shape (2,2))
        b = np.array([[0.0, 3.0], [4.0, -2.0]], dtype=np.float32)

        y = self._run_binary_cmp("gt_cuda", a, b)
        ref = (a > b).astype(np.float32)
        np.testing.assert_allclose(y, ref, rtol=0, atol=0)

    def test_numel_zero_is_ok_f32(self):
        # allocate 1 byte but pass n=0 => should be a no-op and not crash.
        a_dev = self._alloc(0)
        b_dev = self._alloc(0)
        y_dev = self._alloc(0)
        try:
            type(self).gt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).ge_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).lt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).le_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).eq_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).ne_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float32
            )

            type(self).gt_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).ge_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).lt_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).le_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).eq_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            type(self).ne_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float32
            )
            self._sync()
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)


class TestTensorComparisonCtypesF64(_CudaComparisonCtypesBase):
    def test_elementwise_all_ops_f64_match_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float64)
        b = np.array([0.5, 2.0, -10.0, 10.0, 0.0], dtype=np.float64)

        cases = [
            ("gt_cuda", (a > b).astype(np.float32)),
            ("ge_cuda", (a >= b).astype(np.float32)),
            ("lt_cuda", (a < b).astype(np.float32)),
            ("le_cuda", (a <= b).astype(np.float32)),
            ("eq_cuda", (a == b).astype(np.float32)),
            ("ne_cuda", (a != b).astype(np.float32)),
        ]

        for name, ref in cases:
            y = self._run_binary_cmp(name, a, b)
            np.testing.assert_allclose(y, ref, rtol=0, atol=0)
            self.assertEqual(y.dtype, np.float32)

    def test_scalar_all_ops_f64_match_numpy(self):
        a = np.array([1.0, 2.0, -3.0, 4.0, 0.0], dtype=np.float64)
        alpha = 2.0

        cases = [
            ("gt_scalar_cuda", (a > alpha).astype(np.float32)),
            ("ge_scalar_cuda", (a >= alpha).astype(np.float32)),
            ("lt_scalar_cuda", (a < alpha).astype(np.float32)),
            ("le_scalar_cuda", (a <= alpha).astype(np.float32)),
            ("eq_scalar_cuda", (a == alpha).astype(np.float32)),
            ("ne_scalar_cuda", (a != alpha).astype(np.float32)),
        ]

        for name, ref in cases:
            y = self._run_scalar_cmp(name, a, alpha)
            np.testing.assert_allclose(y, ref, rtol=0, atol=0)
            self.assertEqual(y.dtype, np.float32)

    def test_numel_zero_is_ok_f64(self):
        a_dev = self._alloc(0)
        b_dev = self._alloc(0)
        y_dev = self._alloc(0)
        try:
            type(self).gt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).ge_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).lt_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).le_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).eq_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).ne_cuda(
                self.lib, a_dev=a_dev, b_dev=b_dev, y_dev=y_dev, n=0, dtype=np.float64
            )

            type(self).gt_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).ge_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).lt_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).le_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).eq_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            type(self).ne_scalar_cuda(
                self.lib, a_dev=a_dev, alpha=0.0, y_dev=y_dev, n=0, dtype=np.float64
            )
            self._sync()
        finally:
            self._free(a_dev)
            self._free(b_dev)
            self._free(y_dev)


if __name__ == "__main__":
    unittest.main()
