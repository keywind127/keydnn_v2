from __future__ import annotations

import unittest
import numpy as np


def _cuda_available() -> bool:
    try:
        # pick a known-good loader in your tree
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _alloc_cuda_tensor_from_numpy(arr: np.ndarray, *, device_str: str = "cuda:0"):
    from src.keydnn.infrastructure.tensor._tensor import Tensor
    from src.keydnn.domain.device._device import Device

    t = Tensor(
        shape=arr.shape, device=Device(device_str), requires_grad=False, dtype=arr.dtype
    )
    t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    lib = t._get_cuda_lib()

    # import memcpy wrapper from your existing ctypes module set
    from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
        cudaMemcpyHtoD,  # type: ignore
    )

    cudaMemcpyHtoD(lib, int(t.data), arr, int(arr.nbytes))
    return t


def _to_numpy_cuda_tensor(t) -> np.ndarray:
    return t.to_numpy()


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestBiasAddCuda(unittest.TestCase):
    def test_bias_add_forward_cuda_matches_numpy_f32(self):
        self._bias_add_forward_matches_numpy(np.float32)

    def test_bias_add_forward_cuda_matches_numpy_f64(self):
        self._bias_add_forward_matches_numpy(np.float64)

    def _bias_add_forward_matches_numpy(self, dtype: np.dtype) -> None:
        from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

        rng = np.random.default_rng(0)
        batch = 64
        out = 128

        x = rng.standard_normal((batch, out)).astype(dtype, copy=False)
        b = rng.standard_normal((out,)).astype(dtype, copy=False)

        x_t = _alloc_cuda_tensor_from_numpy(x)
        b_t = _alloc_cuda_tensor_from_numpy(b)

        y_t = bias_add_forward(x_t, b_t, device=0, sync=True)
        y = _to_numpy_cuda_tensor(y_t)

        ref = x + b[None, :]
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_bias_add_inplace_cuda_matches_numpy_f32(self):
        self._bias_add_inplace_matches_numpy(np.float32)

    def test_bias_add_inplace_cuda_matches_numpy_f64(self):
        self._bias_add_inplace_matches_numpy(np.float64)

    def _bias_add_inplace_matches_numpy(self, dtype: np.dtype) -> None:
        from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_inplace

        rng = np.random.default_rng(1)
        batch = 32
        out = 257  # odd size to shake out indexing bugs

        y0 = rng.standard_normal((batch, out)).astype(dtype, copy=False)
        b = rng.standard_normal((out,)).astype(dtype, copy=False)

        y_t = _alloc_cuda_tensor_from_numpy(y0.copy())
        b_t = _alloc_cuda_tensor_from_numpy(b)

        bias_add_inplace(y_t, b_t, device=0, sync=True)
        y = _to_numpy_cuda_tensor(y_t)

        ref = y0 + b[None, :]
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

    def test_bias_add_raises_on_bad_shapes(self):
        from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

        x = np.ones((4, 8), dtype=np.float32)
        b = np.ones((7,), dtype=np.float32)  # wrong out_features

        x_t = _alloc_cuda_tensor_from_numpy(x)
        b_t = _alloc_cuda_tensor_from_numpy(b)

        with self.assertRaises(ValueError):
            _ = bias_add_forward(x_t, b_t, device=0, sync=True)

    def test_bias_add_raises_on_bad_dtype(self):
        from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

        x = np.ones((4, 8), dtype=np.float32)
        b = np.ones((8,), dtype=np.float64)  # mismatch

        x_t = _alloc_cuda_tensor_from_numpy(x)
        b_t = _alloc_cuda_tensor_from_numpy(b)

        with self.assertRaises(TypeError):
            _ = bias_add_forward(x_t, b_t, device=0, sync=True)


if __name__ == "__main__":
    unittest.main()
