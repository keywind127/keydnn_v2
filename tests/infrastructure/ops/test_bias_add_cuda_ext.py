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

# import unittest
# import numpy as np


# def _cuda_available() -> bool:
#     try:
#         # pick a known-good loader in your tree
#         from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
#             load_keydnn_cuda_native,  # type: ignore
#         )

#         _ = load_keydnn_cuda_native()
#         return True
#     except Exception:
#         return False


# def _alloc_cuda_tensor_from_numpy(arr: np.ndarray, *, device_str: str = "cuda:0"):
#     from src.keydnn.infrastructure.tensor._tensor import Tensor
#     from src.keydnn.domain.device._device import Device

#     t = Tensor(
#         shape=arr.shape, device=Device(device_str), requires_grad=False, dtype=arr.dtype
#     )
#     t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
#     lib = t._get_cuda_lib()

#     # import memcpy wrapper from your existing ctypes module set
#     from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
#         cudaMemcpyHtoD,  # type: ignore
#     )

#     cudaMemcpyHtoD(lib, int(t.data), arr, int(arr.nbytes))
#     return t


# def _to_numpy_cuda_tensor(t) -> np.ndarray:
#     return t.to_numpy()


# def _find_param(obj, names: tuple[str, ...]):
#     for n in names:
#         if hasattr(obj, n):
#             return getattr(obj, n)
#     return None


# def _get_grad_tensor(param):
#     # prefer public .grad if present, otherwise fall back to _grad
#     g = getattr(param, "grad", None)
#     if g is None:
#         g = getattr(param, "_grad", None)
#     return g


# @unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
# class TestBiasAddCuda(unittest.TestCase):
#     def test_bias_add_forward_cuda_matches_numpy_f32(self):
#         self._bias_add_forward_matches_numpy(np.float32)

#     def test_bias_add_forward_cuda_matches_numpy_f64(self):
#         self._bias_add_forward_matches_numpy(np.float64)

#     def _bias_add_forward_matches_numpy(self, dtype: np.dtype) -> None:
#         from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

#         rng = np.random.default_rng(0)
#         batch = 64
#         out = 128

#         x = rng.standard_normal((batch, out)).astype(dtype, copy=False)
#         b = rng.standard_normal((out,)).astype(dtype, copy=False)

#         x_t = _alloc_cuda_tensor_from_numpy(x)
#         b_t = _alloc_cuda_tensor_from_numpy(b)

#         y_t = bias_add_forward(x_t, b_t, device=0, sync=True)
#         y = _to_numpy_cuda_tensor(y_t)

#         ref = x + b[None, :]
#         np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

#     def test_bias_add_inplace_cuda_matches_numpy_f32(self):
#         self._bias_add_inplace_matches_numpy(np.float32)

#     def test_bias_add_inplace_cuda_matches_numpy_f64(self):
#         self._bias_add_inplace_matches_numpy(np.float64)

#     def _bias_add_inplace_matches_numpy(self, dtype: np.dtype) -> None:
#         from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_inplace

#         rng = np.random.default_rng(1)
#         batch = 32
#         out = 257  # odd size to shake out indexing bugs

#         y0 = rng.standard_normal((batch, out)).astype(dtype, copy=False)
#         b = rng.standard_normal((out,)).astype(dtype, copy=False)

#         y_t = _alloc_cuda_tensor_from_numpy(y0.copy())
#         b_t = _alloc_cuda_tensor_from_numpy(b)

#         bias_add_inplace(y_t, b_t, device=0, sync=True)
#         y = _to_numpy_cuda_tensor(y_t)

#         ref = y0 + b[None, :]
#         np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)

#     def test_bias_add_raises_on_bad_shapes(self):
#         from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

#         x = np.ones((4, 8), dtype=np.float32)
#         b = np.ones((7,), dtype=np.float32)  # wrong out_features

#         x_t = _alloc_cuda_tensor_from_numpy(x)
#         b_t = _alloc_cuda_tensor_from_numpy(b)

#         with self.assertRaises(ValueError):
#             _ = bias_add_forward(x_t, b_t, device=0, sync=True)

#     def test_bias_add_raises_on_bad_dtype(self):
#         from src.keydnn.infrastructure.ops.bias_add_cuda_ext import bias_add_forward

#         x = np.ones((4, 8), dtype=np.float32)
#         b = np.ones((8,), dtype=np.float64)  # mismatch

#         x_t = _alloc_cuda_tensor_from_numpy(x)
#         b_t = _alloc_cuda_tensor_from_numpy(b)

#         with self.assertRaises(TypeError):
#             _ = bias_add_forward(x_t, b_t, device=0, sync=True)

#     # ------------------------------------------------------------------
#     # NEW: backward/gradient tests (this is what would catch your XOR bug)
#     # ------------------------------------------------------------------

#     def test_linear_bias_backward_cuda_matches_numpy_f32(self):
#         self._linear_backward_matches_numpy(dtype=np.float32, bias=True)

#     def test_linear_bias_backward_cuda_matches_numpy_f64(self):
#         self._linear_backward_matches_numpy(dtype=np.float64, bias=True)

#     def test_linear_no_bias_backward_cuda_matches_numpy_f32(self):
#         self._linear_backward_matches_numpy(dtype=np.float32, bias=False)

#     def test_linear_no_bias_backward_cuda_matches_numpy_f64(self):
#         self._linear_backward_matches_numpy(dtype=np.float64, bias=False)

#     def _linear_backward_matches_numpy(self, *, dtype: np.dtype, bias: bool) -> None:
#         """
#         Sanity-check CUDA backward for Linear(+bias) against NumPy reference grads.

#         We force a controlled upstream gradient by defining:
#             loss = sum(y * gy)
#         so dY = gy exactly.

#         Reference:
#             y  = x @ W.T + b
#             dW = dY.T @ x
#             db = sum(dY, axis=0)
#             dx = dY @ W
#         """
#         from src.keydnn.infrastructure._linear import Linear
#         from src.keydnn.infrastructure.tensor._tensor import Tensor
#         from src.keydnn.domain.device._device import Device

#         rng = np.random.default_rng(123)

#         batch = 33
#         in_features = 7
#         out_features = 11

#         x_np = rng.standard_normal((batch, in_features)).astype(dtype, copy=False)
#         w_np = rng.standard_normal((out_features, in_features)).astype(
#             dtype, copy=False
#         )
#         b_np = (
#             rng.standard_normal((out_features,)).astype(dtype, copy=False)
#             if bias
#             else None
#         )
#         gy_np = rng.standard_normal((batch, out_features)).astype(dtype, copy=False)

#         device = Device("cuda:0")

#         # --- Build module on CUDA ---
#         # Try to pass dtype if Linear supports it; otherwise fall back.
#         try:
#             lin = Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)  # type: ignore[call-arg]
#         except TypeError:
#             lin = Linear(in_features, out_features, bias=bias, device=device)

#         # Locate weight/bias tensors
#         w_t = _find_param(lin, ("weight", "W", "w"))
#         self.assertIsNotNone(
#             w_t, "Could not find Linear weight tensor attribute (weight/W/w)."
#         )

#         b_t = _find_param(lin, ("bias", "b")) if bias else None
#         if bias:
#             self.assertIsNotNone(
#                 b_t, "Could not find Linear bias tensor attribute (bias/b)."
#             )

#         # --- Ensure parameter dtype matches requested dtype ---
#         # Many implementations default CUDA params to float32.
#         w_dtype = np.dtype(getattr(w_t, "dtype", np.float32))
#         if w_dtype != np.dtype(dtype):
#             # If Tensor has a dtype conversion API, prefer it.
#             if hasattr(w_t, "to") and callable(getattr(w_t, "to")):
#                 try:
#                     w_t = w_t.to(dtype=dtype)  # type: ignore[call-arg]
#                     # re-attach into module if possible
#                     if hasattr(lin, "weight"):
#                         lin.weight = w_t  # type: ignore[attr-defined]
#                     elif hasattr(lin, "W"):
#                         lin.W = w_t  # type: ignore[attr-defined]
#                 except Exception:
#                     pass

#             # Re-check; if still mismatched, skip float64 because backend is float32-only
#             w_dtype2 = np.dtype(getattr(w_t, "dtype", np.float32))
#             if w_dtype2 != np.dtype(dtype):
#                 if dtype == np.float64:
#                     self.skipTest(
#                         "CUDA Linear parameters appear to be float32-only in this build "
#                         "(weight.dtype != float64). Skipping float64 backward parity test."
#                     )
#                 else:
#                     self.fail(
#                         f"Unexpected weight dtype mismatch: weight.dtype={w_dtype2} vs requested={dtype}"
#                     )

#         if bias and b_t is not None:
#             b_dtype = np.dtype(getattr(b_t, "dtype", np.float32))
#             if b_dtype != np.dtype(dtype):
#                 if hasattr(b_t, "to") and callable(getattr(b_t, "to")):
#                     try:
#                         b_t = b_t.to(dtype=dtype)  # type: ignore[call-arg]
#                         if hasattr(lin, "bias"):
#                             lin.bias = b_t  # type: ignore[attr-defined]
#                         elif hasattr(lin, "b"):
#                             lin.b = b_t  # type: ignore[attr-defined]
#                     except Exception:
#                         pass
#                 b_dtype2 = np.dtype(getattr(b_t, "dtype", np.float32))
#                 if b_dtype2 != np.dtype(dtype):
#                     if dtype == np.float64:
#                         self.skipTest(
#                             "CUDA Linear bias appears to be float32-only in this build "
#                             "(bias.dtype != float64). Skipping float64 backward parity test."
#                         )
#                     else:
#                         self.fail(
#                             f"Unexpected bias dtype mismatch: bias.dtype={b_dtype2} vs requested={dtype}"
#                         )

#         # Copy known weights/bias into module params (now dtype-aligned)
#         w_t.copy_from_numpy(w_np)
#         if bias and b_t is not None:
#             b_t.copy_from_numpy(b_np)

#         # Inputs must require grad if we want dx checked
#         x = Tensor(shape=x_np.shape, device=device, requires_grad=True, dtype=dtype)
#         x.copy_from_numpy(x_np)

#         # Forward
#         y = lin.forward(x) if hasattr(lin, "forward") else lin(x)

#         # Upstream gradient tensor on CUDA
#         gy = Tensor(shape=gy_np.shape, device=device, requires_grad=False, dtype=dtype)
#         gy.copy_from_numpy(gy_np)

#         # loss = sum(y * gy)  => dY = gy
#         prod = y * gy
#         if hasattr(prod, "sum"):
#             loss = prod.sum()
#         else:
#             self.assertTrue(
#                 hasattr(prod, "mean"),
#                 "Need Tensor.sum() or Tensor.mean() for this test.",
#             )
#             loss = prod.mean() * float(prod.shape[0] * prod.shape[1])

#         loss.backward()

#         # Grab grads
#         dx_t = _get_grad_tensor(x)
#         dw_t = _get_grad_tensor(w_t)
#         db_t = _get_grad_tensor(b_t) if bias else None

#         self.assertIsNotNone(dw_t, "Expected weight grad to be populated on CUDA.")
#         self.assertIsNotNone(dx_t, "Expected input grad to be populated on CUDA.")
#         if bias:
#             self.assertIsNotNone(db_t, "Expected bias grad to be populated on CUDA.")

#         dx = dx_t.to_numpy()
#         dw = dw_t.to_numpy()
#         db = db_t.to_numpy() if bias else None

#         # NumPy reference
#         dY = gy_np
#         dw_ref = dY.T @ x_np
#         dx_ref = dY @ w_np
#         db_ref = dY.sum(axis=0) if bias else None

#         if dtype == np.float64:
#             rtol, atol = 1e-10, 1e-10
#         else:
#             rtol, atol = 1e-4, 1e-4

#         np.testing.assert_allclose(dw, dw_ref, rtol=rtol, atol=atol)
#         np.testing.assert_allclose(dx, dx_ref, rtol=rtol, atol=atol)
#         if bias:
#             np.testing.assert_allclose(db, db_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
