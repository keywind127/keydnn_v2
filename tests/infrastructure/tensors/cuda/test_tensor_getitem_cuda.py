# tests/infrastructure/tensors/cuda/test_tensor_getitem_cuda.py
from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor


def _cuda_available() -> bool:
    try:
        # Use a known-good loader in your tree
        from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (  # type: ignore
            load_keydnn_cuda_native,
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _make_cuda_tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool
) -> Tensor:
    """
    Create a CUDA tensor and copy host data into it via H2D memcpy.

    Assumes:
    - Tensor._ensure_cuda_alloc(dtype=...) allocates device storage into .data
    - memcpy_ctypes exposes memcpy_htod with keyword-only signature
    """
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t._ensure_cuda_alloc(dtype=arr.dtype)

    from src.keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc

    lib = t._get_cuda_lib()
    mc.memcpy_htod(
        lib,
        dst_dev=int(t.data),
        src_host=arr,
        nbytes=int(arr.nbytes),
        sync=True,
    )
    return t


def _cuda_to_numpy(t: Tensor) -> np.ndarray:
    """
    Copy CUDA tensor device buffer back to host and return ndarray.
    """
    dt = np.dtype(getattr(t, "dtype", np.float32))
    out = np.empty(t.shape, dtype=dt)

    from src.keydnn.infrastructure.native_cuda.python.ops import memcpy_ctypes as mc

    lib = t._get_cuda_lib()
    mc.memcpy_dtoh(
        lib,
        dst_host=out,
        src_dev=int(t.data),
        nbytes=int(out.nbytes),
        sync=True,
    )
    return out


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorGetItemCudaForward(unittest.TestCase):
    def setUp(self) -> None:
        self.dev = Device("cuda:0")
        np.random.seed(0)

    def test_getitem_cuda_forward_basic_slice_matches_numpy(self):
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=False)

        y = x[1:3, 2:5]
        got = _cuda_to_numpy(y)
        exp = x_np[1:3, 2:5]

        self.assertEqual(y.shape, exp.shape)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_getitem_cuda_forward_int_index_matches_numpy(self):
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=False)

        y = x[2]
        got = _cuda_to_numpy(y)
        exp = x_np[2]

        self.assertEqual(y.shape, exp.shape)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_getitem_cuda_forward_scalar_result_shape_is_empty_tuple(self):
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=False)

        y = x[1, 2]  # scalar
        got = _cuda_to_numpy(y)
        exp = np.array(x_np[1, 2], dtype=np.float32)

        self.assertEqual(y.shape, ())
        self.assertEqual(got.shape, ())
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_getitem_cuda_forward_fancy_index_matches_numpy(self):
        x_np = np.random.randn(4, 6).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=False)

        rows = np.array([3, 1, 1], dtype=np.int64)
        cols = np.array([0, 2, 5], dtype=np.int64)
        y = x[rows, cols]
        got = _cuda_to_numpy(y)
        exp = x_np[rows, cols]

        self.assertEqual(y.shape, exp.shape)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0)

    def test_to_cuda_from_noncontiguous_cpu_view_roundtrip_matches(self):
        x = Tensor(shape=(4, 6), device=self.dev, requires_grad=False, ctx=None)
        x_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        x.copy_from_numpy(x_np)

        # non-contiguous view (column slice)
        view = x.to_numpy()[:, 1:5]  # shape (4,4), typically non-contiguous
        self.assertFalse(view.flags["C_CONTIGUOUS"])

        v = Tensor(shape=view.shape, device=self.dev, requires_grad=False, ctx=None)
        v.copy_from_numpy(view)  # may store non-contig depending on your implementation

        v_cuda = v.to(self.dev, copy=True)
        v_back = v_cuda.to(self.dev, copy=True).to_numpy()

        np.testing.assert_allclose(v_back, np.ascontiguousarray(view), rtol=0, atol=0)


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestTensorGetItemCudaBackward(unittest.TestCase):
    def setUp(self) -> None:
        self.dev = Device("cuda:0")
        self.cpu = Device("cpu")
        np.random.seed(0)

    def test_getitem_cuda_backward_basic_slice_scatter_matches_numpy(self):
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=True)

        key = (slice(1, 4), slice(2, 5))
        y = x[key]

        # grad_out = ones like y (on CUDA)
        go_np = np.ones(y.shape, dtype=np.float32)
        grad_out = _make_cuda_tensor_from_numpy(
            go_np, device=self.dev, requires_grad=False
        )

        # Run backward fn directly (consistent with your other unit tests)
        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)

        grads = ctx.backward_fn(grad_out)  # type: ignore[attr-defined]
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        gx = grads[0]
        self.assertIsNotNone(gx)

        gx_np = _cuda_to_numpy(gx)  # type: ignore[arg-type]

        exp = np.zeros_like(x_np)
        exp[key] += go_np
        np.testing.assert_allclose(gx_np, exp, rtol=0, atol=0)

    def test_getitem_cuda_backward_fancy_index_accumulates_like_add_at(self):
        x_np = np.random.randn(4, 6).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=True)

        rows = np.array([1, 1, 3, 1], dtype=np.int64)  # repeats to test accumulation
        cols = np.array([2, 2, 0, 5], dtype=np.int64)
        key = (rows, cols)
        y = x[key]

        go_np = np.ones(y.shape, dtype=np.float32)
        grad_out = _make_cuda_tensor_from_numpy(
            go_np, device=self.dev, requires_grad=False
        )

        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)

        grads = ctx.backward_fn(grad_out)  # type: ignore[attr-defined]
        gx = grads[0]
        self.assertIsNotNone(gx)

        gx_np = _cuda_to_numpy(gx)  # type: ignore[arg-type]

        exp = np.zeros_like(x_np)
        np.add.at(exp, key, go_np)  # must accumulate at repeated indices
        np.testing.assert_allclose(gx_np, exp, rtol=0, atol=0)

    def test_getitem_cuda_backward_grad_out_wrong_device_raises(self):
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = _make_cuda_tensor_from_numpy(x_np, device=self.dev, requires_grad=True)

        y = x[1:3, 2:4]

        # grad_out on CPU should raise in CUDA backward
        go_np = np.ones(y.shape, dtype=np.float32)
        grad_out_cpu = Tensor(
            shape=go_np.shape, device=self.cpu, requires_grad=False, ctx=None
        )
        grad_out_cpu.copy_from_numpy(go_np)

        ctx = getattr(y, "_ctx", None) or getattr(y, "ctx", None)
        self.assertIsNotNone(ctx)

        with self.assertRaises(RuntimeError):
            _ = ctx.backward_fn(grad_out_cpu)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
