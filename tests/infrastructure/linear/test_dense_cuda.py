import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.fully_connected._dense import Dense
from src.keydnn.infrastructure.tensor._tensor import Tensor


# -----------------------------
# CUDA test utilities (same style as your Linear CUDA tests)
# -----------------------------
def _cuda_available() -> bool:
    try:
        _ = Tensor._get_cuda_lib()
        return True
    except Exception:
        return False


def _make_cuda_device(index: int = 0) -> Device:
    try:
        return Device(f"cuda:{index}")
    except Exception:
        return Device("cuda")


def _cuda_lib_and_ctypes():
    from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

    lib = Tensor._get_cuda_lib()
    return lib, m


def _cuda_set_device(device: Device) -> None:
    lib = Tensor._get_cuda_lib()
    from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
        cuda_set_device,
    )

    cuda_set_device(lib, int(getattr(device, "index", 0) or 0))


def _cuda_sync() -> None:
    lib, m = _cuda_lib_and_ctypes()
    if hasattr(m, "cuda_synchronize"):
        m.cuda_synchronize(lib)
        return

    # optional fallback if you ever expose direct export
    if hasattr(lib, "keydnn_cuda_synchronize"):
        import ctypes

        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        st = int(fn())
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_synchronize failed: status={st}")
        return


def _cuda_tensor_from_numpy(
    arr: np.ndarray, device: Device, *, requires_grad: bool = False
) -> Tensor:
    arr = np.asarray(arr)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32, copy=False)

    _cuda_set_device(device)

    t = Tensor(
        shape=arr.shape,
        device=device,
        requires_grad=requires_grad,
        ctx=None,
        dtype=np.dtype(arr.dtype),
    )
    t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))
    dev_ptr = int(t.data)
    if dev_ptr == 0 and arr.size != 0:
        raise RuntimeError("Failed to allocate CUDA tensor buffer (devptr == 0).")

    lib, m = _cuda_lib_and_ctypes()
    if not hasattr(m, "cudaMemcpyHtoD"):
        raise RuntimeError("Missing cudaMemcpyHtoD wrapper in maxpool2d_ctypes.")
    m.cudaMemcpyHtoD(lib, dev_ptr, arr, int(arr.nbytes))
    _cuda_sync()
    return t


def _cuda_readback(t: Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy())


@unittest.skipUnless(
    _cuda_available(), "CUDA native library not available; skipping CUDA Dense tests."
)
class TestDenseCuda(TestCase):
    def test_dense_cuda_builds_on_first_forward_and_exposes_parameters(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        x = Tensor(shape=(2, 3), device=dev)
        x._ensure_cuda_alloc(dtype=np.float32)

        y = d.forward(x)

        self.assertTrue(d.is_built)
        self.assertEqual(d.in_features, 3)
        self.assertEqual(y.shape, (2, 4))

        # After build, params should show up (weight + bias)
        params = list(d.parameters())
        unique = {id(p) for p in params}
        self.assertEqual(len(unique), 2)

        lin = d._linear
        self.assertIsNotNone(lin)
        self.assertEqual(lin.weight.shape, (4, 3))
        self.assertIsNotNone(lin.bias)
        self.assertEqual(lin.bias.shape, (4,))

        # Ensure param buffers exist
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)
        self.assertNotEqual(int(lin.weight.data), 0)
        self.assertNotEqual(int(lin.bias.data), 0)

    def test_dense_cuda_forward_rejects_non_2d_input(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        x = Tensor(shape=(3,), device=dev)
        x._ensure_cuda_alloc(dtype=np.float32)

        with self.assertRaises(ValueError):
            d.forward(x)

    def test_dense_cuda_forward_rejects_feature_mismatch_after_built(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        x1 = Tensor(shape=(2, 3), device=dev)
        x1._ensure_cuda_alloc(dtype=np.float32)
        _ = d.forward(x1)

        x2 = Tensor(shape=(2, 5), device=dev)
        x2._ensure_cuda_alloc(dtype=np.float32)
        with self.assertRaises(RuntimeError):
            d.forward(x2)

    def test_dense_cuda_device_mismatch_raises_when_device_specified(self):
        dev0 = _make_cuda_device(0)

        # Try to see if cuda:1 is usable; otherwise skip.
        try:
            dev1 = _make_cuda_device(1)
            # attempt a minimal set_device to validate dev1 exists
            _cuda_set_device(dev1)
        except Exception as e:
            self.skipTest(
                f"Second CUDA device not available (cuda:1). Skipping. Reason: {e}"
            )

        d = Dense(4, bias=True, device=dev0)

        # x on dev1 should raise device mismatch before/inside forward
        x = Tensor(shape=(2, 3), device=dev1)
        x._ensure_cuda_alloc(dtype=np.float32)

        with self.assertRaises(RuntimeError):
            d.forward(x)

    def test_dense_cuda_forward_outputs_expected_shape_and_values_zero_params(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        # build
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=False)
        _ = d.forward(x)

        lin = d._linear
        self.assertIsNotNone(lin)

        # ensure buffers
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        # write zeros into params
        W0 = np.zeros((4, 3), dtype=np.float32)
        b0 = np.zeros((4,), dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W0, int(W0.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b0, int(b0.nbytes))
        _cuda_sync()

        y = d.forward(x)
        self.assertEqual(y.shape, (2, 4))

        y_np = _cuda_readback(y)
        np.testing.assert_allclose(
            y_np, np.zeros((2, 4), dtype=np.float32), rtol=0, atol=0
        )

    def test_dense_cuda_forward_matches_numpy_reference(self):
        dev = _make_cuda_device(0)
        d = Dense(2, bias=True, device=dev)

        # build with in_features=3
        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=False)
        _ = d.forward(x)

        lin = d._linear
        self.assertIsNotNone(lin)

        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        W = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W, int(W.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b, int(b.nbytes))
        _cuda_sync()

        y = d.forward(x)
        y_np = _cuda_readback(y)

        ref = x_np @ W.T + b
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y_np, ref, rtol=1e-5, atol=1e-5)

    def test_dense_cuda_forward_attaches_context_when_requires_grad(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        x = _cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), dev, requires_grad=True
        )
        out = d.forward(x)

        self.assertTrue(out.requires_grad)
        ctx = out._get_ctx()
        self.assertIsNotNone(ctx)
        self.assertTrue(callable(ctx.backward_fn))

        lin = d._linear
        self.assertIsNotNone(lin)

        # parents order: (x, weight, bias)
        self.assertEqual(len(ctx.parents), 3)
        self.assertIs(ctx.parents[0], x)
        self.assertIs(ctx.parents[1], lin.weight)
        self.assertIs(ctx.parents[2], lin.bias)

        # saved_tensors: (x, weight)
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], x)
        self.assertIs(ctx.saved_tensors[1], lin.weight)

        grad_out = _cuda_tensor_from_numpy(
            np.ones(out.shape, dtype=np.float32), dev, requires_grad=False
        )
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 3)
        grad_x, grad_w, grad_b = grads

        self.assertIsNotNone(grad_x)
        self.assertIsNotNone(grad_w)
        self.assertIsNotNone(grad_b)

        self.assertTrue(grad_x.device.is_cuda())
        self.assertTrue(grad_w.device.is_cuda())
        self.assertTrue(grad_b.device.is_cuda())

        self.assertEqual(grad_x.shape, x.shape)
        self.assertEqual(grad_w.shape, lin.weight.shape)
        self.assertEqual(grad_b.shape, lin.bias.shape)

    def test_dense_cuda_backward_matches_numpy_reference(self):
        """
        End-to-end autograd on CUDA:
          out = d(x)
          out.backward(ones)
        Check x.grad, weight.grad, bias.grad numerically by readback.

        Assumes your CUDA backward works for this graph and bias grad uses CUDA axis reduction.
        """
        dev = _make_cuda_device(0)
        d = Dense(2, bias=True, device=dev)

        # Build
        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=True)
        out = d.forward(x)

        lin = d._linear
        self.assertIsNotNone(lin)

        # Ensure params allocated and set deterministic values
        lin.weight.requires_grad = True
        lin.bias.requires_grad = True
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        W = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W, int(W.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b, int(b.nbytes))
        _cuda_sync()

        # Re-run forward with deterministic params
        out = d.forward(x)

        go_np = np.ones(out.shape, dtype=np.float32)
        go = _cuda_tensor_from_numpy(go_np, dev, requires_grad=False)
        out.backward(go)

        gx = _cuda_readback(x.grad)
        gW = _cuda_readback(lin.weight.grad)
        gb = _cuda_readback(lin.bias.grad)

        ref_gx = go_np @ W
        ref_gW = go_np.T @ x_np
        ref_gb = go_np.sum(axis=0)

        np.testing.assert_allclose(gx, ref_gx, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gW, ref_gW, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gb, ref_gb, rtol=1e-4, atol=1e-4)

    def test_dense_cuda_forward_no_context_when_no_requires_grad(self):
        dev = _make_cuda_device(0)
        d = Dense(4, bias=True, device=dev)

        x = _cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), dev, requires_grad=False
        )

        # Build once
        _ = d.forward(x)

        lin = d._linear
        self.assertIsNotNone(lin)

        lin.weight.requires_grad = False
        if lin.bias is not None:
            lin.bias.requires_grad = False

        # Ensure param buffers exist (avoid "data == 0" checks)
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        if lin.bias is not None:
            lin.bias._ensure_cuda_alloc(dtype=np.float32)

        out = d.forward(x)

        self.assertFalse(out.requires_grad)
        self.assertIsNone(
            out._get_ctx(), "Expected no Context when nothing requires grad."
        )


if __name__ == "__main__":
    unittest.main()
