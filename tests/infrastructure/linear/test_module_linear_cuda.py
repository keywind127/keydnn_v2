import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.fully_connected._linear import Linear
from src.keydnn.infrastructure.tensor._tensor import Tensor


# -----------------------------
# CUDA test utilities
# -----------------------------
def _cuda_available() -> bool:
    """
    Best-effort check: can we load the native CUDA DLL via Tensor._get_cuda_lib()?
    """
    try:
        _ = Tensor._get_cuda_lib()
        return True
    except Exception:
        return False


def _make_cuda_device(index: int = 0) -> Device:
    # Your Device likely accepts "cuda" or "cuda:0". Prefer explicit index.
    try:
        return Device(f"cuda:{index}")
    except Exception:
        return Device("cuda")


def _cuda_lib_and_ctypes():
    """
    Centralize imports so tests fail/skip cleanly if wrappers are unavailable.
    """
    from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as m

    lib = Tensor._get_cuda_lib()
    return lib, m


def _cuda_set_device(device: Device) -> None:
    lib = Tensor._get_cuda_lib()
    # You have cuda_set_device wrapper in avgpool2d_ctypes, but for tests we can
    # use whatever your native exports are. Prefer your wrapper if available.
    from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
        cuda_set_device,
    )

    cuda_set_device(lib, int(getattr(device, "index", 0) or 0))


def _cuda_sync() -> None:
    lib, m = _cuda_lib_and_ctypes()
    if hasattr(m, "cuda_synchronize"):
        m.cuda_synchronize(lib)
    elif hasattr(lib, "keydnn_cuda_synchronize"):
        # fall back if direct export exists
        import ctypes

        fn = lib.keydnn_cuda_synchronize
        fn.argtypes = []
        fn.restype = ctypes.c_int
        st = int(fn())
        if st != 0:
            raise RuntimeError(f"keydnn_cuda_synchronize failed: status={st}")


def _cuda_tensor_from_numpy(
    arr: np.ndarray, device: Device, *, requires_grad: bool = False
) -> Tensor:
    """
    Allocate a CUDA Tensor and write host contents using cudaMemcpyHtoD.
    """
    arr = np.asarray(arr)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32, copy=False)

    _cuda_set_device(device)

    # Allocate dev buffer
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

    # Copy H->D
    lib, m = _cuda_lib_and_ctypes()
    if not hasattr(m, "cudaMemcpyHtoD"):
        raise RuntimeError("Missing cudaMemcpyHtoD wrapper in maxpool2d_ctypes.")
    m.cudaMemcpyHtoD(lib, dev_ptr, arr, int(arr.nbytes))
    _cuda_sync()
    return t


def _cuda_readback(t: Tensor) -> np.ndarray:
    """
    Read back a CUDA tensor to host (uses Tensor.to_numpy()).
    """
    out = t.to_numpy()
    return np.asarray(out)


# -----------------------------
# CUDA Linear tests
# -----------------------------
@unittest.skipUnless(
    _cuda_available(), "CUDA native library not available; skipping CUDA Linear tests."
)
class TestLinearCuda(TestCase):
    def test_linear_cuda_forward_rejects_non_2d_input(self):
        dev = _make_cuda_device(0)
        lin = Linear(3, 4, bias=True, device=dev)

        x = Tensor(shape=(3,), device=dev)  # 1D
        # Ensure x has a dev buffer so we don't fail earlier for "data == 0" depending on your forward checks.
        x._ensure_cuda_alloc(dtype=np.float32)

        with self.assertRaises(ValueError):
            lin.forward(x)

    def test_linear_cuda_forward_rejects_feature_mismatch(self):
        dev = _make_cuda_device(0)
        lin = Linear(3, 4, bias=True, device=dev)

        x = Tensor(shape=(2, 5), device=dev)  # mismatch
        x._ensure_cuda_alloc(dtype=np.float32)

        with self.assertRaises(ValueError):
            lin.forward(x)

    def test_linear_cuda_forward_outputs_expected_shape_and_values_zero_params(self):
        """
        Deterministic check: if W=0 and b=0, output must be all zeros.

        NOTE: Since CUDA Parameters are not settable via copy_from_numpy(),
        we write their device buffers via cudaMemcpyHtoD.
        """
        dev = _make_cuda_device(0)
        lin = Linear(3, 4, bias=True, device=dev)

        # Make sure param buffers exist (Linear init creates Parameter tensors; but CUDA may have data==0).
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        # Write zeros into weight/bias
        W0 = np.zeros((4, 3), dtype=np.float32)
        b0 = np.zeros((4,), dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W0, int(W0.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b0, int(b0.nbytes))
        _cuda_sync()

        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=False)

        y = lin.forward(x)
        self.assertEqual(y.shape, (2, 4))

        y_np = _cuda_readback(y)
        np.testing.assert_allclose(
            y_np, np.zeros((2, 4), dtype=np.float32), rtol=0, atol=0
        )

    def test_linear_cuda_forward_matches_numpy_reference(self):
        """
        Numeric correctness on CUDA via readback:
          y = x @ W^T + b
        """
        dev = _make_cuda_device(0)
        lin = Linear(3, 2, bias=True, device=dev)

        # Ensure buffers exist
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        W = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W, int(W.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b, int(b.nbytes))
        _cuda_sync()

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev)

        y = lin.forward(x)
        y_np = _cuda_readback(y)

        ref = x_np @ W.T + b
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y_np, ref, rtol=1e-5, atol=1e-5)

    def test_linear_cuda_forward_attaches_context_when_requires_grad(self):
        """
        Validate ctx wiring and backward_fn output shapes/devices on CUDA.

        This test assumes your CUDA Linear.forward attaches Context with parents:
          (x, weight, bias)
        and that backward returns (grad_x, grad_w, grad_b) as CUDA tensors.
        """
        dev = _make_cuda_device(0)
        lin = Linear(3, 4, bias=True, device=dev)

        # Ensure params allocated
        lin.weight.requires_grad = True
        lin.bias.requires_grad = True
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        # Use deterministic small params (zeros are fine here)
        W0 = np.zeros((4, 3), dtype=np.float32)
        b0 = np.zeros((4,), dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W0, int(W0.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b0, int(b0.nbytes))
        _cuda_sync()

        x_np = np.ones((2, 3), dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=True)

        out = lin.forward(x)

        self.assertTrue(out.requires_grad)
        ctx = out._get_ctx()
        self.assertIsNotNone(
            ctx, "Expected Context to be attached to CUDA output Tensor."
        )
        self.assertTrue(callable(ctx.backward_fn))

        # parents order
        self.assertEqual(len(ctx.parents), 3)
        self.assertIs(ctx.parents[0], x)
        self.assertIs(ctx.parents[1], lin.weight)
        self.assertIs(ctx.parents[2], lin.bias)

        # saved_tensors: (x, weight)
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], x)
        self.assertIs(ctx.saved_tensors[1], lin.weight)

        # backward_fn shape contracts
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

    def test_linear_cuda_backward_matches_numpy_reference(self):
        """
        End-to-end autograd on CUDA:
          out = lin(x)
          out.backward(ones)
        Check x.grad, weight.grad, bias.grad numerically by readback.

        This assumes your CUDA Tensor.backward works for this graph AND that no node
        receives multiple gradient contributions (your CUDA backward limitation).
        """
        dev = _make_cuda_device(0)
        lin = Linear(3, 2, bias=True, device=dev)

        # Allocate params
        lin.weight.requires_grad = True
        lin.bias.requires_grad = True
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        lin.bias._ensure_cuda_alloc(dtype=np.float32)

        # Deterministic params
        W = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        lib, m = _cuda_lib_and_ctypes()
        _cuda_set_device(dev)
        m.cudaMemcpyHtoD(lib, int(lin.weight.data), W, int(W.nbytes))
        m.cudaMemcpyHtoD(lib, int(lin.bias.data), b, int(b.nbytes))
        _cuda_sync()

        # Input
        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _cuda_tensor_from_numpy(x_np, dev, requires_grad=True)

        # Forward
        out = lin.forward(x)

        # Backward with grad_out = ones
        go_np = np.ones(out.shape, dtype=np.float32)
        go = _cuda_tensor_from_numpy(go_np, dev, requires_grad=False)
        out.backward(go)

        # Read back grads
        gx = _cuda_readback(x.grad)
        gW = _cuda_readback(lin.weight.grad)
        gb = _cuda_readback(lin.bias.grad)

        # NumPy reference grads:
        # out = x @ W^T + b
        # L = sum(out) (since grad_out is ones)
        # dL/dx = ones @ W
        # dL/dW = ones^T @ x  (shape (out, in))
        # dL/db = sum(ones, axis=0) = batch
        ref_gx = go_np @ W
        ref_gW = go_np.T @ x_np
        ref_gb = go_np.sum(axis=0)

        np.testing.assert_allclose(gx, ref_gx, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gW, ref_gW, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gb, ref_gb, rtol=1e-4, atol=1e-4)

    def test_linear_cuda_forward_no_context_when_no_requires_grad(self):
        dev = _make_cuda_device(0)
        lin = Linear(3, 4, bias=True, device=dev)

        lin.weight.requires_grad = False
        if lin.bias is not None:
            lin.bias.requires_grad = False

        # Allocate params anyway to avoid failing forward due to data==0 checks
        lin.weight._ensure_cuda_alloc(dtype=np.float32)
        if lin.bias is not None:
            lin.bias._ensure_cuda_alloc(dtype=np.float32)

        x = _cuda_tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), dev, requires_grad=False
        )

        out = lin.forward(x)

        self.assertFalse(out.requires_grad)
        self.assertIsNone(
            out._get_ctx(), "Expected no Context when nothing requires grad."
        )


if __name__ == "__main__":
    unittest.main()
