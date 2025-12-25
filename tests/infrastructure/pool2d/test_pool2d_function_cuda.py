import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.pooling._pooling_function import (
    MaxPool2dFn,
    AvgPool2dFn,
    GlobalAvgPool2dFn,
)

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)

# CUDA helpers come from your CUDA ops module
from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)

# NOTE: host<->device memcpy wrappers are expected to exist in your native bindings.
# If your project names differ, adjust `_cuda_memcpy_htod/_cuda_memcpy_dtoh` below.
from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as cuda_ctypes


def tensor_from_numpy_cpu(
    arr: np.ndarray, device: Device, requires_grad: bool
) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _cuda_memcpy_htod(lib, dst_dev: int, src_host: np.ndarray, nbytes: int) -> None:
    """
    Copy host -> device.

    IMPORTANT: Update this if your ctypes function names differ.
    """
    fn = getattr(cuda_ctypes, "cuda_memcpy_htod", None) or getattr(
        cuda_ctypes, "cudaMemcpyHtoD", None
    )
    if fn is None:
        raise RuntimeError(
            "Missing CUDA host->device memcpy binding. Expected `cuda_memcpy_htod` "
            "or `cudaMemcpyHtoD` in native_cuda.python.maxpool2d_ctypes."
        )
    fn(lib, int(dst_dev), src_host, int(nbytes))


def _cuda_memcpy_dtoh(lib, dst_host: np.ndarray, src_dev: int, nbytes: int) -> None:
    """
    Copy device -> host.

    IMPORTANT: Update this if your ctypes function names differ.
    """
    fn = getattr(cuda_ctypes, "cuda_memcpy_dtoh", None) or getattr(
        cuda_ctypes, "cudaMemcpyDtoH", None
    )
    if fn is None:
        raise RuntimeError(
            "Missing CUDA device->host memcpy binding. Expected `cuda_memcpy_dtoh` "
            "or `cudaMemcpyDtoH` in native_cuda.python.maxpool2d_ctypes."
        )
    fn(lib, dst_host, int(src_dev), int(nbytes))


class _CudaAllocs:
    """
    Tiny RAII-style allocator tracker for tests.

    Notes
    -----
    Your Tensor._from_devptr does not auto-free. These tests therefore explicitly
    free every DevPtr we allocate or receive from CUDA kernels (including argmax_idx).
    """

    def __init__(self, *, device_index: int = 0) -> None:
        self.lib = _load_cuda_lib()
        self.device_index = int(device_index)
        cuda_set_device(self.lib, self.device_index)
        self._ptrs: list[int] = []

    def malloc(self, nbytes: int) -> int:
        p = int(cuda_malloc(self.lib, int(nbytes)))
        self._ptrs.append(p)
        return p

    def track(self, p: int) -> int:
        """Track an externally-allocated DevPtr so it will be freed in cleanup."""
        self._ptrs.append(int(p))
        return int(p)

    def free_all(self) -> None:
        # free in reverse order
        for p in reversed(self._ptrs):
            try:
                cuda_free(self.lib, int(p))
            except Exception:
                # best-effort cleanup during tests
                pass
        self._ptrs.clear()


def tensor_from_numpy_cuda(
    arr: np.ndarray,
    device: Device,
    requires_grad: bool,
    allocs: _CudaAllocs,
) -> Tensor:
    """
    Create a CUDA Tensor by:
    - allocating device memory,
    - memcpy host->device,
    - wrapping via Tensor._from_devptr.
    """
    if not device.is_cuda():
        raise ValueError("tensor_from_numpy_cuda expects a CUDA Device")

    arr = np.asarray(arr)
    if arr.dtype not in (np.float32, np.float64):
        raise TypeError(f"CUDA pooling tests expect float32/float64; got {arr.dtype}")

    nbytes = int(arr.size * arr.dtype.itemsize)
    dev_ptr = allocs.malloc(nbytes)
    _cuda_memcpy_htod(allocs.lib, dev_ptr, arr, nbytes)

    return Tensor._from_devptr(
        int(dev_ptr),
        shape=arr.shape,
        device=device,
        requires_grad=requires_grad,
        ctx=None,
    )


def cuda_tensor_to_numpy(t: Tensor, allocs: _CudaAllocs) -> np.ndarray:
    """
    Copy CUDA Tensor back to host NumPy.

    NOTE: This assumes `t.dtype` is present (your CUDA wrappers rely on it).
    """
    if not t.device.is_cuda():
        raise ValueError("cuda_tensor_to_numpy expects a CUDA tensor")

    dt = t.dtype
    out = np.empty(t.shape, dtype=dt)
    nbytes = int(out.size * out.dtype.itemsize)
    _cuda_memcpy_dtoh(allocs.lib, out, int(t.data), nbytes)
    return out


class TestPool2dFunctionCuda(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        device_index = 0
        self.device = Device(f"cuda:{device_index}")
        self.allocs = _CudaAllocs(device_index=device_index)

    def tearDown(self) -> None:
        self.allocs.free_all()

    def test_maxpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(1, 2, 5, 6).astype(np.float32)
        x = tensor_from_numpy_cuda(
            x_np, self.device, requires_grad=True, allocs=self.allocs
        )

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = MaxPool2dFn.forward(ctx, x, kernel_size=2, stride=2, padding=0)

        # Track y buffer for cleanup (forward allocates y_dev inside CUDA ext)
        self.allocs.track(int(y.data))

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy_cuda(
            grad_out_np, self.device, requires_grad=False, allocs=self.allocs
        )

        (grad_x,) = MaxPool2dFn.backward(ctx, grad_out)

        self.assertIsNotNone(grad_x)
        self.allocs.track(int(grad_x.data))

        # Track argmax pointer (CUDA ext allocates it and returns DevPtr int)
        self.allocs.track(int(ctx.saved_meta["argmax_idx"]))

        # CPU reference
        y_ref, argmax_idx_cpu = maxpool2d_forward_cpu(
            x_np, kernel_size=2, stride=2, padding=0
        )
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = maxpool2d_backward_cpu(
            grad_out_np,
            argmax_idx_cpu,
            x_shape=x_np.shape,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        grad_x_host = cuda_tensor_to_numpy(grad_x, self.allocs)
        self.assertTrue(np.allclose(grad_x_host, grad_x_ref, atol=1e-5, rtol=1e-5))

    def test_avgpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(2, 3, 6, 5).astype(np.float32)
        x = tensor_from_numpy_cuda(
            x_np, self.device, requires_grad=True, allocs=self.allocs
        )

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = AvgPool2dFn.forward(
            ctx, x, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0)
        )
        self.allocs.track(int(y.data))

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy_cuda(
            grad_out_np, self.device, requires_grad=False, allocs=self.allocs
        )

        (grad_x,) = AvgPool2dFn.backward(ctx, grad_out)
        self.assertIsNotNone(grad_x)
        self.allocs.track(int(grad_x.data))

        # CPU reference
        y_ref = avgpool2d_forward_cpu(
            x_np, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0)
        )
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = avgpool2d_backward_cpu(
            grad_out_np,
            x_shape=x_np.shape,
            kernel_size=(2, 3),
            stride=(2, 1),
            padding=(1, 0),
        )

        grad_x_host = cuda_tensor_to_numpy(grad_x, self.allocs)
        self.assertTrue(np.allclose(grad_x_host, grad_x_ref, atol=1e-5, rtol=1e-5))

    def test_global_avgpool2d_fn_backward_matches_cpu(self):
        x_np = np.random.randn(2, 4, 3, 5).astype(np.float32)
        x = tensor_from_numpy_cuda(
            x_np, self.device, requires_grad=True, allocs=self.allocs
        )

        ctx = Context(parents=(x,), backward_fn=lambda go: ())
        y = GlobalAvgPool2dFn.forward(ctx, x)
        self.allocs.track(int(y.data))

        grad_out_np = np.random.randn(*y.shape).astype(np.float32)
        grad_out = tensor_from_numpy_cuda(
            grad_out_np, self.device, requires_grad=False, allocs=self.allocs
        )

        (grad_x,) = GlobalAvgPool2dFn.backward(ctx, grad_out)
        self.assertIsNotNone(grad_x)
        self.allocs.track(int(grad_x.data))

        # CPU reference
        y_ref = global_avgpool2d_forward_cpu(x_np)
        self.assertEqual(y_ref.shape, y.shape)

        grad_x_ref = global_avgpool2d_backward_cpu(grad_out_np, x_shape=x_np.shape)

        grad_x_host = cuda_tensor_to_numpy(grad_x, self.allocs)
        self.assertTrue(np.allclose(grad_x_host, grad_x_ref, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
