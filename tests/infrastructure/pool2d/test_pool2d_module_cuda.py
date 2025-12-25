import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.pooling._pooling_module import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool2d,
)

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
    avgpool2d_forward_cpu,
    avgpool2d_backward_cpu,
    global_avgpool2d_forward_cpu,
    global_avgpool2d_backward_cpu,
)


def _maybe_free_cuda_tensor(t: Tensor, lib, cuda_free_fn) -> None:
    """
    Best-effort cleanup for CUDA tensors that hold DevPtr handles.
    Safe to call even if tensor is CPU or already freed (dev_ptr==0).
    """
    if t is None:
        return
    d = getattr(t, "device", None)
    if d is None:
        return
    is_cuda = getattr(d, "is_cuda", None)
    if not (callable(is_cuda) and is_cuda()):
        return

    try:
        dev_ptr = int(t.data)
    except Exception:
        return

    if dev_ptr != 0:
        try:
            cuda_free_fn(lib, dev_ptr)
        except Exception:
            # Don't fail tests due to cleanup issues.
            pass


class _CudaTestBase(unittest.TestCase):
    """
    Base class that loads the CUDA DLL once and skips tests if unavailable.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = Device("cuda:0")
        cls._cuda_available = False
        cls._skip_reason = ""

        try:
            # We use maxpool2d_ctypes as the "CUDA utils" source (malloc/free/memcpy).
            from src.keydnn.infrastructure.native_cuda.python import (
                maxpool2d_ctypes as m,
            )

            cls._m = m
            cls.lib = m.load_keydnn_cuda_native()
            cls._cuda_available = True
        except Exception as e:
            cls._cuda_available = False
            cls._skip_reason = f"CUDA native DLL unavailable or failed to load: {e!r}"

    def setUp(self) -> None:
        if not self.__class__._cuda_available:
            self.skipTest(self.__class__._skip_reason)
        np.random.seed(0)

    def tensor_from_numpy_cuda(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        """
        Allocate device memory, upload arr (HtoD), and construct a CUDA Tensor
        backed by the resulting DevPtr. Uses legacy memcpy alias signature.
        """
        m = self.__class__._m
        lib = self.__class__.lib

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        # Allocate and copy host -> device
        dev_ptr = m.cuda_malloc(lib, int(arr.nbytes))

        try:
            # Legacy signature: (lib, dst_dev, src_host, nbytes)
            # Your wrappers should accept this for back-compat.
            m.cudaMemcpyHtoD(lib, int(dev_ptr), arr, int(arr.nbytes))

            # Construct Tensor from devptr (expects your updated Tensor supports dtype)
            t = Tensor._from_devptr(
                dev_ptr=int(dev_ptr),
                shape=arr.shape,
                device=self.device,
                requires_grad=requires_grad,
                ctx=None,
                dtype=arr.dtype,
            )
            return t

        except Exception:
            # Prevent leak if construction/upload fails
            try:
                m.cuda_free(lib, int(dev_ptr))
            except Exception:
                pass
            raise


class TestPool2dModuleCuda(_CudaTestBase):
    def test_maxpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(1, 2, 5, 6).astype(np.float32)
        x = self.tensor_from_numpy_cuda(x_np, requires_grad=True)

        m = self.__class__._m
        lib = self.__class__.lib

        try:
            pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
            y = pool.forward(x)

            # loss = sum(y) => grad_out = ones
            loss = y.sum()
            loss.backward()

            self.assertIsNotNone(x.grad)

            grad_out_np = np.ones(y.shape, dtype=np.float32)
            y_ref, argmax_idx = maxpool2d_forward_cpu(
                x_np, kernel_size=2, stride=2, padding=0
            )
            self.assertEqual(y_ref.shape, y.shape)

            grad_x_ref = maxpool2d_backward_cpu(
                grad_out_np,
                argmax_idx,
                x_shape=x_np.shape,
                kernel_size=2,
                stride=2,
                padding=0,
            )

            self.assertTrue(
                np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-5, rtol=1e-5)
            )

        finally:
            _maybe_free_cuda_tensor(x, lib, m.cuda_free)
            _maybe_free_cuda_tensor(getattr(x, "grad", None), lib, m.cuda_free)

    def test_avgpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(2, 3, 6, 5).astype(np.float32)
        x = self.tensor_from_numpy_cuda(x_np, requires_grad=True)

        m = self.__class__._m
        lib = self.__class__.lib

        try:
            pool = AvgPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(1, 0))
            y = pool.forward(x)

            loss = y.sum()
            loss.backward()

            self.assertIsNotNone(x.grad)

            grad_out_np = np.ones(y.shape, dtype=np.float32)
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

            self.assertTrue(
                np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-5, rtol=1e-5)
            )

        finally:
            _maybe_free_cuda_tensor(x, lib, m.cuda_free)
            _maybe_free_cuda_tensor(getattr(x, "grad", None), lib, m.cuda_free)

    def test_global_avgpool2d_module_backward_matches_cpu(self):
        x_np = np.random.randn(2, 4, 3, 5).astype(np.float32)
        x = self.tensor_from_numpy_cuda(x_np, requires_grad=True)

        m = self.__class__._m
        lib = self.__class__.lib

        try:
            pool = GlobalAvgPool2d()
            y = pool.forward(x)

            loss = y.sum()
            loss.backward()

            self.assertIsNotNone(x.grad)

            grad_out_np = np.ones(y.shape, dtype=np.float32)
            y_ref = global_avgpool2d_forward_cpu(x_np)
            self.assertEqual(y_ref.shape, y.shape)

            grad_x_ref = global_avgpool2d_backward_cpu(grad_out_np, x_shape=x_np.shape)

            self.assertTrue(
                np.allclose(x.grad.to_numpy(), grad_x_ref, atol=1e-5, rtol=1e-5)
            )

        finally:
            _maybe_free_cuda_tensor(x, lib, m.cuda_free)
            _maybe_free_cuda_tensor(getattr(x, "grad", None), lib, m.cuda_free)


if __name__ == "__main__":
    unittest.main()
