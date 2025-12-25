import unittest
import numpy as np

from src.keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes import (
    load_keydnn_cuda_native,
    cuda_set_device,
)

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class TestTensorZerosOnesCuda(unittest.TestCase):
    """
    CUDA tests for Tensor.zeros() / Tensor.ones().

    Validates:
    - Correct values on GPU (round-trip to NumPy)
    - Shape is preserved
    - Dtype is float32 (matches current Tensor.zeros/ones contract)
    - numel=0 edge case (no crash, empty output)
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Skip suite if CUDA DLL/device isn't available.
        try:
            cls.lib = load_keydnn_cuda_native()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL unavailable: {e}")

        try:
            cuda_set_device(cls.lib, 0)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA device unavailable / cannot set device: {e}")

        # Your Device API: Device("cuda:<index>")
        cls.cuda_device = Device("cuda:0")

    def _assert_tensor_matches_constant(
        self, t: Tensor, shape: tuple[int, ...], value: float
    ) -> None:
        self.assertEqual(tuple(t.shape), tuple(shape))

        # Readback to CPU for verification.
        if hasattr(t, "to_numpy"):
            arr = t.to_numpy()
        elif hasattr(t, "numpy"):
            arr = t.numpy()
        elif hasattr(t, "as_numpy"):
            arr = t.as_numpy()
        elif hasattr(t, "copy_to_numpy"):
            arr = t.copy_to_numpy()
        else:
            raise AttributeError(
                "Tensor must provide a CPU readback method (to_numpy/numpy/as_numpy/copy_to_numpy) "
                "for these CUDA correctness tests."
            )

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, shape)

        # Empty: just ensure shape/dtype correct.
        if arr.size == 0:
            return

        np.testing.assert_allclose(
            arr, np.full(shape, value, dtype=np.float32), rtol=0.0, atol=0.0
        )

    def test_zeros_cuda_small(self) -> None:
        shape = (2, 3, 4)
        t = Tensor.zeros(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 0.0)

    def test_ones_cuda_small(self) -> None:
        shape = (2, 3, 4)
        t = Tensor.ones(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 1.0)

    def test_zeros_cuda_1d(self) -> None:
        shape = (17,)
        t = Tensor.zeros(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 0.0)

    def test_ones_cuda_2d(self) -> None:
        shape = (5, 7)
        t = Tensor.ones(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 1.0)

    def test_zeros_cuda_numel_zero_is_noop(self) -> None:
        shape = (0, 3, 4)
        t = Tensor.zeros(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 0.0)

    def test_ones_cuda_numel_zero_is_noop(self) -> None:
        shape = (2, 0)
        t = Tensor.ones(shape=shape, device=self.cuda_device, requires_grad=False)
        self._assert_tensor_matches_constant(t, shape, 1.0)

    def test_requires_grad_flag_preserved_cuda(self) -> None:
        shape = (2, 2)
        t0 = Tensor.zeros(shape=shape, device=self.cuda_device, requires_grad=True)
        t1 = Tensor.ones(shape=shape, device=self.cuda_device, requires_grad=True)
        self.assertTrue(bool(getattr(t0, "requires_grad", False)))
        self.assertTrue(bool(getattr(t1, "requires_grad", False)))


if __name__ == "__main__":
    unittest.main()
