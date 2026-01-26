import unittest
from typing import Callable, Optional

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.utils._preprocessing import one_hot, numpy_to_tensor
from src.keydnn.infrastructure.tensor._tensor import Tensor


def _try_get_tensor_to_numpy() -> Optional[Callable[[Tensor], np.ndarray]]:
    """
    Return a callable that converts a Tensor to a NumPy array using public APIs,
    or None if such a conversion is not available.

    This keeps tests resilient as the Tensor API evolves.
    """

    if hasattr(Tensor, "to_numpy") and callable(getattr(Tensor, "to_numpy")):
        return lambda t: t.to_numpy()
    if hasattr(Tensor, "numpy") and callable(getattr(Tensor, "numpy")):
        return lambda t: t.numpy()
    if hasattr(Tensor, "as_numpy") and callable(getattr(Tensor, "as_numpy")):
        return lambda t: t.as_numpy()

    if hasattr(Tensor, "copy_to_numpy") and callable(getattr(Tensor, "copy_to_numpy")):

        def _copy_to_numpy(t: Tensor) -> np.ndarray:
            out = np.empty(tuple(getattr(t, "shape", ())), dtype=np.float32)
            t.copy_to_numpy(out)
            return out

        return _copy_to_numpy

    return None


_TENSOR_TO_NUMPY = _try_get_tensor_to_numpy()


class TestPreprocessingOneHot(unittest.TestCase):
    def test_one_hot_basic(self) -> None:
        labels = np.array([0, 2, 1, 2], dtype=np.int64)
        y = one_hot(labels, num_classes=3)

        self.assertEqual(y.shape, (4, 3))
        self.assertEqual(y.dtype, np.float32)

        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(y, expected, rtol=0.0, atol=0.0)

    def test_one_hot_flattens_input(self) -> None:
        labels = np.array([[1], [0], [2]], dtype=np.int64)
        y = one_hot(labels, num_classes=3)

        self.assertEqual(y.shape, (3, 3))
        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(y, expected, rtol=0.0, atol=0.0)

    def test_one_hot_raises_on_out_of_range(self) -> None:
        labels = np.array([0, 3], dtype=np.int64)
        with self.assertRaises(IndexError):
            _ = one_hot(labels, num_classes=3)


class TestPreprocessingNumpyToTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpu = Device("cpu")

    def test_numpy_to_tensor_shape_and_dtype(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        t = numpy_to_tensor(arr, device=self.cpu, requires_grad=False)

        t_shape = getattr(t, "shape", None)
        self.assertIsNotNone(t_shape)
        self.assertEqual(tuple(t_shape), arr.shape)

        if hasattr(t, "requires_grad"):
            self.assertFalse(bool(getattr(t, "requires_grad")))

        if _TENSOR_TO_NUMPY is None:
            self.skipTest(
                "Tensor has no public NumPy export (to_numpy/numpy/as_numpy/copy_to_numpy)."
            )
        out = _TENSOR_TO_NUMPY(t)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, arr.astype(np.float32), rtol=0.0, atol=0.0)

    def test_numpy_to_tensor_requires_grad_flag(self) -> None:
        arr = np.random.randn(3, 4).astype(np.float32)
        t = numpy_to_tensor(arr, device=self.cpu, requires_grad=True)

        if hasattr(t, "requires_grad"):
            self.assertTrue(bool(getattr(t, "requires_grad")))
        else:
            self.assertIsInstance(t, Tensor)

    def test_numpy_to_tensor_copies_data(self) -> None:
        if _TENSOR_TO_NUMPY is None:
            self.skipTest(
                "Tensor has no public NumPy export (to_numpy/numpy/as_numpy/copy_to_numpy)."
            )

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = numpy_to_tensor(arr, device=self.cpu, requires_grad=False)

        arr[:] = 999.0

        out = _TENSOR_TO_NUMPY(t)
        np.testing.assert_allclose(
            out,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
