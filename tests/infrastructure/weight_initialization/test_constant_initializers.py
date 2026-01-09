import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.utils.weight_initializer._base import WeightInitializer


class DummyTensor:
    """
    Minimal tensor stub for testing constant initializers.

    Required surface area for zeros/ones initializers:
      - shape, dtype, device attributes
      - copy_from(TensorLike)
      - Tensor.zeros / Tensor.ones classmethods (provided below)
    """

    def __init__(self, *, shape, device=Device("cpu"), dtype=np.float32):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = np.dtype(dtype)
        self.data = np.full(
            self.shape, 7, dtype=self.dtype
        )  # non-trivial initial content

    def copy_from(self, other: "DummyTensor") -> None:
        # Mimic in-place copy semantics
        assert other.shape == self.shape
        self.data[...] = other.data

    @classmethod
    def zeros(cls, *, shape, device, dtype):
        t = cls(shape=shape, device=device, dtype=dtype)
        t.data[...] = np.zeros(shape, dtype=np.dtype(dtype))
        return t

    @classmethod
    def ones(cls, *, shape, device, dtype):
        t = cls(shape=shape, device=device, dtype=dtype)
        t.data[...] = np.ones(shape, dtype=np.dtype(dtype))
        return t


class TestConstantInitializers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from src.keydnn.infrastructure.utils.weight_initializer import (
                _constant,
            )  # noqa: F401
        except Exception:
            pass

    def test_zeros_registered(self):
        self.assertIn("zeros", WeightInitializer.available())

    def test_ones_registered(self):
        self.assertIn("ones", WeightInitializer.available())

    def test_zeros_initializes_in_place_float32(self):
        init = WeightInitializer("zeros")
        t = DummyTensor(shape=(3, 4), device=Device("cpu"), dtype=np.float32)
        init(t)
        self.assertTrue(
            np.all(t.data == 0), msg="zeros initializer did not fill with 0"
        )
        self.assertEqual(t.data.dtype, np.float32)

    def test_ones_initializes_in_place_float32(self):
        init = WeightInitializer("ones")
        t = DummyTensor(shape=(3, 4), device=Device("cpu"), dtype=np.float32)
        init(t)
        self.assertTrue(np.all(t.data == 1), msg="ones initializer did not fill with 1")
        self.assertEqual(t.data.dtype, np.float32)

    def test_zeros_preserves_float64_dtype(self):
        init = WeightInitializer("zeros")
        t = DummyTensor(shape=(2, 2, 2), device=Device("cpu"), dtype=np.float64)
        init(t)
        self.assertTrue(np.all(t.data == 0))
        self.assertEqual(t.data.dtype, np.float64)

    def test_ones_preserves_float64_dtype(self):
        init = WeightInitializer("ones")
        t = DummyTensor(shape=(2, 2, 2), device=Device("cpu"), dtype=np.float64)
        init(t)
        self.assertTrue(np.all(t.data == 1))
        self.assertEqual(t.data.dtype, np.float64)

    def test_initializer_returns_same_object(self):
        z = WeightInitializer("zeros")
        t = DummyTensor(shape=(5,), device=Device("cpu"), dtype=np.float32)
        out = z(t)
        self.assertIs(out, t)


if __name__ == "__main__":
    unittest.main()
