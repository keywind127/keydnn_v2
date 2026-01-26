from __future__ import annotations

import unittest
import numpy as np


def _import_infra_tensor():
    """
    Import the *infrastructure* Tensor class (the one that defines __len__ and nbytes).
    """
    try:
        from src.keydnn.infrastructure.tensor import Tensor

        return Tensor
    except Exception:
        pass

    try:
        from src.keydnn.infrastructure.tensor._tensor import Tensor

        return Tensor
    except Exception:
        pass

    from src.keydnn.infrastructure.tensor._tensor import Tensor

    return Tensor


class TestTensorLenAndNbytes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.Tensor = _import_infra_tensor()

        from src.keydnn.presentation.apis.tensors import Device

        cls.Device = Device

        try:
            cls.cpu = Device("cpu")
        except Exception:
            cls.cpu = Device.cpu()

    def test_len_scalar_raises_value_error(self) -> None:
        t = self.Tensor((), self.cpu, requires_grad=False, dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = len(t)

    def test_len_1d(self) -> None:
        t = self.Tensor((7,), self.cpu, requires_grad=False, dtype=np.float32)
        self.assertEqual(len(t), 7)

    def test_len_nd(self) -> None:
        t = self.Tensor((3, 4, 5), self.cpu, requires_grad=False, dtype=np.float32)
        self.assertEqual(len(t), 3)

    def test_len_zero_first_dim(self) -> None:
        t = self.Tensor((0, 10), self.cpu, requires_grad=False, dtype=np.float32)
        self.assertEqual(len(t), 0)

    def test_nbytes_scalar(self) -> None:
        t = self.Tensor((), self.cpu, requires_grad=False, dtype=np.float32)
        self.assertEqual(t.nbytes, np.dtype(np.float32).itemsize)

    def test_nbytes_small_f32(self) -> None:
        shape = (2, 3, 4)
        t = self.Tensor(shape, self.cpu, requires_grad=False, dtype=np.float32)
        expected = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        self.assertEqual(t.nbytes, expected)

    def test_nbytes_small_f64(self) -> None:
        shape = (2, 3, 4)
        t = self.Tensor(shape, self.cpu, requires_grad=False, dtype=np.float64)
        expected = int(np.prod(shape)) * np.dtype(np.float64).itemsize
        self.assertEqual(t.nbytes, expected)

    def test_nbytes_empty_tensor_is_zero(self) -> None:
        t = self.Tensor((0, 3, 4), self.cpu, requires_grad=False, dtype=np.float32)
        self.assertEqual(t.nbytes, 0)

    def test_nbytes_metadata_only_does_not_require_storage(self) -> None:
        """
        nbytes must be correct even if backing storage is absent/uninitialized.
        """
        shape = (2, 2, 2)
        t = self.Tensor(shape, self.cpu, requires_grad=False, dtype=np.float32)
        expected = int(np.prod(shape)) * np.dtype(np.float32).itemsize

        for attr in ("_storage", "storage", "_data", "data"):
            if hasattr(t, attr):
                try:
                    setattr(t, attr, None)
                except Exception:
                    pass

        self.assertEqual(t.nbytes, expected)

    def test_nbytes_matches_numel_times_itemsize_for_various_shapes(self) -> None:
        cases = [
            ((1,), np.float32),
            ((5,), np.float64),
            ((2, 0, 3), np.float32),
            ((3, 4), np.float32),
            ((2, 3, 4, 5), np.float64),
        ]
        for shape, dt in cases:
            with self.subTest(shape=shape, dtype=str(np.dtype(dt))):
                t = self.Tensor(shape, self.cpu, requires_grad=False, dtype=dt)
                numel = int(np.prod(shape)) if len(shape) > 0 else 1
                expected = 0 if numel == 0 else numel * np.dtype(dt).itemsize
                self.assertEqual(t.nbytes, expected)


if __name__ == "__main__":
    unittest.main()
