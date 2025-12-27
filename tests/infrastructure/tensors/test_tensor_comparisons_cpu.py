import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class TestTensorComparisonsCPU(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def _make_cpu_tensor(
        self, x_np: np.ndarray, *, requires_grad: bool = False
    ) -> Tensor:
        t = Tensor(shape=x_np.shape, device=self.device, requires_grad=requires_grad)
        t.copy_from_numpy(np.ascontiguousarray(x_np))
        return t

    def _assert_mask_tensor(self, y: Tensor, shape: tuple[int, ...]) -> None:
        self.assertEqual(tuple(y.shape), tuple(shape))
        self.assertEqual(np.dtype(y.dtype), np.float32)
        self.assertFalse(
            bool(getattr(y, "requires_grad", False)),
            "comparison outputs must not require grad",
        )

        y_np = y.to_numpy()
        self.assertEqual(y_np.dtype, np.float32)
        # Ensure all values are exactly 0.0 or 1.0
        self.assertTrue(np.all((y_np == 0.0) | (y_np == 1.0)))

    def test_gt_tensor_tensor_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 5)).astype(np.float32)
        b = rng.standard_normal((4, 5)).astype(np.float32)

        a_t = self._make_cpu_tensor(a)
        b_t = self._make_cpu_tensor(b)

        y_t = a_t > b_t
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a > b).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_ge_tensor_tensor_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((6,)).astype(np.float32)
        b = rng.standard_normal((6,)).astype(np.float32)

        a_t = self._make_cpu_tensor(a)
        b_t = self._make_cpu_tensor(b)

        y_t = a_t >= b_t
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a >= b).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_lt_tensor_tensor_matches_numpy(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((3, 2, 4)).astype(np.float32)
        b = rng.standard_normal((3, 2, 4)).astype(np.float32)

        a_t = self._make_cpu_tensor(a)
        b_t = self._make_cpu_tensor(b)

        y_t = a_t < b_t
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a < b).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_le_tensor_tensor_matches_numpy(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((8, 1)).astype(np.float32)
        b = rng.standard_normal((8, 1)).astype(np.float32)

        a_t = self._make_cpu_tensor(a)
        b_t = self._make_cpu_tensor(b)

        y_t = a_t <= b_t
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a <= b).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_gt_tensor_scalar_matches_numpy(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.standard_normal((5, 5)).astype(np.float32)
        s = 0.25  # python float

        a_t = self._make_cpu_tensor(a)
        y_t = a_t > s
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a > np.float32(s)).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_ge_tensor_scalar_matches_numpy(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.standard_normal((10,)).astype(np.float32)
        s = -0.1

        a_t = self._make_cpu_tensor(a)
        y_t = a_t >= s
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a >= np.float32(s)).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_lt_tensor_scalar_matches_numpy(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.standard_normal((2, 7)).astype(np.float32)
        s = 1.5

        a_t = self._make_cpu_tensor(a)
        y_t = a_t < s
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a < np.float32(s)).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_le_tensor_scalar_matches_numpy(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((9, 3)).astype(np.float32)
        s = 0.0

        a_t = self._make_cpu_tensor(a)
        y_t = a_t <= s
        self._assert_mask_tensor(y_t, a.shape)

        ref = (a <= np.float32(s)).astype(np.float32)
        np.testing.assert_array_equal(y_t.to_numpy(), ref)

    def test_comparison_raises_on_shape_mismatch(self) -> None:
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 4), dtype=np.float32)
        a_t = self._make_cpu_tensor(a)
        b_t = self._make_cpu_tensor(b)

        # Your Tensor CPU path uses _binary_op_shape_check, so ValueError is expected.
        with self.assertRaises(ValueError):
            _ = a_t >= b_t

    def test_comparison_outputs_never_require_grad(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.standard_normal((4, 4)).astype(np.float32)
        b = rng.standard_normal((4, 4)).astype(np.float32)

        a_t = self._make_cpu_tensor(a, requires_grad=True)
        b_t = self._make_cpu_tensor(b, requires_grad=True)

        y1 = a_t > b_t
        y2 = a_t >= b_t
        y3 = a_t < b_t
        y4 = a_t <= b_t

        for y in (y1, y2, y3, y4):
            self.assertFalse(bool(getattr(y, "requires_grad", False)))


if __name__ == "__main__":
    unittest.main()
