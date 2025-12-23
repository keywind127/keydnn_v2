from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.domain.device._device import Device


class _TensorFactoryMixin:
    def _tensor_from_numpy(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        t = Tensor(arr.shape, Device("cpu"), requires_grad=requires_grad)
        t.copy_from_numpy(arr)
        return t


class TestTensorExpForward(TestCase, _TensorFactoryMixin):
    def test_exp_forward_matches_numpy(self):
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.exp()
        expected = np.exp(x_np)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_exp_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.exp()


class TestTensorExpBackward(TestCase, _TensorFactoryMixin):
    def test_exp_backward_matches_closed_form(self):
        """
        y = exp(x), loss = sum(y)
        dloss/dx = exp(x)
        """
        x_np = np.random.randn(5, 6).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.exp()
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.exp(x_np).astype(np.float32)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, rtol=1e-5, atol=1e-6))


class TestTensorSumAxisKeepdimsForward(TestCase, _TensorFactoryMixin):
    def test_sum_all_forward_scalar(self):
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=None)
        expected = np.sum(x_np)

        self.assertEqual(y.shape, ())
        self.assertAlmostEqual(
            float(np.asarray(y.to_numpy())), float(expected), places=6
        )

    def test_sum_axis0_forward_keepdims_false(self):
        x_np = np.random.randn(5, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=0, keepdims=False)
        expected = np.sum(x_np, axis=0)

        self.assertEqual(y.shape, (4,))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_axis1_forward_keepdims_true(self):
        x_np = np.random.randn(5, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=1, keepdims=True)
        expected = np.sum(x_np, axis=1, keepdims=True)

        self.assertEqual(y.shape, (5, 1))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_negative_axis_forward_keepdims_true(self):
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=-1, keepdims=True)
        expected = np.sum(x_np, axis=-1, keepdims=True)

        self.assertEqual(y.shape, (2, 3, 1))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_axis_out_of_bounds_raises(self):
        x = self._tensor_from_numpy(
            np.random.randn(2, 3).astype(np.float32), requires_grad=False
        )
        with self.assertRaises(ValueError):
            _ = x.sum(axis=2)

    def test_sum_axis_type_error(self):
        x = self._tensor_from_numpy(
            np.random.randn(2, 3).astype(np.float32), requires_grad=False
        )
        with self.assertRaises(TypeError):
            _ = x.sum(axis="0")  # type: ignore[arg-type]

    def test_sum_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.sum(axis=0)


class TestTensorSumAxisKeepdimsBackward(TestCase, _TensorFactoryMixin):
    def test_sum_axis0_backward_broadcasts_grad_keepdims_false(self):
        """
        x: (B,C), y = sum(x, axis=0) -> (C,)
        loss = sum(y) => grad_x = ones(B,C)
        """
        x = self._tensor_from_numpy(
            np.random.randn(5, 4).astype(np.float32), requires_grad=True
        )
        y = x.sum(axis=0, keepdims=False)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.ones((5, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_sum_axis1_backward_broadcasts_grad_keepdims_true(self):
        """
        x: (B,C), y = sum(x, axis=1, keepdims=True) -> (B,1)
        loss = sum(y) => grad_x = ones(B,C)
        """
        x = self._tensor_from_numpy(
            np.random.randn(5, 4).astype(np.float32), requires_grad=True
        )
        y = x.sum(axis=1, keepdims=True)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.ones((5, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_sum_all_backward_is_ones(self):
        x = self._tensor_from_numpy(
            np.random.randn(3, 4).astype(np.float32), requires_grad=True
        )
        loss = x.sum(axis=None)
        loss.backward()

        expected = np.ones((3, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))


class TestTensorBroadcastToForward(TestCase, _TensorFactoryMixin):
    def test_broadcast_to_forward_matches_numpy(self):
        x_np = np.random.randn(3, 1).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.broadcast_to((3, 4))
        expected = np.broadcast_to(x_np, (3, 4))

        self.assertEqual(y.shape, (3, 4))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_broadcast_to_forward_adds_leading_dims(self):
        # (3,) -> (2,3) by leading broadcast
        x_np = np.random.randn(3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.broadcast_to((2, 3))
        expected = np.broadcast_to(x_np, (2, 3))

        self.assertEqual(y.shape, (2, 3))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_broadcast_to_same_shape_is_ok(self):
        x_np = np.random.randn(2, 3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.broadcast_to((2, 3))
        self.assertEqual(y.shape, (2, 3))
        self.assertTrue(np.allclose(y.to_numpy(), x_np, rtol=1e-5, atol=1e-6))

    def test_broadcast_to_invalid_raises(self):
        x_np = np.random.randn(2, 3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        with self.assertRaises(ValueError):
            _ = x.broadcast_to((3, 3))  # cannot change 2 -> 3 on dim0

    def test_broadcast_to_on_cuda_raises(self):
        x = Tensor((2, 1), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.broadcast_to((2, 5))


class TestTensorBroadcastToBackward(TestCase, _TensorFactoryMixin):
    def test_broadcast_to_backward_sums_over_broadcasted_axis(self):
        """
        x: (3,1) -> y: (3,4)
        loss = sum(y) => grad_y = ones(3,4)
        grad_x should be sum over axis=1 => ones(3,1)*4
        """
        x_np = np.random.randn(3, 1).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.broadcast_to((3, 4))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.ones((3, 1), dtype=np.float32) * 4.0
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_broadcast_to_backward_sums_over_leading_broadcast_dim(self):
        """
        x: (3,) -> y: (2,3)
        loss = sum(y) => grad_y = ones(2,3)
        grad_x should be sum over axis=0 => ones(3)*2
        """
        x_np = np.random.randn(3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.broadcast_to((2, 3))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.ones((3,), dtype=np.float32) * 2.0
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_broadcast_to_backward_multiple_reduction_axes(self):
        """
        x: (1,3,1) -> y: (2,3,4)
        loss = sum(y) => grad_y = ones(2,3,4)
        grad_x sums over axis 0 and 2 => 2*4 = 8
        """
        x_np = np.random.randn(1, 3, 1).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.broadcast_to((2, 3, 4))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        expected = np.ones((1, 3, 1), dtype=np.float32) * 8.0
        self.assertTrue(np.allclose(x.grad.to_numpy(), expected, rtol=1e-5, atol=1e-6))


class TestTensorMaxForwardBackward(TestCase, _TensorFactoryMixin):
    def test_max_forward_matches_numpy_keepdims_false(self):
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.max(axis=1, keepdims=False)
        expected = np.max(x_np, axis=1)

        self.assertEqual(y.shape, (4,))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_max_forward_matches_numpy_keepdims_true(self):
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.max(axis=1, keepdims=True)
        expected = np.max(x_np, axis=1, keepdims=True)

        self.assertEqual(y.shape, (4, 1))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_max_backward_routes_grad_to_argmax_positions(self):
        """
        x:
          [[1, 3, 2],
           [5, 4, 5]]  (tie on row1)
        y = max(x, axis=1), loss = sum(y)
        grad should be 1 at max positions; ties receive 1 on each max position (mask-based).
        """
        x_np = np.array([[1, 3, 2], [5, 4, 5]], dtype=np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=True)

        y = x.max(axis=1, keepdims=False)  # (2,)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        grad = x.grad.to_numpy()

        expected = np.zeros_like(x_np)
        expected[0, 1] = 1.0  # 3 is max in row0
        expected[1, 0] = 1.0  # tie max 5
        expected[1, 2] = 1.0  # tie max 5

        self.assertTrue(np.array_equal(grad, expected))

    def test_max_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.max(axis=1, keepdims=False)


if __name__ == "__main__":
    unittest.main()
