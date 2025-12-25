from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain.device._device import Device


class _TensorFactoryMixin:
    def _tensor_from_numpy(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        t = Tensor(arr.shape, Device("cpu"), requires_grad=requires_grad)
        t.copy_from_numpy(arr)
        return t


class TestTensorMatmulForward(TestCase, _TensorFactoryMixin):
    def test_matmul_forward_matches_numpy(self):
        a = self._tensor_from_numpy(np.random.randn(3, 4), requires_grad=False)
        b = self._tensor_from_numpy(np.random.randn(4, 2), requires_grad=False)

        y = a @ b
        expected = a.to_numpy() @ b.to_numpy()

        self.assertEqual(y.shape, (3, 2))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_matmul_rejects_non_2d(self):
        a = self._tensor_from_numpy(np.random.randn(2, 3, 4), requires_grad=False)
        b = self._tensor_from_numpy(np.random.randn(4, 5), requires_grad=False)

        with self.assertRaises(ValueError):
            _ = a @ b

    def test_matmul_shape_mismatch_raises(self):
        a = self._tensor_from_numpy(np.random.randn(3, 4), requires_grad=False)
        b = self._tensor_from_numpy(np.random.randn(5, 2), requires_grad=False)

        with self.assertRaises(ValueError):
            _ = a @ b

    def test_matmul_on_cuda_raises(self):
        a = Tensor((3, 4), Device("cuda:0"), requires_grad=False)
        b = Tensor((4, 2), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = a @ b


class TestTensorMatmulBackward(TestCase, _TensorFactoryMixin):
    def test_matmul_backward_grads_match_closed_form(self):
        """
        For y = A @ B, loss = sum(y),
        grad_out = ones(N, M)
        dA = grad_out @ B^T
        dB = A^T @ grad_out
        """
        A_np = np.random.randn(3, 4).astype(np.float32)
        B_np = np.random.randn(4, 2).astype(np.float32)

        A = self._tensor_from_numpy(A_np, requires_grad=True)
        B = self._tensor_from_numpy(B_np, requires_grad=True)

        y = A @ B
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(B.grad)

        grad_out = np.ones((3, 2), dtype=np.float32)
        expected_dA = grad_out @ B_np.T
        expected_dB = A_np.T @ grad_out

        self.assertEqual(A.grad.shape, A.shape)
        self.assertEqual(B.grad.shape, B.shape)
        self.assertTrue(
            np.allclose(A.grad.to_numpy(), expected_dA, rtol=1e-5, atol=1e-6)
        )
        self.assertTrue(
            np.allclose(B.grad.to_numpy(), expected_dB, rtol=1e-5, atol=1e-6)
        )

    def test_matmul_backward_only_left_requires_grad(self):
        A_np = np.random.randn(2, 3).astype(np.float32)
        B_np = np.random.randn(3, 4).astype(np.float32)

        A = self._tensor_from_numpy(A_np, requires_grad=True)
        B = self._tensor_from_numpy(B_np, requires_grad=False)

        y = A @ B
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(A.grad)
        self.assertIsNone(B.grad)

        grad_out = np.ones((2, 4), dtype=np.float32)
        expected_dA = grad_out @ B_np.T
        self.assertTrue(
            np.allclose(A.grad.to_numpy(), expected_dA, rtol=1e-5, atol=1e-6)
        )

    def test_matmul_backward_only_right_requires_grad(self):
        A_np = np.random.randn(2, 3).astype(np.float32)
        B_np = np.random.randn(3, 4).astype(np.float32)

        A = self._tensor_from_numpy(A_np, requires_grad=False)
        B = self._tensor_from_numpy(B_np, requires_grad=True)

        y = A @ B
        loss = y.sum()
        loss.backward()

        self.assertIsNone(A.grad)
        self.assertIsNotNone(B.grad)

        grad_out = np.ones((2, 4), dtype=np.float32)
        expected_dB = A_np.T @ grad_out
        self.assertTrue(
            np.allclose(B.grad.to_numpy(), expected_dB, rtol=1e-5, atol=1e-6)
        )


class TestTensorTransposeForwardBackward(TestCase, _TensorFactoryMixin):
    def test_transpose_forward_matches_numpy(self):
        x = self._tensor_from_numpy(np.random.randn(2, 5), requires_grad=False)
        y = x.T

        expected = x.to_numpy().T
        self.assertEqual(y.shape, (5, 2))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_transpose_rejects_non_2d(self):
        x = self._tensor_from_numpy(np.random.randn(2, 3, 4), requires_grad=False)
        with self.assertRaises(ValueError):
            _ = x.T

    def test_transpose_backward_routes_grad(self):
        """
        For y = x.T, loss = sum(y), grad_out = ones(shape(y)),
        dx = grad_out.T = ones(shape(x)).
        """
        x = self._tensor_from_numpy(np.random.randn(3, 4), requires_grad=True)
        y = x.T
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        expected = np.ones((3, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_transpose_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.T


class TestTensorSumAxisForward(TestCase, _TensorFactoryMixin):
    def test_sum_axis0_forward_matches_numpy(self):
        x_np = np.random.randn(4, 3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=0)
        expected = np.sum(x_np, axis=0)

        self.assertEqual(y.shape, (3,))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_axis1_forward_matches_numpy(self):
        x_np = np.random.randn(4, 3).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=1)
        expected = np.sum(x_np, axis=1)

        self.assertEqual(y.shape, (4,))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_negative_axis_forward(self):
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = self._tensor_from_numpy(x_np, requires_grad=False)

        y = x.sum(axis=-1)
        expected = np.sum(x_np, axis=-1)

        self.assertEqual(y.shape, (2, 3))
        self.assertTrue(np.allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-6))

    def test_sum_axis_out_of_bounds_raises(self):
        x = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=False)
        with self.assertRaises(ValueError):
            _ = x.sum(axis=2)

    def test_sum_axis_non_int_raises(self):
        x = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=False)
        with self.assertRaises(TypeError):
            _ = x.sum(axis="0")  # type: ignore[arg-type]

    def test_sum_axis_on_scalar_raises(self):
        s = Tensor((), Device("cpu"), requires_grad=False)
        s.copy_from_numpy(np.array(1.0, dtype=np.float32))
        with self.assertRaises(ValueError):
            _ = s.sum(axis=0)

    def test_sum_axis_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.sum(axis=0)


class TestTensorSumAxisBackward(TestCase, _TensorFactoryMixin):
    def test_sum_axis0_backward_broadcasts_grad(self):
        """
        x: (B, C), y = sum(x, axis=0) -> (C,)
        loss = sum(y) => grad_y = ones(C,)
        grad_x should be ones(B, C)
        """
        x = self._tensor_from_numpy(np.random.randn(5, 4), requires_grad=True)
        y = x.sum(axis=0)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        expected = np.ones((5, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_sum_axis1_backward_broadcasts_grad(self):
        """
        x: (B, C), y = sum(x, axis=1) -> (B,)
        loss = sum(y) => grad_y = ones(B,)
        grad_x should be ones(B, C)
        """
        x = self._tensor_from_numpy(np.random.randn(5, 4), requires_grad=True)
        y = x.sum(axis=1)
        loss = y.sum()
        loss.backward()

        expected = np.ones((5, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_sum_axis_negative_backward(self):
        """
        x: (2,3,4), y = sum(x, axis=-1) -> (2,3)
        loss = sum(y) => grad_y = ones(2,3)
        grad_x should be ones(2,3,4)
        """
        x = self._tensor_from_numpy(np.random.randn(2, 3, 4), requires_grad=True)
        y = x.sum(axis=-1)
        loss = y.sum()
        loss.backward()

        expected = np.ones((2, 3, 4), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))


if __name__ == "__main__":
    unittest.main()
