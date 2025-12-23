import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor


# -----------------------------
# Helpers (match your style)
# -----------------------------
def _tensor_from_numpy(
    arr: np.ndarray, device: Device, requires_grad: bool = False
) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestTensorConcatForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_concat_axis0_matches_numpy(self):
        a = _tensor_from_numpy(
            np.array([[1, 2], [3, 4]], dtype=np.float32), self.device
        )
        b = _tensor_from_numpy(np.array([[5, 6]], dtype=np.float32), self.device)

        y = Tensor.concat([a, b], axis=0)

        ref = np.concatenate([a.to_numpy(), b.to_numpy()], axis=0)
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y.to_numpy(), ref, rtol=0, atol=0)

    def test_concat_axis1_matches_numpy(self):
        a = _tensor_from_numpy(
            np.array([[1, 2], [3, 4]], dtype=np.float32), self.device
        )
        b = _tensor_from_numpy(np.array([[5], [6]], dtype=np.float32), self.device)

        y = Tensor.concat([a, b], axis=1)

        ref = np.concatenate([a.to_numpy(), b.to_numpy()], axis=1)
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y.to_numpy(), ref, rtol=0, atol=0)

    def test_concat_negative_axis_matches_numpy(self):
        a = _tensor_from_numpy(np.random.randn(2, 3, 4).astype(np.float32), self.device)
        b = _tensor_from_numpy(np.random.randn(2, 3, 5).astype(np.float32), self.device)

        y = Tensor.concat([a, b], axis=-1)

        ref = np.concatenate([a.to_numpy(), b.to_numpy()], axis=-1)
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y.to_numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_concat_requires_non_empty(self):
        with self.assertRaises(ValueError):
            _ = Tensor.concat([], axis=0)

    def test_concat_rejects_scalar_inputs(self):
        a = _tensor_from_numpy(np.array(1.0, dtype=np.float32), self.device)
        b = _tensor_from_numpy(np.array(2.0, dtype=np.float32), self.device)
        with self.assertRaises(ValueError):
            _ = Tensor.concat([a, b], axis=0)

    def test_concat_axis_out_of_bounds_raises(self):
        a = _tensor_from_numpy(np.ones((2, 3), dtype=np.float32), self.device)
        b = _tensor_from_numpy(np.ones((2, 3), dtype=np.float32), self.device)

        with self.assertRaises(ValueError):
            _ = Tensor.concat([a, b], axis=2)

        with self.assertRaises(ValueError):
            _ = Tensor.concat([a, b], axis=-3)

    def test_concat_shape_mismatch_non_axis_raises(self):
        a = _tensor_from_numpy(np.ones((2, 3), dtype=np.float32), self.device)
        b = _tensor_from_numpy(np.ones((4, 3), dtype=np.float32), self.device)

        # concat on axis=1 requires dim0 to match -> should raise
        with self.assertRaises(ValueError):
            _ = Tensor.concat([a, b], axis=1)


class TestTensorConcatAutograd(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_concat_attaches_ctx_when_any_requires_grad(self):
        a = _tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), self.device, requires_grad=True
        )
        b = _tensor_from_numpy(
            np.ones((2, 1), dtype=np.float32), self.device, requires_grad=False
        )

        y = Tensor.concat([a, b], axis=1)
        self.assertTrue(y.requires_grad)
        self.assertIsNotNone(
            y._get_ctx(),
            "Expected Context to be attached when any parent requires grad",
        )

        ctx = y._get_ctx()
        self.assertEqual(len(ctx.parents), 2)
        self.assertIs(ctx.parents[0], a)
        self.assertIs(ctx.parents[1], b)

    def test_concat_no_ctx_when_no_requires_grad(self):
        a = _tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), self.device, requires_grad=False
        )
        b = _tensor_from_numpy(
            np.ones((2, 1), dtype=np.float32), self.device, requires_grad=False
        )

        y = Tensor.concat([a, b], axis=1)
        self.assertFalse(y.requires_grad)
        self.assertIsNone(
            y._get_ctx(), "Expected no Context when no parent requires grad"
        )

    def test_backward_splits_grad_correctly_axis1(self):
        """
        If y = concat([a, b], axis=1), and loss = y.sum(),
        then grad for a and b should both be ones of their shapes.
        """
        a = _tensor_from_numpy(
            np.random.randn(2, 3).astype(np.float32), self.device, requires_grad=True
        )
        b = _tensor_from_numpy(
            np.random.randn(2, 1).astype(np.float32), self.device, requires_grad=True
        )

        y = Tensor.concat([a, b], axis=1)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

        np.testing.assert_allclose(
            a.grad.to_numpy(), np.ones(a.shape, dtype=np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            b.grad.to_numpy(), np.ones(b.shape, dtype=np.float32), rtol=0, atol=0
        )

    def test_backward_splits_grad_correctly_axis0(self):
        """
        If y = concat([a, b], axis=0), and loss = y.sum(),
        then grads should be ones of each original shape.
        """
        a = _tensor_from_numpy(
            np.random.randn(2, 3).astype(np.float32), self.device, requires_grad=True
        )
        b = _tensor_from_numpy(
            np.random.randn(1, 3).astype(np.float32), self.device, requires_grad=True
        )

        y = Tensor.concat([a, b], axis=0)
        y.sum().backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        np.testing.assert_allclose(
            a.grad.to_numpy(), np.ones(a.shape, dtype=np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            b.grad.to_numpy(), np.ones(b.shape, dtype=np.float32), rtol=0, atol=0
        )

    def test_backward_respects_requires_grad_flags(self):
        a = _tensor_from_numpy(
            np.random.randn(2, 3).astype(np.float32), self.device, requires_grad=False
        )
        b = _tensor_from_numpy(
            np.random.randn(2, 1).astype(np.float32), self.device, requires_grad=True
        )

        y = Tensor.concat([a, b], axis=1)
        y.sum().backward()

        self.assertIsNone(a.grad, "Expected no grad for a since requires_grad=False")
        self.assertIsNotNone(b.grad, "Expected grad for b since requires_grad=True")
        np.testing.assert_allclose(
            b.grad.to_numpy(), np.ones(b.shape, dtype=np.float32), rtol=0, atol=0
        )

    def test_backward_matches_manual_weighted_sum(self):
        """
        More stringent: y = concat([a,b], axis=1)
        loss = sum(y * w) where w is constant tensor of same shape as y
        Then grads should equal the corresponding w slices.
        """
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 2).astype(np.float32)
        a = _tensor_from_numpy(a_np, self.device, requires_grad=True)
        b = _tensor_from_numpy(b_np, self.device, requires_grad=True)

        y = Tensor.concat([a, b], axis=1)

        w_np = np.arange(np.prod(y.shape), dtype=np.float32).reshape(y.shape) / 10.0
        w = _tensor_from_numpy(w_np, self.device, requires_grad=False)

        loss = (y * w).sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

        w_a = w_np[:, : a.shape[1]]
        w_b = w_np[:, a.shape[1] :]

        np.testing.assert_allclose(a.grad.to_numpy(), w_a, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(b.grad.to_numpy(), w_b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
