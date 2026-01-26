"""
Unit tests for Module-based loss layers.

These tests validate that KeyDNN's infrastructure loss `Module` wrappers:
- match NumPy reference computations for forward passes,
- attach an autograd `Context` when any parent requires gradients,
- delegate backward computation to the corresponding loss `Function`,
- return gradients with correct shapes and values,
- respect the "target is constant" convention for classification losses
  (i.e., BCE/CCE return `None` for target gradients).

Notes
-----
- These tests follow the same style as the activation module tests:
  they inspect the attached `Context` and call `ctx.backward_fn(...)` directly,
  without relying on a full `Tensor.backward()` engine.
- For classification losses, numerical stability is not addressed here beyond
  choosing safe probability inputs away from 0/1.
"""

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.losses._modules import (
    SSE,
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)


def _cpu() -> Device:
    """Return a CPU device instance."""
    return Device("cpu")


def tensor_from_np(arr: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    """
    Create a CPU Tensor from a NumPy array using only public APIs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    requires_grad : bool, default=False
        Whether the Tensor should track gradients.

    Returns
    -------
    Tensor
        A KeyDNN Tensor with copied data.
    """
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=_cpu(), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def scalar_tensor(x: float, *, requires_grad: bool = False) -> Tensor:
    """
    Create a scalar Tensor with a specified float value.

    Parameters
    ----------
    x : float
        Scalar value to store.
    requires_grad : bool, default=False
        Whether the Tensor should track gradients.

    Returns
    -------
    Tensor
        A scalar KeyDNN Tensor.
    """
    t = Tensor(shape=(), device=_cpu(), requires_grad=requires_grad)
    t.copy_from_numpy(np.array(x, dtype=np.float32))
    return t


def ones_scalar() -> Tensor:
    """
    Convenience helper: return a scalar Tensor filled with 1.

    Returns
    -------
    Tensor
        Scalar tensor = 1.0 (no grad).
    """
    return scalar_tensor(1.0, requires_grad=False)


def as_np(t: Tensor) -> np.ndarray:
    """
    Convert a Tensor to a float32 NumPy array.

    Parameters
    ----------
    t : Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        NumPy view/copy of tensor values.
    """
    return np.asarray(t.to_numpy(), dtype=np.float32)


class _ModuleLossAsserts:
    """
    Mixin assertions for loss module behavior.
    """

    def assert_ctx_attached_two_parents(
        self, out: Tensor, a: Tensor, b: Tensor
    ) -> None:
        """
        Assert output has a Context attached with exactly two parents (a, b).

        Parameters
        ----------
        out : Tensor
            Output tensor.
        a : Tensor
            First parent tensor.
        b : Tensor
            Second parent tensor.
        """
        ctx = out._get_ctx()
        self.assertIsNotNone(ctx, "Expected Context to be attached to output Tensor.")
        assert ctx is not None
        self.assertEqual(len(ctx.parents), 2)
        self.assertIs(ctx.parents[0], a)
        self.assertIs(ctx.parents[1], b)
        self.assertTrue(callable(ctx.backward_fn))

    def assert_ctx_not_attached(self, out: Tensor) -> None:
        """
        Assert output has no Context attached.

        Parameters
        ----------
        out : Tensor
            Output tensor.
        """
        self.assertIsNone(out._get_ctx(), "Did not expect Context to be attached.")


class TestSSEModule(unittest.TestCase, _ModuleLossAsserts):
    """
    Tests for the SSE loss module wrapper.
    """

    def test_forward_matches_numpy(self) -> None:
        """
        SSE forward should match NumPy: sum((pred-target)^2), scalar output.
        """
        loss_mod = SSE()

        pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        target_np = np.array([[0.5, 2.5], [2.0, 6.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        expected = np.sum((pred_np - target_np) ** 2, dtype=np.float32)
        self.assertEqual(out.shape, ())
        np.testing.assert_allclose(
            as_np(out), np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-6
        )

        self.assertFalse(out.requires_grad)
        self.assert_ctx_not_attached(out)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        """
        SSE should attach Context when pred requires grad, and backward should
        return grads for (pred, target).
        """
        loss_mod = SSE()

        pred_np = np.array([[1.0, -2.0], [0.25, 3.5]], dtype=np.float32)
        target_np = np.array([[0.5, 1.0], [0.0, 2.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        self.assertTrue(out.requires_grad)
        self.assert_ctx_attached_two_parents(out, pred, target)

        ctx = out._get_ctx()
        assert ctx is not None

        self.assertEqual(len(ctx.saved_tensors), 1)
        diff_saved = ctx.saved_tensors[0]
        self.assertEqual(diff_saved.shape, pred.shape)

        grads = ctx.backward_fn(ones_scalar())
        self.assertEqual(len(grads), 2)

        grad_pred, grad_target = grads
        self.assertIsNotNone(grad_pred)
        self.assertIsNotNone(grad_target)
        assert grad_pred is not None and grad_target is not None

        diff = pred_np - target_np
        expected_grad_pred = 2.0 * diff
        expected_grad_target = -2.0 * diff

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            as_np(grad_target), expected_grad_target, rtol=1e-6, atol=1e-6
        )

    def test_target_requires_grad_also_attaches_ctx(self) -> None:
        """
        SSE should attach Context when target requires grad even if pred does not.
        """
        loss_mod = SSE()

        pred_np = np.array([[1.0, 2.0]], dtype=np.float32)
        target_np = np.array([[1.5, 0.5]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=True)

        out = loss_mod.forward(pred, target)

        self.assertTrue(out.requires_grad)
        self.assert_ctx_attached_two_parents(out, pred, target)


class TestMSEModule(unittest.TestCase, _ModuleLossAsserts):
    """
    Tests for the MSE loss module wrapper.
    """

    def test_forward_matches_numpy(self) -> None:
        """
        MSE forward should match NumPy: mean((pred-target)^2), scalar output.
        """
        loss_mod = MSE()

        pred_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target_np = np.array([[0.0, 2.5, 2.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        expected = np.mean((pred_np - target_np) ** 2, dtype=np.float32)
        self.assertEqual(out.shape, ())
        np.testing.assert_allclose(
            as_np(out), np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-6
        )

        self.assertFalse(out.requires_grad)
        self.assert_ctx_not_attached(out)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        """
        MSE should attach Context when pred requires grad, and backward should
        return grads for (pred, target) scaled by 2/N.
        """
        loss_mod = MSE()

        pred_np = np.array([[1.0, -2.0], [0.25, 3.5]], dtype=np.float32)
        target_np = np.array([[0.5, 1.0], [0.0, 2.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        self.assertTrue(out.requires_grad)
        self.assert_ctx_attached_two_parents(out, pred, target)

        ctx = out._get_ctx()
        assert ctx is not None

        self.assertEqual(len(ctx.saved_tensors), 1)
        self.assertIn("n", ctx.saved_meta)
        self.assertEqual(int(ctx.saved_meta["n"]), pred_np.size)

        grads = ctx.backward_fn(ones_scalar())
        self.assertEqual(len(grads), 2)

        grad_pred, grad_target = grads
        self.assertIsNotNone(grad_pred)
        self.assertIsNotNone(grad_target)
        assert grad_pred is not None and grad_target is not None

        diff = pred_np - target_np
        n = float(pred_np.size)
        expected_grad_pred = (2.0 / n) * diff
        expected_grad_target = -(2.0 / n) * diff

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            as_np(grad_target), expected_grad_target, rtol=1e-6, atol=1e-6
        )

    def test_backward_scales_with_upstream_grad(self) -> None:
        """
        MSE backward should be linear in upstream scalar grad_out.
        """
        loss_mod = MSE()

        pred_np = np.array([[1.0, 2.0]], dtype=np.float32)
        target_np = np.array([[0.0, 0.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        ctx = out._get_ctx()
        self.assertIsNotNone(ctx)
        assert ctx is not None

        g = 3.25
        grads = ctx.backward_fn(scalar_tensor(g))
        grad_pred, grad_target = grads
        assert grad_pred is not None and grad_target is not None

        diff = pred_np - target_np
        n = float(pred_np.size)
        expected_grad_pred = (2.0 / n) * diff * g
        expected_grad_target = -(2.0 / n) * diff * g

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            as_np(grad_target), expected_grad_target, rtol=1e-6, atol=1e-6
        )


class TestBinaryCrossEntropyModule(unittest.TestCase, _ModuleLossAsserts):
    """
    Tests for the BinaryCrossEntropy loss module wrapper.
    """

    def test_forward_matches_numpy(self) -> None:
        """
        BCE forward should match NumPy mean:
          -[t*log(p) + (1-t)*log(1-p)].
        """
        loss_mod = BinaryCrossEntropy()

        pred_np = np.array([[0.2, 0.9], [0.7, 0.4]], dtype=np.float32)
        target_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        expected = -(
            target_np * np.log(pred_np) + (1.0 - target_np) * np.log(1.0 - pred_np)
        )
        expected = np.mean(expected, dtype=np.float32)

        self.assertEqual(out.shape, ())
        np.testing.assert_allclose(
            as_np(out), np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-6
        )

        self.assertFalse(out.requires_grad)
        self.assert_ctx_not_attached(out)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        """
        BCE should attach Context when pred requires grad.
        Backward should return (grad_pred, None).
        """
        loss_mod = BinaryCrossEntropy()

        pred_np = np.array([[0.2, 0.9], [0.7, 0.4]], dtype=np.float32)
        target_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        self.assertTrue(out.requires_grad)
        self.assert_ctx_attached_two_parents(out, pred, target)

        ctx = out._get_ctx()
        assert ctx is not None

        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIn("n", ctx.saved_meta)
        self.assertEqual(int(ctx.saved_meta["n"]), pred_np.size)

        grad_pred, grad_target = ctx.backward_fn(ones_scalar())
        self.assertIsNotNone(grad_pred)
        self.assertIsNone(grad_target)

        n = float(pred_np.size)
        expected_grad_pred = (pred_np - target_np) / (pred_np * (1.0 - pred_np)) / n

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-5, atol=1e-6
        )

    def test_backward_scales_with_upstream_grad(self) -> None:
        """
        BCE backward should scale linearly with upstream scalar grad_out.
        """
        loss_mod = BinaryCrossEntropy()

        pred_np = np.array([[0.25, 0.75]], dtype=np.float32)
        target_np = np.array([[0.0, 1.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        ctx = out._get_ctx()
        self.assertIsNotNone(ctx)
        assert ctx is not None

        g = 2.0
        grad_pred, grad_target = ctx.backward_fn(scalar_tensor(g))
        self.assertIsNone(grad_target)
        assert grad_pred is not None

        n = float(pred_np.size)
        expected_grad_pred = (
            (pred_np - target_np) / (pred_np * (1.0 - pred_np)) / n
        ) * g

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-5, atol=1e-6
        )


class TestCategoricalCrossEntropyModule(unittest.TestCase, _ModuleLossAsserts):
    """
    Tests for the CategoricalCrossEntropy loss module wrapper.
    """

    def test_forward_matches_numpy(self) -> None:
        """
        CCE forward should match NumPy:
          -sum(target*log(pred)) / N
        where N is batch size.
        """
        loss_mod = CategoricalCrossEntropy()

        pred_np = np.array(
            [
                [0.1, 0.6, 0.3],
                [0.8, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        target_np = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        pred = tensor_from_np(pred_np, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        expected = -(
            np.sum(target_np * np.log(pred_np), dtype=np.float32) / pred_np.shape[0]
        )

        self.assertEqual(out.shape, ())
        np.testing.assert_allclose(
            as_np(out), np.array(expected, dtype=np.float32), rtol=1e-6, atol=1e-6
        )

        self.assertFalse(out.requires_grad)
        self.assert_ctx_not_attached(out)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        """
        CCE should attach Context when pred requires grad.
        Backward should return (grad_pred, None).
        """
        loss_mod = CategoricalCrossEntropy()

        pred_np = np.array(
            [
                [0.1, 0.6, 0.3],
                [0.8, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        target_np = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        self.assertTrue(out.requires_grad)
        self.assert_ctx_attached_two_parents(out, pred, target)

        ctx = out._get_ctx()
        assert ctx is not None

        self.assertEqual(len(ctx.saved_tensors), 2)

        grad_pred, grad_target = ctx.backward_fn(ones_scalar())
        self.assertIsNotNone(grad_pred)
        self.assertIsNone(grad_target)
        assert grad_pred is not None

        n = float(pred_np.shape[0])
        expected_grad_pred = -(target_np / pred_np) / n

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-6, atol=1e-6
        )

    def test_backward_scales_with_upstream_grad(self) -> None:
        """
        CCE backward should scale linearly with upstream scalar grad_out.
        """
        loss_mod = CategoricalCrossEntropy()

        pred_np = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
        target_np = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        out = loss_mod.forward(pred, target)

        ctx = out._get_ctx()
        self.assertIsNotNone(ctx)
        assert ctx is not None

        g = 4.0
        grad_pred, grad_target = ctx.backward_fn(scalar_tensor(g))
        self.assertIsNone(grad_target)
        assert grad_pred is not None

        n = float(pred_np.shape[0])
        expected_grad_pred = (-(target_np / pred_np) / n) * g

        np.testing.assert_allclose(
            as_np(grad_pred), expected_grad_pred, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
