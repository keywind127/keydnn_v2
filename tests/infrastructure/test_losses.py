# tests/test_loss_mse_sse.py

import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.infrastructure._losses import SSEFn, MSEFn
from src.keydnn.infrastructure._losses import (
    BinaryCrossEntropyFn,
    CategoricalCrossEntropyFn,
)


def _cpu() -> Device:
    # Adjust if your Device constructor differs.
    # Common patterns: Device("cpu") or Device("cpu:0")
    return Device("cpu")


def _tensor_from_np(arr: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=_cpu(), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def _scalar_from_float(x: float) -> Tensor:
    # A scalar tensor: shape=()
    t = Tensor(shape=(), device=_cpu(), requires_grad=False)
    t.copy_from_numpy(np.array(x, dtype=np.float32))
    return t


def _as_scalar_float(t: Tensor) -> float:
    v = t.to_numpy()
    # Supports either shape=() or shape=(1,)
    return float(np.asarray(v).reshape(-1)[0])


def tensor_from_np(arr, *, requires_grad: bool = False) -> Tensor:
    """Create a CPU Tensor initialized from a NumPy array (float32)."""
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=_cpu(), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def _finite_diff_grad(
    loss_forward, pred_np: np.ndarray, target_np: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """
    Central difference gradient wrt pred.
    loss_forward(ctx, pred_tensor, target_tensor) -> scalar tensor
    """
    pred_np = np.asarray(pred_np, dtype=np.float32)
    target_np = np.asarray(target_np, dtype=np.float32)
    grad = np.zeros_like(pred_np, dtype=np.float32)

    it = np.nditer(pred_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        pred_plus = pred_np.copy()
        pred_minus = pred_np.copy()
        pred_plus[idx] += eps
        pred_minus[idx] -= eps

        pred_t_plus = _tensor_from_np(pred_plus, requires_grad=False)
        pred_t_minus = _tensor_from_np(pred_minus, requires_grad=False)
        target_t = _tensor_from_np(target_np, requires_grad=False)

        ctx1 = Context(parents=(pred_t_plus, target_t), backward_fn=lambda _g: ())
        ctx2 = Context(parents=(pred_t_minus, target_t), backward_fn=lambda _g: ())

        loss_plus = _as_scalar_float(loss_forward(ctx1, pred_t_plus, target_t))
        loss_minus = _as_scalar_float(loss_forward(ctx2, pred_t_minus, target_t))

        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return grad


class TestSSEFn(unittest.TestCase):
    def test_forward_value(self):
        pred = _tensor_from_np(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        target = _tensor_from_np(np.array([1.0, 1.0, 2.0], dtype=np.float32))

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        loss = SSEFn.forward(ctx, pred, target)

        # (0^2 + 1^2 + 1^2) = 2
        self.assertAlmostEqual(_as_scalar_float(loss), 2.0, places=6)

    def test_backward_grads_match_analytic(self):
        pred_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target_np = np.array([1.0, 1.0, 2.0], dtype=np.float32)

        pred = _tensor_from_np(pred_np, requires_grad=True)
        target = _tensor_from_np(target_np, requires_grad=True)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = SSEFn.forward(ctx, pred, target)

        grad_out = _scalar_from_float(1.0)
        grad_pred, grad_target = SSEFn.backward(ctx, grad_out)

        expected_pred = 2.0 * (pred_np - target_np)
        expected_target = -2.0 * (pred_np - target_np)

        np.testing.assert_allclose(
            grad_pred.to_numpy(), expected_pred, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            grad_target.to_numpy(), expected_target, rtol=1e-5, atol=1e-6
        )

    def test_backward_scales_with_grad_out(self):
        pred_np = np.array([2.0, -1.0], dtype=np.float32)
        target_np = np.array([1.0, 3.0], dtype=np.float32)

        pred = _tensor_from_np(pred_np, requires_grad=True)
        target = _tensor_from_np(target_np, requires_grad=True)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = SSEFn.forward(ctx, pred, target)

        g = 7.5
        grad_out = _scalar_from_float(g)
        grad_pred, grad_target = SSEFn.backward(ctx, grad_out)

        base_pred = 2.0 * (pred_np - target_np)
        base_target = -2.0 * (pred_np - target_np)

        np.testing.assert_allclose(
            grad_pred.to_numpy(), g * base_pred, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            grad_target.to_numpy(), g * base_target, rtol=1e-5, atol=1e-6
        )

    def test_numeric_grad_pred_matches(self):
        pred_np = np.array([[0.2, -1.3], [2.5, 0.7]], dtype=np.float32)
        target_np = np.array([[0.1, 0.4], [-0.5, 0.3]], dtype=np.float32)

        pred = _tensor_from_np(pred_np, requires_grad=True)
        target = _tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = SSEFn.forward(ctx, pred, target)

        grad_out = _scalar_from_float(1.0)
        grad_pred, _grad_target = SSEFn.backward(ctx, grad_out)

        numeric = _finite_diff_grad(SSEFn.forward, pred_np, target_np, eps=1e-3)
        np.testing.assert_allclose(grad_pred.to_numpy(), numeric, rtol=2e-3, atol=2e-3)


class TestMSEFn(unittest.TestCase):
    def test_forward_value(self):
        pred = _tensor_from_np(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        target = _tensor_from_np(np.array([1.0, 1.0, 2.0], dtype=np.float32))

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        loss = MSEFn.forward(ctx, pred, target)

        # SSE=2, N=3 => 2/3
        self.assertAlmostEqual(_as_scalar_float(loss), 2.0 / 3.0, places=6)

    def test_backward_grads_match_analytic(self):
        pred_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target_np = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        n = pred_np.size

        pred = _tensor_from_np(pred_np, requires_grad=True)
        target = _tensor_from_np(target_np, requires_grad=True)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = MSEFn.forward(ctx, pred, target)

        grad_out = _scalar_from_float(1.0)
        grad_pred, grad_target = MSEFn.backward(ctx, grad_out)

        expected_pred = (2.0 / n) * (pred_np - target_np)
        expected_target = (-2.0 / n) * (pred_np - target_np)

        np.testing.assert_allclose(
            grad_pred.to_numpy(), expected_pred, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            grad_target.to_numpy(), expected_target, rtol=1e-5, atol=1e-6
        )

    def test_numeric_grad_pred_matches(self):
        pred_np = np.array([[0.25, -0.75, 1.1]], dtype=np.float32)
        target_np = np.array([[0.15, 0.20, -0.3]], dtype=np.float32)

        pred = _tensor_from_np(pred_np, requires_grad=True)
        target = _tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = MSEFn.forward(ctx, pred, target)

        grad_out = _scalar_from_float(1.0)
        grad_pred, _ = MSEFn.backward(ctx, grad_out)

        numeric = _finite_diff_grad(MSEFn.forward, pred_np, target_np, eps=1e-3)
        np.testing.assert_allclose(grad_pred.to_numpy(), numeric, rtol=2e-3, atol=2e-3)


def scalar_tensor(x: float) -> Tensor:
    t = Tensor(shape=(), device=_cpu(), requires_grad=False)
    t.copy_from_numpy(np.array(x, dtype=np.float32))
    return t


def as_float(t: Tensor) -> float:
    return float(np.asarray(t.to_numpy()).reshape(-1)[0])


def finite_diff_grad_pred(
    forward_fn, pred_np: np.ndarray, target_np: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    """
    Central difference gradient wrt pred only.
    forward_fn signature: forward_fn(ctx, pred_tensor, target_tensor) -> scalar Tensor
    """
    pred_np = np.asarray(pred_np, dtype=np.float32)
    target_np = np.asarray(target_np, dtype=np.float32)
    grad = np.zeros_like(pred_np, dtype=np.float32)

    it = np.nditer(pred_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        p_plus = pred_np.copy()
        p_minus = pred_np.copy()
        p_plus[idx] += eps
        p_minus[idx] -= eps

        pred_plus = tensor_from_np(p_plus, requires_grad=False)
        pred_minus = tensor_from_np(p_minus, requires_grad=False)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx_plus = Context(parents=(pred_plus, target), backward_fn=lambda _g: ())
        ctx_minus = Context(parents=(pred_minus, target), backward_fn=lambda _g: ())

        loss_plus = as_float(forward_fn(ctx_plus, pred_plus, target))
        loss_minus = as_float(forward_fn(ctx_minus, pred_minus, target))

        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return grad


class TestBinaryCrossEntropyFn(unittest.TestCase):
    def test_forward_value_mean_reduction(self):
        """
        Spec:
          BCE = mean( -[ y*log(p) + (1-y)*log(1-p) ] )
        """
        pred_np = np.array([0.2, 0.7, 0.9], dtype=np.float32)
        target_np = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        pred = tensor_from_np(pred_np)
        target = tensor_from_np(target_np)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        loss = BinaryCrossEntropyFn.forward(ctx, pred, target)

        expected = -(
            target_np * np.log(pred_np) + (1.0 - target_np) * np.log(1.0 - pred_np)
        ).mean()

        self.assertAlmostEqual(as_float(loss), float(expected), places=6)

    def test_backward_grad_pred_matches_analytic(self):
        """
        For BCE mean reduction:
          dL/dp = (p - y) / (p*(1-p)) / N
        """
        pred_np = np.array([0.2, 0.7, 0.9], dtype=np.float32)
        target_np = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        n = pred_np.size

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = BinaryCrossEntropyFn.forward(ctx, pred, target)

        grad_out = scalar_tensor(1.0)
        grad_pred, grad_target = BinaryCrossEntropyFn.backward(ctx, grad_out)

        expected_grad_pred = ((pred_np - target_np) / (pred_np * (1.0 - pred_np))) / n

        np.testing.assert_allclose(
            grad_pred.to_numpy(), expected_grad_pred, rtol=1e-5, atol=1e-6
        )
        self.assertTrue(grad_target is None or isinstance(grad_target, Tensor))

    def test_backward_scales_with_grad_out(self):
        pred_np = np.array([0.3, 0.6], dtype=np.float32)
        target_np = np.array([1.0, 0.0], dtype=np.float32)
        n = pred_np.size

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = BinaryCrossEntropyFn.forward(ctx, pred, target)

        g = 7.5
        grad_out = scalar_tensor(g)
        grad_pred, _ = BinaryCrossEntropyFn.backward(ctx, grad_out)

        base = ((pred_np - target_np) / (pred_np * (1.0 - pred_np))) / n
        np.testing.assert_allclose(grad_pred.to_numpy(), g * base, rtol=1e-5, atol=1e-6)

    def test_numeric_grad_pred_matches(self):
        pred_np = np.array([[0.25, 0.65], [0.8, 0.35]], dtype=np.float32)
        target_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = BinaryCrossEntropyFn.forward(ctx, pred, target)

        grad_pred, _ = BinaryCrossEntropyFn.backward(ctx, scalar_tensor(1.0))
        numeric = finite_diff_grad_pred(
            BinaryCrossEntropyFn.forward, pred_np, target_np, eps=1e-4
        )

        # Finite diff around log can be sensitive; tolerances slightly looser
        np.testing.assert_allclose(grad_pred.to_numpy(), numeric, rtol=2e-3, atol=2e-3)

    def test_shape_mismatch_raises(self):
        pred = tensor_from_np(np.array([0.2, 0.7, 0.9], dtype=np.float32))
        target = tensor_from_np(
            np.array([[0.0, 1.0, 1.0]], dtype=np.float32)
        )  # different shape

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        with self.assertRaises(ValueError):
            _ = BinaryCrossEntropyFn.forward(ctx, pred, target)


class TestCategoricalCrossEntropyFn(unittest.TestCase):
    def test_forward_value_batch_mean(self):
        """
        Spec:
          CCE = -(sum(y * log(p))) / N
        with p,y shape (N,C), y one-hot.
        """
        pred_np = np.array(
            [
                [0.1, 0.7, 0.2],
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

        pred = tensor_from_np(pred_np)
        target = tensor_from_np(target_np)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        loss = CategoricalCrossEntropyFn.forward(ctx, pred, target)

        n = pred_np.shape[0]
        expected = -(target_np * np.log(pred_np)).sum() / n

        self.assertAlmostEqual(as_float(loss), float(expected), places=6)

    def test_backward_grad_pred_matches_analytic(self):
        """
        For CCE with batch mean:
          L = -(sum(y*log(p))) / N
          dL/dp = -(y/p) / N
        """
        pred_np = np.array(
            [
                [0.1, 0.7, 0.2],
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

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = CategoricalCrossEntropyFn.forward(ctx, pred, target)

        grad_pred, grad_target = CategoricalCrossEntropyFn.backward(
            ctx, scalar_tensor(1.0)
        )

        n = pred_np.shape[0]
        expected = -(target_np / pred_np) / n

        np.testing.assert_allclose(grad_pred.to_numpy(), expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(grad_target is None or isinstance(grad_target, Tensor))

    def test_backward_scales_with_grad_out(self):
        pred_np = np.array(
            [
                [0.2, 0.5, 0.3],
            ],
            dtype=np.float32,
        )
        target_np = np.array(
            [
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = CategoricalCrossEntropyFn.forward(ctx, pred, target)

        g = 3.25
        grad_pred, _ = CategoricalCrossEntropyFn.backward(ctx, scalar_tensor(g))

        n = pred_np.shape[0]
        base = -(target_np / pred_np) / n
        np.testing.assert_allclose(grad_pred.to_numpy(), g * base, rtol=1e-5, atol=1e-6)

    def test_numeric_grad_pred_matches(self):
        pred_np = np.array(
            [
                [0.15, 0.35, 0.5],
                [0.6, 0.25, 0.15],
            ],
            dtype=np.float32,
        )
        target_np = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        pred = tensor_from_np(pred_np, requires_grad=True)
        target = tensor_from_np(target_np, requires_grad=False)

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        _ = CategoricalCrossEntropyFn.forward(ctx, pred, target)

        grad_pred, _ = CategoricalCrossEntropyFn.backward(ctx, scalar_tensor(1.0))
        numeric = finite_diff_grad_pred(
            CategoricalCrossEntropyFn.forward, pred_np, target_np, eps=1e-4
        )

        np.testing.assert_allclose(grad_pred.to_numpy(), numeric, rtol=2e-3, atol=2e-3)

    def test_shape_mismatch_raises(self):
        pred = tensor_from_np(np.array([[0.2, 0.8]], dtype=np.float32))
        target = tensor_from_np(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )  # different C

        ctx = Context(parents=(pred, target), backward_fn=lambda _g: ())
        with self.assertRaises(ValueError):
            _ = CategoricalCrossEntropyFn.forward(ctx, pred, target)


if __name__ == "__main__":
    unittest.main()
