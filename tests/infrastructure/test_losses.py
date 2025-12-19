# tests/test_loss_mse_sse.py

import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.infrastructure._losses import SSEFn, MSEFn


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


if __name__ == "__main__":
    unittest.main()
