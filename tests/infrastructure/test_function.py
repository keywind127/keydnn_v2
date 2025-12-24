import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._function import (
    SigmoidFn,
    ReLUFn,
    LeakyReLUFn,
    TanhFn,
    SoftmaxFn,
)


def make_cpu_tensor(arr: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=Device("cpu"), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def ones_like(t: Tensor) -> Tensor:
    g = Tensor(shape=t.shape, device=t.device, requires_grad=False)
    g.fill(1.0)
    return g


def _cpu() -> Device:
    # Adjust if your Device API differs
    return Device("cpu")


def tensor_from_np(arr, *, requires_grad: bool = False) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=_cpu(), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def scalar_tensor(x: float) -> Tensor:
    t = Tensor(shape=(), device=_cpu(), requires_grad=False)
    t.copy_from_numpy(np.array(x, dtype=np.float32))
    return t


def as_np(t: Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(), dtype=np.float32)


def stable_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def finite_diff_grad_x(
    forward_fn, x_np: np.ndarray, eps: float = 1e-4, axis: int = -1
) -> np.ndarray:
    """
    Central difference gradient wrt x for scalar objective:
      L(x) = sum( softmax(x) * w )
    where w is fixed (in closure) via forward_fn.
    forward_fn: callable(x_tensor) -> scalar float
    """
    x_np = np.asarray(x_np, dtype=np.float32)
    grad = np.zeros_like(x_np, dtype=np.float32)

    it = np.nditer(x_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps

        loss_plus = forward_fn(x_plus)
        loss_minus = forward_fn(x_minus)

        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return grad


class TestReLU(unittest.TestCase):

    def test_relu_forward_values(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        # ctx only needs save_for_backward for these tests
        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = ReLUFn.forward(ctx, x)

        expected = np.maximum(0.0, x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=0, atol=0)

        # Ensure backward saved something (mask)
        self.assertEqual(len(ctx.saved_tensors), 1)

    def test_relu_backward_mask(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        _ = ReLUFn.forward(ctx, x)

        grad_out = make_cpu_tensor(np.ones_like(x_np), requires_grad=False)
        grad_x = ReLUFn.backward(ctx, grad_out)

        # derivative: 1 where x > 0 else 0 (note: x == 0 -> 0 in your implementation)
        expected = (x_np > 0).astype(np.float32)
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=0, atol=0)


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_forward_values(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = SigmoidFn.forward(ctx, x)

        expected = 1.0 / (1.0 + np.exp(-x_np))
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        # Ensure backward saved something (out)
        self.assertEqual(len(ctx.saved_tensors), 1)

    def test_sigmoid_backward_formula(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = SigmoidFn.forward(ctx, x)

        grad_out = make_cpu_tensor(np.ones_like(x_np), requires_grad=False)
        grad_x = SigmoidFn.backward(ctx, grad_out)

        y_np = y.to_numpy()
        expected = y_np * (1.0 - y_np)  # since grad_out is all ones
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_sigmoid_backward_finite_difference(self) -> None:
        """
        Optional: numeric gradient check on a small tensor.
        Checks d/dx sum(sigmoid(x)) == sigmoid(x)*(1-sigmoid(x)).
        """
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = SigmoidFn.forward(ctx, x)

        # analytic grad for sum(y): grad_out is ones
        grad_out = ones_like(y)
        grad_x = SigmoidFn.backward(ctx, grad_out).to_numpy()

        # numeric grad via central differences
        eps = 1e-3
        num_grad = np.zeros_like(x_np, dtype=np.float32)

        def f(x_arr: np.ndarray) -> float:
            # sum(sigmoid(x)) using numpy for scalar function
            s = 1.0 / (1.0 + np.exp(-x_arr))
            return float(np.sum(s))

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                x_pos = x_np.copy()
                x_neg = x_np.copy()
                x_pos[i, j] += eps
                x_neg[i, j] -= eps
                num_grad[i, j] = (f(x_pos) - f(x_neg)) / (2.0 * eps)

        np.testing.assert_allclose(grad_x, num_grad, rtol=1e-2, atol=1e-2)


class TestTanh(unittest.TestCase):

    def test_tanh_forward_values(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = TanhFn.forward(ctx, x)

        expected = np.tanh(x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        # Forward should save output for backward
        self.assertEqual(len(ctx.saved_tensors), 1)

    def test_tanh_backward_formula(self) -> None:
        """
        Checks: d/dx tanh(x) = 1 - tanh(x)^2
        With grad_out = 1, grad_x = 1 - out^2.
        """
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        out = TanhFn.forward(ctx, x)

        grad_out = ones_like(out)
        grad_x = TanhFn.backward(ctx, grad_out)

        out_np = out.to_numpy()
        expected = 1.0 - out_np * out_np
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_tanh_backward_finite_difference(self) -> None:
        """
        Numeric gradient check on sum(tanh(x)).
        grad should match 1 - tanh(x)^2.
        """
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        out = TanhFn.forward(ctx, x)

        # analytic grad for sum(out): grad_out is ones
        grad_out = ones_like(out)
        grad_x = TanhFn.backward(ctx, grad_out).to_numpy()

        # numeric grad via central differences
        eps = 1e-3
        num_grad = np.zeros_like(x_np, dtype=np.float32)

        def f(x_arr: np.ndarray) -> float:
            return float(np.sum(np.tanh(x_arr)))

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                x_pos = x_np.copy()
                x_neg = x_np.copy()
                x_pos[i, j] += eps
                x_neg[i, j] -= eps
                num_grad[i, j] = (f(x_pos) - f(x_neg)) / (2.0 * eps)

        np.testing.assert_allclose(grad_x, num_grad, rtol=1e-2, atol=1e-2)


class TestLeakyReLU(unittest.TestCase):

    def test_leaky_relu_forward_values_default_alpha(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = LeakyReLUFn.forward(ctx, x)  # default alpha=0.01

        alpha = 0.01
        expected = np.where(x_np > 0, x_np, alpha * x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        # Forward should save pos_mask and neg_mask
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIn("alpha", ctx.saved_meta)

    def test_leaky_relu_forward_values_custom_alpha(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        alpha = 0.2
        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = LeakyReLUFn.forward(ctx, x, alpha=alpha)

        expected = np.where(x_np > 0, x_np, alpha * x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(ctx.saved_meta["alpha"], alpha)

    def test_leaky_relu_backward_values(self) -> None:
        """
        Checks derivative:
          1 for x > 0
          alpha for x <= 0
        """
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        alpha = 0.1
        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        _ = LeakyReLUFn.forward(ctx, x, alpha=alpha)

        grad_out_np = np.ones_like(x_np, dtype=np.float32)
        grad_out = make_cpu_tensor(grad_out_np, requires_grad=False)
        grad_x = LeakyReLUFn.backward(ctx, grad_out)

        expected = np.where(x_np > 0, 1.0, alpha).astype(np.float32)
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=0, atol=0)

    def test_leaky_relu_backward_finite_difference(self) -> None:
        """
        Numeric gradient check on sum(leaky_relu(x)).
        """
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        alpha = 0.05
        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        out = LeakyReLUFn.forward(ctx, x, alpha=alpha)

        grad_out = ones_like(out)
        grad_x = LeakyReLUFn.backward(ctx, grad_out).to_numpy()

        eps = 1e-3
        num_grad = np.zeros_like(x_np, dtype=np.float32)

        def leaky_relu_np(z: np.ndarray) -> np.ndarray:
            return np.where(z > 0, z, alpha * z).astype(np.float32)

        def f(z: np.ndarray) -> float:
            return float(np.sum(leaky_relu_np(z)))

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                z_pos = x_np.copy()
                z_neg = x_np.copy()
                z_pos[i, j] += eps
                z_neg[i, j] -= eps
                num_grad[i, j] = (f(z_pos) - f(z_neg)) / (2.0 * eps)

        np.testing.assert_allclose(grad_x, num_grad, rtol=1e-2, atol=1e-2)


class TestSoftmaxFn(unittest.TestCase):
    def test_forward_matches_numpy_2d_axis_last(self):
        x_np = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]], dtype=np.float32)
        x = tensor_from_np(x_np)

        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        y = SoftmaxFn.forward(ctx, x, axis=-1)

        expected = stable_softmax_np(x_np, axis=-1)
        np.testing.assert_allclose(as_np(y), expected, rtol=1e-6, atol=1e-7)

    def test_forward_probabilities_sum_to_one(self):
        x_np = np.array([[10.0, 0.0, -10.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        x = tensor_from_np(x_np)

        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        y = SoftmaxFn.forward(ctx, x, axis=-1)

        sums = np.sum(as_np(y), axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-6, atol=1e-6)

        self.assertTrue(np.all(as_np(y) >= 0.0))

    def test_backward_jvp_matches_closed_form(self):
        """
        Validate backward formula:
          dx = y * (g - sum(g*y, axis, keepdims=True))
        """
        x_np = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
        g_np = np.array([[0.2, -0.1, 0.05], [1.0, -2.0, 0.5]], dtype=np.float32)

        x = tensor_from_np(x_np, requires_grad=True)
        grad_out = tensor_from_np(g_np, requires_grad=False)

        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        y = SoftmaxFn.forward(ctx, x, axis=-1)

        (dx,) = SoftmaxFn.backward(ctx, grad_out)

        y_np = stable_softmax_np(x_np, axis=-1)
        dot = np.sum(g_np * y_np, axis=-1, keepdims=True)
        expected = y_np * (g_np - dot)

        np.testing.assert_allclose(as_np(dx), expected, rtol=1e-6, atol=1e-7)

        # Sum of gradients along softmax axis should be ~0 (property of softmax)
        row_sums = np.sum(as_np(dx), axis=-1)
        np.testing.assert_allclose(
            row_sums, np.zeros_like(row_sums), rtol=1e-6, atol=1e-6
        )

    def test_backward_matches_finite_difference(self):
        """
        Finite-difference check of JVP by constructing a scalar objective:
          L = sum(softmax(x) * w)
        Then dL/dx = softmax_backward(grad_out=w)
        """
        x_np = np.array([[0.1, -0.2, 0.3], [1.5, 0.0, -1.0]], dtype=np.float32)
        w_np = np.array([[0.3, -0.4, 0.1], [0.05, 0.2, -0.1]], dtype=np.float32)

        x = tensor_from_np(x_np, requires_grad=True)
        w = tensor_from_np(w_np, requires_grad=False)

        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        y = SoftmaxFn.forward(ctx, x, axis=-1)

        # Analytical gradient via backward JVP
        (dx,) = SoftmaxFn.backward(ctx, w)

        def scalar_objective(x_arr: np.ndarray) -> float:
            # Use numpy forward for objective; keeps test independent of Tensor ops
            y_arr = stable_softmax_np(x_arr, axis=-1)
            return float(np.sum(y_arr * w_np))

        numeric = finite_diff_grad_x(scalar_objective, x_np, eps=1e-4, axis=-1)

        np.testing.assert_allclose(as_np(dx), numeric, rtol=2e-3, atol=2e-3)

    def test_invalid_axis_raises(self):
        x = tensor_from_np(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        with self.assertRaises(ValueError):
            _ = SoftmaxFn.forward(ctx, x, axis=5)


if __name__ == "__main__":
    unittest.main()
