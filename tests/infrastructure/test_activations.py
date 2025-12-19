import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.infrastructure._activations import Sigmoid, ReLU, LeakyReLU, Tanh
from src.keydnn.infrastructure._function import SoftmaxFn
from src.keydnn.infrastructure._activations import Softmax


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


class _ModuleActivationAsserts:
    def assert_ctx_attached(self, out: Tensor, x: Tensor) -> None:
        ctx = out._get_ctx()
        self.assertIsNotNone(ctx, "Expected Context to be attached to output Tensor.")
        assert ctx is not None
        self.assertEqual(len(ctx.parents), 1)
        self.assertIs(ctx.parents[0], x)
        self.assertTrue(callable(ctx.backward_fn))

    def assert_ctx_not_attached(self, out: Tensor) -> None:
        self.assertIsNone(out._get_ctx(), "Did not expect Context to be attached.")


class TestSigmoidModule(unittest.TestCase, _ModuleActivationAsserts):

    def test_forward_matches_numpy(self) -> None:
        mod = Sigmoid()
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=False)

        y = mod.forward(x)

        expected = 1.0 / (1.0 + np.exp(-x_np))
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        self.assertFalse(y.requires_grad)
        self.assert_ctx_not_attached(y)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        mod = Sigmoid()
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        y = mod.forward(x)

        self.assertTrue(y.requires_grad)
        self.assert_ctx_attached(y, x)

        ctx = y._get_ctx()
        assert ctx is not None

        # SigmoidFn saves output only
        self.assertEqual(len(ctx.saved_tensors), 1)
        saved_out = ctx.saved_tensors[0]
        self.assertEqual(saved_out.shape, y.shape)

        grad_out = ones_like(y)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 1)
        grad_x = grads[0]
        self.assertIsNotNone(grad_x)
        assert grad_x is not None
        self.assertEqual(grad_x.shape, x.shape)

        y_np = y.to_numpy()
        expected_grad = y_np * (1.0 - y_np)
        np.testing.assert_allclose(
            grad_x.to_numpy(), expected_grad, rtol=1e-6, atol=1e-6
        )


class TestReLUModule(unittest.TestCase, _ModuleActivationAsserts):

    def test_forward_matches_numpy(self) -> None:
        mod = ReLU()
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=False)

        y = mod.forward(x)

        expected = np.maximum(0.0, x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=0, atol=0)

        self.assertFalse(y.requires_grad)
        self.assert_ctx_not_attached(y)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        mod = ReLU()
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        y = mod.forward(x)

        self.assertTrue(y.requires_grad)
        self.assert_ctx_attached(y, x)

        ctx = y._get_ctx()
        assert ctx is not None

        # ReLUFn saves mask only
        self.assertEqual(len(ctx.saved_tensors), 1)
        mask = ctx.saved_tensors[0]
        self.assertEqual(mask.shape, x.shape)

        grad_out = ones_like(y)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 1)
        grad_x = grads[0]
        self.assertIsNotNone(grad_x)
        assert grad_x is not None

        expected_grad = (x_np > 0).astype(np.float32)  # x==0 -> 0 by design (x > 0)
        np.testing.assert_allclose(grad_x.to_numpy(), expected_grad, rtol=0, atol=0)


class TestLeakyReLUModule(unittest.TestCase, _ModuleActivationAsserts):

    def test_forward_matches_numpy_default_alpha(self) -> None:
        mod = LeakyReLU()  # alpha=0.01
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=False)

        y = mod.forward(x)

        alpha = 0.01
        expected = np.where(x_np > 0, x_np, alpha * x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        self.assertFalse(y.requires_grad)
        self.assert_ctx_not_attached(y)

    def test_requires_grad_attaches_ctx_and_backward_custom_alpha(self) -> None:
        alpha = 0.1
        mod = LeakyReLU(alpha=alpha)

        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        y = mod.forward(x)

        self.assertTrue(y.requires_grad)
        self.assert_ctx_attached(y, x)

        ctx = y._get_ctx()
        assert ctx is not None

        # LeakyReLUFn saves pos_mask and neg_mask
        self.assertEqual(len(ctx.saved_tensors), 2)
        pos_mask, neg_mask = ctx.saved_tensors
        self.assertEqual(pos_mask.shape, x.shape)
        self.assertEqual(neg_mask.shape, x.shape)

        self.assertIn("alpha", ctx.saved_meta)
        self.assertAlmostEqual(float(ctx.saved_meta["alpha"]), alpha)

        grad_out = ones_like(y)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 1)
        grad_x = grads[0]
        self.assertIsNotNone(grad_x)
        assert grad_x is not None

        expected_grad = np.where(x_np > 0, 1.0, alpha).astype(
            np.float32
        )  # x==0 -> alpha
        np.testing.assert_allclose(grad_x.to_numpy(), expected_grad, rtol=0, atol=0)


class TestTanhModule(unittest.TestCase, _ModuleActivationAsserts):

    def test_forward_matches_numpy(self) -> None:
        mod = Tanh()
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=False)

        y = mod.forward(x)

        expected = np.tanh(x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        self.assertFalse(y.requires_grad)
        self.assert_ctx_not_attached(y)

    def test_requires_grad_attaches_ctx_and_backward(self) -> None:
        mod = Tanh()
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        y = mod.forward(x)

        self.assertTrue(y.requires_grad)
        self.assert_ctx_attached(y, x)

        ctx = y._get_ctx()
        assert ctx is not None

        # TanhFn saves output only
        self.assertEqual(len(ctx.saved_tensors), 1)
        saved_out = ctx.saved_tensors[0]
        self.assertEqual(saved_out.shape, y.shape)

        grad_out = ones_like(y)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 1)
        grad_x = grads[0]
        self.assertIsNotNone(grad_x)
        assert grad_x is not None

        expected_grad = 1.0 - np.tanh(x_np) ** 2
        np.testing.assert_allclose(
            grad_x.to_numpy(), expected_grad, rtol=1e-6, atol=1e-6
        )


class TestSoftmaxModule(unittest.TestCase):
    def test_module_forward_matches_function(self):
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = tensor_from_np(x_np, requires_grad=True)

        sm = Softmax(axis=-1)

        # Module forward
        y_mod = sm(x)

        # Function forward (fresh ctx)
        ctx = Context(parents=(x,), backward_fn=lambda _g: ())
        y_fn = SoftmaxFn.forward(ctx, x, axis=-1)

        np.testing.assert_allclose(as_np(y_mod), as_np(y_fn), rtol=1e-6, atol=1e-7)

        # If input requires grad, module output should require grad as well
        self.assertTrue(y_mod.requires_grad)

    def test_module_backward_matches_function_backward(self):
        x_np = np.array([[0.2, -0.1, 0.3]], dtype=np.float32)
        g_np = np.array([[0.5, -0.25, 0.75]], dtype=np.float32)

        x1 = tensor_from_np(x_np, requires_grad=True)
        x2 = tensor_from_np(x_np, requires_grad=True)
        g = tensor_from_np(g_np, requires_grad=False)

        sm = Softmax(axis=-1)

        # Module path: get gradient via attached Context (no Tensor.backward yet)
        y_mod = sm(x1)

        ctx_mod = y_mod._get_ctx()
        self.assertIsNotNone(ctx_mod, "Softmax module output should have an attached Context when requires_grad=True.")

        (grad_mod_x,) = ctx_mod.backward_fn(g)

        # Function path: direct backward
        ctx_fn = Context(parents=(x2,), backward_fn=lambda _g: ())
        _ = SoftmaxFn.forward(ctx_fn, x2, axis=-1)
        (grad_fn_x,) = SoftmaxFn.backward(ctx_fn, g)

        np.testing.assert_allclose(
            grad_mod_x.to_numpy(), grad_fn_x.to_numpy(), rtol=1e-6, atol=1e-7
        )



if __name__ == "__main__":
    unittest.main()
