import unittest
import numpy as np

from keydnn.domain._device import Device
from keydnn.infrastructure._tensor import Tensor
from keydnn.infrastructure._activations import Sigmoid, ReLU, LeakyReLU, Tanh


def make_cpu_tensor(arr: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=Device("cpu"), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def ones_like(t: Tensor) -> Tensor:
    g = Tensor(shape=t.shape, device=t.device, requires_grad=False)
    g.fill(1.0)
    return g


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


if __name__ == "__main__":
    unittest.main()
