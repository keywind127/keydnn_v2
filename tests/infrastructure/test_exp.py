import unittest
import numpy as np

from src.keydnn.infrastructure.tensor._tensor_context import Context
from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._function import ExpFn, exp


def make_cpu_tensor(arr: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=Device("cpu"), requires_grad=requires_grad)
    t.copy_from_numpy(arr)
    return t


def ones_like(t: Tensor) -> Tensor:
    g = Tensor(shape=t.shape, device=t.device, requires_grad=False)
    g.fill(1.0)
    return g


def _supports_tensor_mul() -> bool:
    """
    Returns True if Tensor * Tensor works (operator overloading implemented).
    """
    try:
        a = make_cpu_tensor(np.array([1.0], dtype=np.float32))
        b = make_cpu_tensor(np.array([2.0], dtype=np.float32))
        _ = a * b  # type: ignore[operator]
        return True
    except Exception:
        return False


HAS_TENSOR_MUL = _supports_tensor_mul()


class TestContext(unittest.TestCase):
    def test_save_for_backward_appends(self) -> None:
        x = make_cpu_tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        y = make_cpu_tensor(np.array([[3.0, 4.0]], dtype=np.float32))

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        self.assertEqual(len(ctx.saved_tensors), 0)

        ctx.save_for_backward(x, y)

        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], x)
        self.assertIs(ctx.saved_tensors[1], y)


class TestExpFunction(unittest.TestCase):

    def test_exp_forward_values(self) -> None:
        x_np = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = ExpFn.forward(ctx, x)

        expected = np.exp(x_np)
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

        self.assertEqual(len(ctx.saved_tensors), 1)

    def test_exp_forward_requires_grad_propagation(self) -> None:
        x = make_cpu_tensor(np.array([[0.5]], dtype=np.float32), requires_grad=False)
        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = ExpFn.forward(ctx, x)
        self.assertFalse(y.requires_grad)

        x2 = make_cpu_tensor(np.array([[0.5]], dtype=np.float32), requires_grad=True)
        ctx2 = Context(parents=(x2,), backward_fn=lambda grad_out: ())
        y2 = ExpFn.forward(ctx2, x2)
        self.assertTrue(y2.requires_grad)

    @unittest.skipUnless(HAS_TENSOR_MUL, "Tensor * Tensor not implemented yet")
    def test_exp_backward_gradient_matches_out(self) -> None:
        """
        Checks: d/dx exp(x) = exp(x). With grad_out = 1, grad_x == exp(x).
        """
        x_np = np.array([[0.2, -0.7], [1.3, -2.1]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        y = ExpFn.forward(ctx, x)

        grad_out = ones_like(y)
        grad_x = ExpFn.backward(ctx, grad_out)

        expected = np.exp(x_np)
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(HAS_TENSOR_MUL, "Tensor * Tensor not implemented yet")
    def test_exp_backward_scales_by_grad_out(self) -> None:
        """
        Checks chain rule: grad_x = grad_out * exp(x)
        """
        x_np = np.array([[0.0, 1.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        ctx = Context(parents=(x,), backward_fn=lambda grad_out: ())
        _ = ExpFn.forward(ctx, x)

        grad_out_np = np.array([[2.0, 3.0]], dtype=np.float32)
        grad_out = make_cpu_tensor(grad_out_np, requires_grad=False)

        grad_x = ExpFn.backward(ctx, grad_out)

        expected = grad_out_np * np.exp(x_np)
        np.testing.assert_allclose(grad_x.to_numpy(), expected, rtol=1e-6, atol=1e-6)


class TestExpWrapper(unittest.TestCase):

    def test_exp_raises_on_non_tensor(self) -> None:
        with self.assertRaises(TypeError):
            _ = exp(123)  # type: ignore[arg-type]

    def test_exp_output_matches_numpy(self) -> None:
        x_np = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
        x = make_cpu_tensor(x_np, requires_grad=True)

        y = exp(x)
        expected = np.exp(x_np)

        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_exp_ctx_attached_only_if_requires_grad(self) -> None:
        x1 = make_cpu_tensor(
            np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=False
        )
        y1 = exp(x1)
        self.assertIsNone(y1._get_ctx())

        x2 = make_cpu_tensor(
            np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True
        )
        y2 = exp(x2)
        ctx2 = y2._get_ctx()

        self.assertIsNotNone(ctx2)
        assert ctx2 is not None
        self.assertEqual(len(ctx2.parents), 1)
        self.assertIs(ctx2.parents[0], x2)

    @unittest.skipUnless(HAS_TENSOR_MUL, "Tensor * Tensor not implemented yet")
    def test_exp_wrapper_backward_fn_returns_correct_grad(self) -> None:
        x = make_cpu_tensor(
            np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True
        )
        y = exp(x)
        ctx = y._get_ctx()
        self.assertIsNotNone(ctx)
        assert ctx is not None

        grad_out = ones_like(y)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 1)
        self.assertIsInstance(grads[0], Tensor)

        expected = np.exp(x.to_numpy())
        np.testing.assert_allclose(grads[0].to_numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
