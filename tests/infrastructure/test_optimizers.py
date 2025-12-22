import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._optimizers import SGD, Adam


def _cpu() -> Device:
    return Device("cpu")


def param_from_np(arr: np.ndarray) -> Parameter:
    arr = np.asarray(arr, dtype=np.float32)
    p = Parameter(shape=arr.shape, device=_cpu(), requires_grad=True)
    p.copy_from_numpy(arr)
    return p


def tensor_from_np(arr: np.ndarray) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=_cpu(), requires_grad=False)
    t.copy_from_numpy(arr)
    return t


class TestSGD(unittest.TestCase):
    def test_step_updates_parameter(self):
        p = param_from_np(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        p._grad = tensor_from_np(
            np.array([0.1, -0.2, 0.3], dtype=np.float32)
        )  # if you must avoid this, see note below

        opt = SGD([p], lr=0.5)
        opt.step()

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32) - 0.5 * np.array(
            [0.1, -0.2, 0.3], dtype=np.float32
        )
        np.testing.assert_allclose(p.to_numpy(), expected, rtol=1e-6, atol=1e-7)

    def test_step_skips_none_grad(self):
        p = param_from_np(np.array([1.0, 2.0], dtype=np.float32))
        p.zero_grad()

        opt = SGD([p], lr=0.1)
        before = p.to_numpy().copy()
        opt.step()
        np.testing.assert_allclose(p.to_numpy(), before, rtol=1e-6, atol=1e-7)

    def test_zero_grad_clears_grad(self):
        p = param_from_np(np.array([1.0], dtype=np.float32))
        p._grad = tensor_from_np(np.array([2.0], dtype=np.float32))

        opt = SGD([p], lr=0.1)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_weight_decay_applied(self):
        p0 = np.array([1.0, -2.0], dtype=np.float32)
        g0 = np.array([0.5, 0.25], dtype=np.float32)

        p = param_from_np(p0)
        p._grad = tensor_from_np(g0)

        lr = 0.1
        wd = 0.01
        opt = SGD([p], lr=lr, weight_decay=wd)
        opt.step()

        # classical L2: p <- p - lr * (g + wd * p)
        expected = p0 - lr * (g0 + wd * p0)
        np.testing.assert_allclose(p.to_numpy(), expected, rtol=1e-6, atol=1e-7)

    def test_invalid_hyperparams_raise(self):
        p = param_from_np(np.array([1.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            _ = SGD([p], lr=0.0)
        with self.assertRaises(ValueError):
            _ = SGD([p], lr=0.1, weight_decay=-1.0)


class TestAdam(unittest.TestCase):
    def test_step_matches_reference_first_step(self):
        """
        First Adam step reference check (bias correction included).
        """
        p0 = np.array([1.0, -2.0], dtype=np.float32)
        g0 = np.array([0.1, -0.2], dtype=np.float32)

        p = param_from_np(p0)

        # Prefer p._set_grad(...) if available; fallback to direct assignment
        if hasattr(p, "_set_grad"):
            p._set_grad(tensor_from_np(g0))
        else:
            p._grad = tensor_from_np(g0)  # temporary until autograd exists

        lr = 1e-2
        b1, b2 = 0.9, 0.999
        eps = 1e-8

        opt = Adam([p], lr=lr, betas=(b1, b2), eps=eps)
        opt.step()

        # Reference computation for t=1:
        m = (1 - b1) * g0
        v = (1 - b2) * (g0 * g0)
        m_hat = m / (1 - b1)
        v_hat = v / (1 - b2)
        expected = p0 - lr * (m_hat / (np.sqrt(v_hat) + eps))

        np.testing.assert_allclose(p.to_numpy(), expected, rtol=1e-6, atol=1e-7)

    def test_step_skips_none_grad(self):
        p = param_from_np(np.array([1.0, 2.0], dtype=np.float32))
        p.zero_grad()

        opt = Adam([p], lr=1e-3)
        before = p.to_numpy().copy()
        opt.step()
        np.testing.assert_allclose(p.to_numpy(), before, rtol=1e-6, atol=1e-7)

    def test_zero_grad_clears_grad(self):
        p = param_from_np(np.array([1.0], dtype=np.float32))
        if hasattr(p, "_set_grad"):
            p._set_grad(tensor_from_np(np.array([2.0], dtype=np.float32)))
        else:
            p._grad = tensor_from_np(np.array([2.0], dtype=np.float32))  # temporary

        opt = Adam([p], lr=1e-3)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_weight_decay_affects_update(self):
        """
        Classical L2 weight decay: g <- g + wd * p
        """
        p0 = np.array([1.0, -2.0], dtype=np.float32)
        g0 = np.array([0.1, -0.2], dtype=np.float32)
        wd = 0.01

        p = param_from_np(p0)
        if hasattr(p, "_set_grad"):
            p._set_grad(tensor_from_np(g0))
        else:
            p._grad = tensor_from_np(g0)  # temporary

        lr = 1e-2
        b1, b2 = 0.9, 0.999
        eps = 1e-8

        opt = Adam([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
        opt.step()

        g_eff = g0 + wd * p0
        m = (1 - b1) * g_eff
        v = (1 - b2) * (g_eff * g_eff)
        m_hat = m / (1 - b1)
        v_hat = v / (1 - b2)
        expected = p0 - lr * (m_hat / (np.sqrt(v_hat) + eps))

        np.testing.assert_allclose(p.to_numpy(), expected, rtol=1e-6, atol=1e-7)

    def test_invalid_hyperparams_raise(self):
        p = param_from_np(np.array([1.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            _ = Adam([p], lr=0.0)
        with self.assertRaises(ValueError):
            _ = Adam([p], betas=(1.0, 0.999))
        with self.assertRaises(ValueError):
            _ = Adam([p], betas=(0.9, 0.0))
        with self.assertRaises(ValueError):
            _ = Adam([p], eps=0.0)
        with self.assertRaises(ValueError):
            _ = Adam([p], weight_decay=-1.0)


if __name__ == "__main__":
    unittest.main()
