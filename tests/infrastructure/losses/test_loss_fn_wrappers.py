import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.losses._wrappers import (
    sse_loss,
    mse_loss,
    bce_loss,
    cce_loss,
)


def _tensor_from_numpy(a: np.ndarray, *, requires_grad: bool) -> Tensor:
    a = np.asarray(a, dtype=np.float32)
    t = Tensor(shape=a.shape, device=Device("cpu"), requires_grad=requires_grad)
    t.copy_from_numpy(a)
    return t


def _as_np(t: Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(), dtype=np.float32)


def _backward(loss_scalar: Tensor) -> None:

    if not hasattr(loss_scalar, "backward"):
        raise AssertionError("Tensor must expose backward() for this test.")
    loss_scalar.backward()


class TestLossFnWrappersSSE(TestCase):
    def test_sse_forward_matches_numpy(self) -> None:
        loss_fn = sse_loss()

        pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targ_np = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=False)
        targ = _tensor_from_numpy(targ_np, requires_grad=False)

        out = loss_fn(pred, targ)
        got = float(out.item())
        exp = float(np.sum((pred_np - targ_np) ** 2))
        self.assertAlmostEqual(got, exp, places=6)

    def test_sse_backward_gradients(self) -> None:
        loss_fn = sse_loss()

        pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targ_np = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=True)
        targ = _tensor_from_numpy(targ_np, requires_grad=True)

        out = loss_fn(pred, targ)
        _backward(out)

        exp_grad_pred = 2.0 * (pred_np - targ_np)
        exp_grad_targ = -exp_grad_pred

        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(targ.grad)

        np.testing.assert_allclose(
            _as_np(pred.grad), exp_grad_pred, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            _as_np(targ.grad), exp_grad_targ, rtol=1e-5, atol=1e-6
        )


class TestLossFnWrappersMSE(TestCase):
    def test_mse_forward_matches_numpy(self) -> None:
        loss_fn = mse_loss()

        pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targ_np = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=False)
        targ = _tensor_from_numpy(targ_np, requires_grad=False)

        out = loss_fn(pred, targ)
        got = float(out.item())
        exp = float(np.mean((pred_np - targ_np) ** 2))
        self.assertAlmostEqual(got, exp, places=6)

    def test_mse_backward_gradients(self) -> None:
        loss_fn = mse_loss()

        pred_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targ_np = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=True)
        targ = _tensor_from_numpy(targ_np, requires_grad=True)

        out = loss_fn(pred, targ)
        _backward(out)

        n = pred_np.size
        exp_grad_pred = (2.0 / n) * (pred_np - targ_np)
        exp_grad_targ = -exp_grad_pred

        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(targ.grad)

        np.testing.assert_allclose(
            _as_np(pred.grad), exp_grad_pred, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            _as_np(targ.grad), exp_grad_targ, rtol=1e-5, atol=1e-6
        )


class TestLossFnWrappersBCE(TestCase):
    def test_bce_forward_matches_numpy_with_clamp(self) -> None:
        loss_fn = bce_loss()

        pred_np = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        targ_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=False)
        targ = _tensor_from_numpy(targ_np, requires_grad=False)

        out = loss_fn(pred, targ)
        got = float(out.item())

        eps = 1e-7
        p = np.clip(pred_np, eps, 1.0 - eps)
        exp = float(np.mean(-(targ_np * np.log(p) + (1.0 - targ_np) * np.log(1.0 - p))))
        self.assertAlmostEqual(got, exp, places=6)

    def test_bce_backward_gradient_pred_only(self) -> None:
        loss_fn = bce_loss()

        pred_np = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        targ_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        pred = _tensor_from_numpy(pred_np, requires_grad=True)
        targ = _tensor_from_numpy(targ_np, requires_grad=True)

        out = loss_fn(pred, targ)
        _backward(out)

        eps = 1e-7
        p = np.clip(pred_np, eps, 1.0 - eps)
        n = pred_np.size
        exp_grad_pred = (p - targ_np) / (p * (1.0 - p)) / n

        self.assertIsNotNone(pred.grad)
        np.testing.assert_allclose(
            _as_np(pred.grad), exp_grad_pred, rtol=1e-5, atol=1e-6
        )

        self.assertTrue(getattr(targ, "grad", None) is None)


class TestLossFnWrappersCCE(TestCase):
    def test_cce_forward_matches_numpy_with_clamp(self) -> None:
        loss_fn = cce_loss()

        pred_np = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float32,
        )
        targ_np = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        pred = _tensor_from_numpy(pred_np, requires_grad=False)
        targ = _tensor_from_numpy(targ_np, requires_grad=False)

        out = loss_fn(pred, targ)
        got = float(out.item())

        eps = 1e-7
        p = np.clip(pred_np, eps, 1.0 - eps)

        exp = float(-np.sum(targ_np * np.log(p)) / pred_np.shape[0])
        self.assertAlmostEqual(got, exp, places=6)

    def test_cce_backward_gradient_pred_only(self) -> None:
        loss_fn = cce_loss()

        pred_np = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float32,
        )
        targ_np = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        pred = _tensor_from_numpy(pred_np, requires_grad=True)
        targ = _tensor_from_numpy(targ_np, requires_grad=True)

        out = loss_fn(pred, targ)
        _backward(out)

        eps = 1e-7
        p = np.clip(pred_np, eps, 1.0 - eps)
        n_batch = pred_np.shape[0]

        exp_grad_pred = -(targ_np / p) / n_batch

        self.assertIsNotNone(pred.grad)
        np.testing.assert_allclose(
            _as_np(pred.grad), exp_grad_pred, rtol=1e-5, atol=1e-6
        )

        self.assertTrue(getattr(targ, "grad", None) is None)


if __name__ == "__main__":
    unittest.main()
