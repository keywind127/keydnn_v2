import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure._linear import Linear
from src.keydnn.infrastructure.layers._dropout import Dropout
from src.keydnn.infrastructure.layers._batchnorm import BatchNorm1d


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    """
    Supports:
      - Parameter is Tensor-like (has to_numpy/copy_from_numpy/grad)
      - Parameter wraps Tensor in `.data` or `.tensor`
    """
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


class TestBatchNormChainingLinear(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_linear_then_batchnorm_forward_and_backward_runs(self):
        """
        x -> Linear -> BatchNorm1d -> loss -> backward
        """
        np.random.seed(0)
        N, Din, Dout = 4, 5, 6

        lin = Linear(Din, Dout, bias=True, device=self.device)
        bn = BatchNorm1d(num_features=Dout, device=self.device, affine=True)
        bn.training = True

        x = _tensor_from_numpy(np.random.randn(N, Din), self.device, requires_grad=True)

        y = bn(lin(x))
        self.assertEqual(y.shape, (N, Dout))

        loss = y.sum()
        loss.backward()

        # Grad should flow to x
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Grad should flow to Linear params
        any_lin_grad = False
        for p in lin.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                any_lin_grad = True
                break
        self.assertTrue(any_lin_grad, "Expected at least one Linear parameter gradient")

        # Grad should flow to BN affine params
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)
        self.assertEqual(bn.gamma.grad.shape, bn.gamma.shape)
        self.assertEqual(bn.beta.grad.shape, bn.beta.shape)

    def test_batchnorm_then_linear_forward_and_backward_runs(self):
        """
        x -> BatchNorm1d -> Linear -> loss -> backward
        """
        np.random.seed(1)
        N, Din, Dout = 3, 7, 4

        bn = BatchNorm1d(num_features=Din, device=self.device, affine=True)
        bn.training = True
        lin = Linear(Din, Dout, bias=False, device=self.device)

        x = _tensor_from_numpy(np.random.randn(N, Din), self.device, requires_grad=True)

        y = lin(bn(x))
        self.assertEqual(y.shape, (N, Dout))

        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # BN affine grads
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

        # Linear grads
        for p in lin.parameters():
            pt = _unwrap_param_tensor(p)
            self.assertIsNotNone(pt.grad)

    def test_eval_mode_chain_is_deterministic(self):
        """
        Train once to update running stats, then eval:
          x -> Linear -> BN(eval) should be deterministic for same x.
        """
        np.random.seed(2)
        N, Din, Dout = 8, 3, 5

        lin = Linear(Din, Dout, bias=True, device=self.device)
        bn = BatchNorm1d(num_features=Dout, device=self.device, affine=False)

        x_train = _tensor_from_numpy(
            np.random.randn(N, Din), self.device, requires_grad=False
        )

        # Train pass updates running stats
        bn.training = True
        _ = bn(lin(x_train))

        # Eval is deterministic
        bn.training = False
        x_eval = _tensor_from_numpy(
            np.random.randn(N, Din), self.device, requires_grad=False
        )

        y1 = bn(lin(x_eval)).to_numpy()
        y2 = bn(lin(x_eval)).to_numpy()

        np.testing.assert_allclose(y1, y2, rtol=0, atol=0)

    def test_backward_runs_with_affine_false(self):
        """
        Ensure BN(affine=False) still supports chain backward.
        """
        np.random.seed(3)
        N, Din, Dout = 5, 4, 4

        lin = Linear(Din, Dout, bias=False, device=self.device)
        bn = BatchNorm1d(num_features=Dout, device=self.device, affine=False)
        bn.training = True

        x = _tensor_from_numpy(np.random.randn(N, Din), self.device, requires_grad=True)

        out = bn(lin(x))
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


class TestLinearDropoutBatchNormChaining(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_linear_then_dropout_then_batchnorm_forward_backward_runs(self):
        """
        x -> Linear -> Dropout(train) -> BatchNorm1d(train) -> loss -> backward
        """
        np.random.seed(10)
        N, Din, H = 6, 8, 5

        lin = Linear(Din, H, bias=True, device=self.device)
        d = Dropout(p=0.3)
        d.training = True

        bn = BatchNorm1d(num_features=H, device=self.device, affine=True)
        bn.training = True

        x = _tensor_from_numpy(np.random.randn(N, Din), self.device, requires_grad=True)

        y = bn(d(lin(x)))
        self.assertEqual(y.shape, (N, H))

        y.sum().backward()

        # input grad
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # linear grads
        any_lin_grad = False
        for p in lin.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                any_lin_grad = True
                break
        self.assertTrue(any_lin_grad, "Expected at least one Linear parameter gradient")

        # batchnorm affine grads
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

    def test_linear_dropout_off_batchnorm_eval_is_deterministic(self):
        """
        With Dropout eval (identity) and BN eval (running stats),
        the chain should be deterministic.
        """
        np.random.seed(11)
        N, Din, H = 10, 4, 4

        lin = Linear(Din, H, bias=True, device=self.device)
        d = Dropout(p=0.5)
        bn = BatchNorm1d(num_features=H, device=self.device, affine=False)

        # Update BN running stats using train mode once
        bn.training = True
        d.training = True
        x_train = _tensor_from_numpy(
            np.random.randn(N, Din), self.device, requires_grad=False
        )
        _ = bn(d(lin(x_train)))

        # Eval mode: dropout identity + BN uses running stats
        bn.training = False
        d.training = False

        x_eval = _tensor_from_numpy(
            np.random.randn(N, Din), self.device, requires_grad=False
        )
        y1 = bn(d(lin(x_eval))).to_numpy()
        y2 = bn(d(lin(x_eval))).to_numpy()

        np.testing.assert_allclose(y1, y2, rtol=0, atol=0)

    def test_dropout_p_zero_noop_in_chain(self):
        """
        Dropout(p=0) should be a no-op even in training.
        """
        np.random.seed(12)
        N, Din, H = 5, 3, 6

        lin = Linear(Din, H, bias=False, device=self.device)
        d = Dropout(p=0.0)
        d.training = True

        bn = BatchNorm1d(num_features=H, device=self.device, affine=False)
        bn.training = True

        x = _tensor_from_numpy(
            np.random.randn(N, Din), self.device, requires_grad=False
        )

        y_no_dropout = bn(lin(x)).to_numpy()
        y_with_dropout = bn(d(lin(x))).to_numpy()

        # Should be identical since dropout p=0 is identity
        np.testing.assert_allclose(y_no_dropout, y_with_dropout, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
