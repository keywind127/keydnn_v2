import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.layers._batchnorm import BatchNorm2d

try:
    from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
except Exception:  # pragma: no cover
    Conv2d = None  # type: ignore

try:
    from src.keydnn.infrastructure.layers._dropout import Dropout
except Exception:  # pragma: no cover
    Dropout = None  # type: ignore


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


def _make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    device: Device,
    *,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
):
    """
    Best-effort constructor for Conv2d to tolerate signature differences.

    Expected common signatures:
      Conv2d(in_ch, out_ch, kernel_size, stride=..., padding=..., bias=..., device=...)
      Conv2d(in_ch, out_ch, kernel_size, stride=..., padding=..., bias=...)
    """
    if Conv2d is None:
        raise unittest.SkipTest(
            "Conv2d import failed; skip chain tests involving Conv2d."
        )

    try:
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            device=device,
        )
    except TypeError:
        # fallback (no device kw)
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )


class TestBatchNorm2dInfrastructure(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_get_config_and_from_config_roundtrip(self):
        bn1 = BatchNorm2d(
            num_features=8,
            device=self.device,
            eps=1e-5,
            momentum=0.2,
            affine=True,
        )
        cfg = bn1.get_config()

        self.assertEqual(cfg["num_features"], 8)
        self.assertAlmostEqual(cfg["eps"], 1e-5)
        self.assertAlmostEqual(cfg["momentum"], 0.2)
        self.assertEqual(cfg["affine"], True)
        self.assertIn("device", cfg)

        bn2 = BatchNorm2d.from_config(cfg)
        self.assertIsInstance(bn2, BatchNorm2d)
        self.assertEqual(bn2.num_features, 8)
        self.assertAlmostEqual(bn2.eps, 1e-5)
        self.assertAlmostEqual(bn2.momentum, 0.2)
        self.assertEqual(bn2.affine, True)

    def test_parameters_exist_only_when_affine_true(self):
        bn_aff = BatchNorm2d(num_features=4, device=self.device, affine=True)
        self.assertEqual(len(list(bn_aff.parameters())), 2)  # gamma + beta

        bn_no = BatchNorm2d(num_features=4, device=self.device, affine=False)
        self.assertEqual(len(list(bn_no.parameters())), 0)


class TestBatchNorm2dForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_rejects_non_4d_input(self):
        bn = BatchNorm2d(num_features=3, device=self.device)

        with self.assertRaises(ValueError):
            bn.forward(Tensor((3,), self.device))

        with self.assertRaises(ValueError):
            bn.forward(Tensor((2, 3), self.device))

        with self.assertRaises(ValueError):
            bn.forward(Tensor((2, 3, 4), self.device))

    def test_forward_rejects_feature_mismatch(self):
        bn = BatchNorm2d(num_features=3, device=self.device)
        x = Tensor((2, 5, 4, 4), self.device)  # C=5 mismatch
        with self.assertRaises(ValueError):
            bn.forward(x)

    def test_training_forward_outputs_zero_mean_unit_var_affine_false(self):
        """
        Training mode: output per-channel should be normalized (mean~0, var~1)
        when affine=False, computed over axes (N,H,W).
        """
        np.random.seed(0)
        N, C, H, W = 16, 6, 5, 5
        x_np = (np.random.randn(N, C, H, W).astype(np.float32) * 3.0) + 10.0
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        bn = BatchNorm2d(
            num_features=C, device=self.device, eps=1e-5, momentum=0.1, affine=False
        )
        bn.training = True

        y = bn.forward(x).to_numpy()

        # mean/var per channel over (N,H,W)
        mean = y.mean(axis=(0, 2, 3))
        var = y.var(axis=(0, 2, 3))

        np.testing.assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=1e-3)
        np.testing.assert_allclose(var, np.ones_like(var), rtol=0, atol=3e-3)

    def test_running_stats_update_in_training(self):
        np.random.seed(1)
        N, C, H, W = 8, 4, 4, 4
        x_np = np.random.randn(N, C, H, W).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        bn = BatchNorm2d(num_features=C, device=self.device, momentum=0.5, affine=True)
        bn.training = True

        rm0 = bn.running_mean.to_numpy().copy()
        rv0 = bn.running_var.to_numpy().copy()

        _ = bn.forward(x)

        rm1 = bn.running_mean.to_numpy()
        rv1 = bn.running_var.to_numpy()

        self.assertFalse(np.allclose(rm0, rm1))
        self.assertFalse(np.allclose(rv0, rv1))

    def test_eval_uses_running_stats_deterministically(self):
        """
        After one training forward updates running stats, eval forward should be deterministic.
        """
        np.random.seed(2)
        N, C, H, W = 6, 5, 3, 3

        bn = BatchNorm2d(num_features=C, device=self.device, momentum=0.9, affine=False)

        bn.training = True
        x_train = _tensor_from_numpy(
            np.random.randn(N, C, H, W).astype(np.float32),
            self.device,
            requires_grad=False,
        )
        _ = bn.forward(x_train)

        bn.training = False
        x_eval = _tensor_from_numpy(
            np.random.randn(N, C, H, W).astype(np.float32),
            self.device,
            requires_grad=False,
        )

        y1 = bn.forward(x_eval).to_numpy()
        y2 = bn.forward(x_eval).to_numpy()

        np.testing.assert_allclose(y1, y2, rtol=0, atol=0)


class TestBatchNorm2dBackward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_backward_populates_grad_for_x_and_affine_params(self):
        """
        x.requires_grad True and affine=True should yield grads for:
        - x
        - gamma, beta
        """
        np.random.seed(3)
        N, C, H, W = 4, 7, 3, 3
        x = _tensor_from_numpy(
            np.random.randn(N, C, H, W).astype(np.float32),
            self.device,
            requires_grad=True,
        )

        bn = BatchNorm2d(num_features=C, device=self.device, affine=True)
        bn.training = True

        for p in bn.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        out = bn.forward(x)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)
        self.assertEqual(bn.gamma.grad.shape, bn.gamma.shape)
        self.assertEqual(bn.beta.grad.shape, bn.beta.shape)

    def test_backward_runs_when_affine_false(self):
        """
        affine=False should still backprop into x.
        """
        np.random.seed(4)
        N, C, H, W = 3, 4, 4, 2
        x = _tensor_from_numpy(
            np.random.randn(N, C, H, W).astype(np.float32),
            self.device,
            requires_grad=True,
        )

        bn = BatchNorm2d(num_features=C, device=self.device, affine=False)
        bn.training = True

        y = bn.forward(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


# ---------------------------------------------------------------------
# Chained tests
# ---------------------------------------------------------------------
class TestBatchNorm2dChaining(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_conv2d_then_batchnorm2d_forward_and_backward_runs(self):
        """
        x -> Conv2d -> BatchNorm2d -> loss -> backward
        """
        if Conv2d is None:
            self.skipTest("Conv2d import failed; skip chain test.")

        np.random.seed(10)
        N, Cin, H, W = 2, 3, 8, 8
        Cout = 4

        conv = _make_conv2d(
            Cin, Cout, kernel_size=3, device=self.device, stride=1, padding=1, bias=True
        )
        bn = BatchNorm2d(num_features=Cout, device=self.device, affine=True)
        bn.training = True

        x = _tensor_from_numpy(
            np.random.randn(N, Cin, H, W), self.device, requires_grad=True
        )

        y = bn(conv(x))
        self.assertEqual(y.shape, (N, Cout, H, W))

        y.sum().backward()

        # x grad
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # BN affine grads
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

        # Conv params grads (at least one)
        any_conv_grad = False
        for p in conv.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                any_conv_grad = True
                break
        self.assertTrue(
            any_conv_grad, "Expected at least one Conv2d parameter gradient"
        )

    def test_batchnorm2d_then_conv2d_forward_and_backward_runs(self):
        """
        x -> BatchNorm2d -> Conv2d -> loss -> backward
        """
        if Conv2d is None:
            self.skipTest("Conv2d import failed; skip chain test.")

        np.random.seed(11)
        N, Cin, H, W = 2, 3, 7, 7
        Cout = 5

        bn = BatchNorm2d(num_features=Cin, device=self.device, affine=False)
        bn.training = True

        conv = _make_conv2d(
            Cin,
            Cout,
            kernel_size=3,
            device=self.device,
            stride=1,
            padding=1,
            bias=False,
        )

        x = _tensor_from_numpy(
            np.random.randn(N, Cin, H, W), self.device, requires_grad=True
        )

        y = conv(bn(x))
        self.assertEqual(y.shape, (N, Cout, H, W))

        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Conv param grads should exist
        for p in conv.parameters():
            pt = _unwrap_param_tensor(p)
            self.assertIsNotNone(pt.grad)

    def test_conv2d_then_batchnorm2d_then_dropout_runs(self):
        """
        x -> Conv2d -> BatchNorm2d -> Dropout -> loss -> backward
        (Dropout works on arbitrary shapes; BN2d normalizes channel-wise)
        """
        if Conv2d is None:
            self.skipTest("Conv2d import failed; skip chain test.")
        if Dropout is None:
            self.skipTest("Dropout import failed; skip chain test.")

        np.random.seed(12)
        N, Cin, H, W = 2, 2, 6, 6
        Cout = 3

        conv = _make_conv2d(
            Cin, Cout, kernel_size=3, device=self.device, stride=1, padding=1, bias=True
        )
        bn = BatchNorm2d(num_features=Cout, device=self.device, affine=True)
        bn.training = True

        d = Dropout(p=0.25)
        d.training = True

        x = _tensor_from_numpy(
            np.random.randn(N, Cin, H, W), self.device, requires_grad=True
        )

        y = d(bn(conv(x)))
        self.assertEqual(y.shape, (N, Cout, H, W))

        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

    def test_eval_chain_is_deterministic_when_dropout_off(self):
        """
        Train once to update BN running stats, then eval:
        - BN eval should be deterministic
        - Dropout eval is identity -> chain deterministic
        """
        if Conv2d is None:
            self.skipTest("Conv2d import failed; skip chain test.")
        if Dropout is None:
            self.skipTest("Dropout import failed; skip chain test.")

        np.random.seed(13)
        N, Cin, H, W = 2, 3, 5, 5
        Cout = 4

        conv = _make_conv2d(
            Cin,
            Cout,
            kernel_size=3,
            device=self.device,
            stride=1,
            padding=1,
            bias=False,
        )
        bn = BatchNorm2d(num_features=Cout, device=self.device, affine=False)
        d = Dropout(p=0.5)

        # Train forward updates running stats
        bn.training = True
        d.training = True
        x_train = _tensor_from_numpy(
            np.random.randn(N, Cin, H, W), self.device, requires_grad=False
        )
        _ = d(bn(conv(x_train)))

        # Eval: deterministic (dropout identity + BN running stats)
        bn.training = False
        d.training = False
        x_eval = _tensor_from_numpy(
            np.random.randn(N, Cin, H, W), self.device, requires_grad=False
        )

        y1 = d(bn(conv(x_eval))).to_numpy()
        y2 = d(bn(conv(x_eval))).to_numpy()
        np.testing.assert_allclose(y1, y2, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
