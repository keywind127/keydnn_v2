import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure._activations import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Softmax,
    Tanh,
)


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    """
    Helper to create a Tensor from NumPy for tests.
    """
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dWithActivations(unittest.TestCase):
    """
    Validate that Conv2d integrates correctly with activation layers in both
    forward and backward propagation.

    This test suite verifies that a minimal graph of the form:

        x -> Conv2d -> Activation -> sum -> backward

    can execute without errors and produces finite gradients for:
    - input tensor `x`
    - convolution parameters (`weight`, optional `bias`)
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def _run_conv2d_activation_check(
        self,
        activation,
        *,
        expect_range_0_1: bool = False,
        expect_range_minus1_1: bool = False,
    ) -> None:
        """
        Run a single Conv2d -> activation -> sum -> backward check.

        Parameters
        ----------
        activation : Module
            Activation module instance (e.g., Sigmoid(), ReLU()).
        expect_range_0_1 : bool, optional
            If True, asserts output values are within [0, 1].
        expect_range_minus1_1 : bool, optional
            If True, asserts output values are within [-1, 1].
        """
        # ---- input ----
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        # ---- layers ----
        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )

        # ---- forward ----
        y = conv.forward(x)
        z = activation.forward(y)

        # shape sanity check (Conv2d keeps spatial dims with padding=1, k=3, stride=1)
        self.assertEqual(z.shape, (2, 4, 8, 8))

        z_np = z.to_numpy()
        self.assertTrue(
            np.all(np.isfinite(z_np)), "Activation output contains non-finite values"
        )

        if expect_range_0_1:
            self.assertTrue(np.all(z_np >= 0.0))
            self.assertTrue(np.all(z_np <= 1.0))

        if expect_range_minus1_1:
            self.assertTrue(np.all(z_np >= -1.0))
            self.assertTrue(np.all(z_np <= 1.0))

        # ---- backward ----
        loss = z.sum()
        loss.backward()

        # ---- gradient checks ----
        self.assertIsNotNone(x.grad, "x.grad should not be None")
        self.assertIsNotNone(conv.weight.grad, "conv.weight.grad should not be None")

        if conv.bias is not None:
            self.assertIsNotNone(conv.bias.grad, "conv.bias.grad should not be None")

        # shape consistency
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        if conv.bias is not None:
            self.assertEqual(conv.bias.grad.shape, conv.bias.shape)

        # gradients should be finite
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))
        self.assertTrue(np.all(np.isfinite(conv.weight.grad.to_numpy())))
        if conv.bias is not None:
            self.assertTrue(np.all(np.isfinite(conv.bias.grad.to_numpy())))

    def test_conv2d_sigmoid_forward_and_backward(self):
        self._run_conv2d_activation_check(Sigmoid(), expect_range_0_1=True)

    def test_conv2d_relu_forward_and_backward(self):
        self._run_conv2d_activation_check(ReLU())

    def test_conv2d_leakyrelu_forward_and_backward(self):
        self._run_conv2d_activation_check(LeakyReLU())

    def test_conv2d_tanh_forward_and_backward(self):
        self._run_conv2d_activation_check(Tanh(), expect_range_minus1_1=True)

    def test_conv2d_softmax_forward_and_backward(self):
        """
        Softmax sanity check:
        - ensure forward/backward runs
        - output is finite
        - optionally, check normalization along an inferred axis

        Note: KeyDNN's Softmax implementation may define the axis internally
        (commonly the last dimension). We avoid asserting exact normalization
        behavior here to keep the test stable across implementations. This
        test primarily ensures compatibility with Conv2d output tensors.
        """
        self._run_conv2d_activation_check(Softmax())


if __name__ == "__main__":
    unittest.main()
