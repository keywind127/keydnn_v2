import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.layers._dropout import Dropout
from src.keydnn.infrastructure._tensor import Tensor

from src.keydnn.infrastructure._linear import Linear
from src.keydnn.infrastructure.recurrent._rnn_module import RNNCell
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d


def _tensor_supports_numpy_load() -> bool:
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        pass

    t = Tensor((1, 1), Device("cpu"))
    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        return True
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        return True

    return False


def _make_tensor_from_numpy(arr: np.ndarray, device: Device) -> Tensor:
    arr = arr.astype(np.float32, copy=False)

    try:
        return Tensor(data=arr, device=device)
    except TypeError:
        t = Tensor(arr.shape, device)
        if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
            t.from_numpy(arr)
            return t
        if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
            t.copy_from_numpy(arr)
            return t

        raise AssertionError(
            "Tensor cannot be loaded with NumPy data via public APIs. "
            "Implement Tensor(data=...) OR Tensor.from_numpy()/copy_from_numpy()."
        )


class TestDropoutChaining(TestCase):
    def setUp(self):
        self.device = Device("cpu")

    def test_dropout_then_linear_forward_shape_and_backward_runs(self):
        """
        x -> Dropout -> Linear -> sum -> backward

        Verifies:
        - output shape is correct
        - backward completes
        - x.grad exists and has expected shape
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(0)

        x_np = np.random.randn(4, 3).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device)
        x.requires_grad = True

        d = Dropout(p=0.5)
        d.training = True

        lin = Linear(3, 5, bias=True, device=self.device)
        # Make linear deterministic and easy: W=1, b=0
        lin.weight.fill(1.0)
        lin.bias.fill(0.0)

        y = lin(d(x))
        self.assertEqual(y.shape, (4, 5))

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_linear_then_dropout_eval_is_identity(self):
        """
        In eval mode, dropout should not change the Linear output.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(1)

        x_np = np.random.randn(2, 3).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device)

        lin = Linear(3, 4, bias=True, device=self.device)
        lin.weight.fill(0.0)
        lin.bias.fill(2.0)

        d = Dropout(p=0.9)
        d.training = False

        y1 = lin(x)
        y2 = d(lin(x))

        np.testing.assert_allclose(y1.to_numpy(), y2.to_numpy(), rtol=0, atol=0)

    def test_dropout_then_conv2d_forward_and_backward_runs(self):
        """
        x -> Dropout -> Conv2d -> sum -> backward

        Ensures dropout mask application doesn't break conv2d pipeline.
        """
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric tests because Tensor cannot be loaded from NumPy.",
        )

        np.random.seed(3)

        # Typical Conv2d input: (N, C, H, W)
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, self.device)
        x.requires_grad = True

        d = Dropout(p=0.25)
        d.training = True

        # Adjust args to match your Conv2d signature
        conv = Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            device=self.device,
        )

        y = conv(d(x))
        self.assertEqual(y.shape[0], 2)  # batch
        self.assertEqual(y.shape[1], 4)  # out_channels
        self.assertEqual(y.shape[2:], (8, 8))  # padding=1 keeps H,W

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestDropoutChainingRNN(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_dropout_then_rnncell_forward_and_backward_runs(self):
        """
        (x_t -> Dropout) + (h_prev) -> RNNCell -> loss -> backward

        Verifies:
        - forward produces expected shape (N, H)
        - backward runs end-to-end
        - grads exist for x_t and h_prev
        """
        np.random.seed(2)

        N, D, H = 2, 3, 5
        x_np = np.random.randn(N, D).astype(np.float32)
        h_np = np.random.randn(N, H).astype(np.float32)

        x = _tensor_from_numpy(x_np, self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(h_np, self.device, requires_grad=True)

        d = Dropout(p=0.3)
        d.training = True

        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

        out = cell.forward(d(x), h_prev)

        self.assertEqual(out.shape, (N, H))

        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h_prev.grad.shape, h_prev.shape)

    def test_dropout_on_both_inputs_then_rnncell_backward_runs(self):
        """
        (x_t -> Dropout) + (h_prev -> Dropout) -> RNNCell -> loss -> backward

        Ensures dropout can sit anywhere in the recurrent pipeline.
        """
        np.random.seed(7)

        N, D, H = 3, 4, 6
        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        dx = Dropout(p=0.4)
        dh = Dropout(p=0.2)
        dx.training = True
        dh.training = True

        cell = RNNCell(input_size=D, hidden_size=H, bias=False)

        out = cell.forward(dx(x), dh(h_prev))
        self.assertEqual(out.shape, (N, H))

        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h_prev.grad.shape, h_prev.shape)


if __name__ == "__main__":
    unittest.main()
