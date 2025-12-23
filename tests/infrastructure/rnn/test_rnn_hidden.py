from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.recurrent._rnn_module import RNN


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestRNNHiddenGradients(TestCase):
    """
    Hidden-state gradient propagation tests.

    These tests specifically validate that gradients can flow:
      - from h_T back into h0 (if provided and requires_grad=True),
      - from h_seq back into earlier timesteps and ultimately back into x,
      - through long-ish sequences without shape/None errors.
    """

    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_rnn_backward_propagates_grad_to_h0_when_requires_grad_true(self):
        np.random.seed(0)
        T, N, D, H = 5, 2, 3, 4

        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        h0_np = np.random.randn(N, H).astype(np.float32)

        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h0 = _tensor_from_numpy(h0_np, self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x, h0=h0)

        # Use only final hidden state to drive gradients.
        loss = h_T.sum()
        loss.backward()

        self.assertIsNotNone(
            h0.grad, "h0.grad should be populated when h0.requires_grad=True"
        )
        self.assertEqual(h0.grad.shape, h0.shape)

        # Sanity: gradients should be finite.
        g = h0.grad.to_numpy()
        self.assertTrue(np.all(np.isfinite(g)))

    def test_rnn_backward_h0_grad_none_when_requires_grad_false(self):
        np.random.seed(1)
        T, N, D, H = 4, 2, 3, 5

        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        h0 = _tensor_from_numpy(np.random.randn(N, H), self.device, requires_grad=False)

        h_seq, h_T = rnn.forward(x, h0=h0)
        h_T.sum().backward()

        self.assertIsNone(
            h0.grad, "h0.grad should remain None when h0.requires_grad=False"
        )
        self.assertIsNotNone(
            x.grad, "x.grad should be populated when x.requires_grad=True"
        )

    def test_rnn_backward_from_h_seq_sum_populates_x_and_parameter_grads(self):
        np.random.seed(2)
        T, N, D, H = 6, 3, 2, 4

        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        h_seq, h_T = rnn.forward(x, h0=None)

        # Drive gradient from the full sequence output (more demanding than h_T only).
        loss = h_seq.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "x.grad should be populated from h_seq path")
        self.assertEqual(x.grad.shape, x.shape)

        # Parameters should also receive grads (via cell backward across timesteps).
        params = list(rnn.parameters())
        self.assertGreater(len(params), 0)
        for p in params:
            # Parameter inherits Tensor-like interface in your design
            self.assertIsNotNone(
                p.grad, "Each parameter should have grad after backward from h_seq"
            )
            self.assertEqual(p.grad.shape, p.shape)

    def test_rnn_backward_from_single_timestep_of_h_seq(self):
        """
        Ensure Tensor.stack backward can route gradient from a single timestep
        selection h_seq[t] back through the graph.
        """
        np.random.seed(3)
        T, N, D, H = 7, 2, 3, 3

        rnn = RNN(input_size=D, hidden_size=H, bias=True)
        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        h_seq, h_T = rnn.forward(x, h0=None)

        # Pick a single timestep slice. This relies on your Tensor.__getitem__ implementation.
        # The gradient should affect only the relevant timestep contribution *through time*.
        t_pick = 4
        h_pick = h_seq[t_pick]  # shape (N, H)

        loss = h_pick.sum()
        loss.backward()

        self.assertIsNotNone(
            x.grad, "x.grad should exist when backprop from a single timestep output"
        )
        self.assertEqual(x.grad.shape, x.shape)

        # Should still produce finite grads
        gx = x.grad.to_numpy()
        self.assertTrue(np.all(np.isfinite(gx)))

    def test_rnn_backward_longer_sequence_stability_smoke(self):
        """
        Not a numerical quality test (RNNs can explode/vanish),
        but a regression check that BPTT doesn't crash on longer T.
        """
        np.random.seed(4)
        T, N, D, H = 25, 2, 4, 6

        rnn = RNN(input_size=D, hidden_size=H, bias=True)
        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        h_seq, h_T = rnn.forward(x, h0=None)

        # Mild loss: sum of last hidden state
        loss = h_T.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(x.grad.to_numpy())))

    def test_rnn_backward_with_h0_and_x_both_requires_grad(self):
        np.random.seed(5)
        T, N, D, H = 8, 2, 3, 4

        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        h0 = _tensor_from_numpy(np.random.randn(N, H), self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x, h0=h0)
        # Use both outputs to drive gradients
        (h_seq.sum() + h_T.sum()).backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h0.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h0.grad.shape, h0.shape)


if __name__ == "__main__":
    unittest.main()
