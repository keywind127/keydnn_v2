import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.recurrent._rnn_module import RNN


def _tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool
) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(arr.shape, device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestRNNReturnModes(unittest.TestCase):
    """
    Tests for both:
      - seq2seq: output is the full sequence h_seq of shape (T, N, H)
      - seq2one: output is the final state h_T of shape (N, H)

    These tests assume your backward-compatible RNN supports:
      - return_sequences: bool
      - return_state: bool
      - keras_compat: bool (opt-in Keras-style returns)
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    # -------------------------
    # Keras-compat return modes
    # -------------------------

    def test_seq2seq_return_sequences_true_return_state_false(self):
        """
        Keras-style seq2seq: forward() returns h_seq only.
        """
        T, N, D, H = 5, 3, 4, 6
        rnn = RNN(
            D,
            H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        out = rnn.forward(x)
        self.assertIsInstance(out, Tensor)
        self.assertEqual(out.shape, (T, N, H))

        # gradients should flow from sequence loss
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # weight grads should exist
        for p in rnn.parameters():
            self.assertIsNotNone(
                p.grad, "Parameter grad should be populated under seq2seq loss"
            )

    def test_seq2one_return_sequences_false_return_state_false(self):
        """
        Keras-style seq2one: forward() returns h_T only.
        """
        T, N, D, H = 7, 2, 3, 5
        rnn = RNN(
            D,
            H,
            bias=True,
            return_sequences=False,
            return_state=False,
            keras_compat=True,
        )

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        out = rnn.forward(x)
        self.assertIsInstance(out, Tensor)
        self.assertEqual(out.shape, (N, H))

        # gradients should flow from final-state loss
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn.parameters():
            self.assertIsNotNone(
                p.grad, "Parameter grad should be populated under seq2one loss"
            )

    def test_seq2seq_return_state_true(self):
        """
        Keras-style seq2seq + return_state: forward() returns (h_seq, h_T).
        """
        T, N, D, H = 4, 3, 2, 3
        rnn = RNN(
            D,
            H,
            bias=False,
            return_sequences=True,
            return_state=True,
            keras_compat=True,
        )

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x)
        self.assertEqual(h_seq.shape, (T, N, H))
        self.assertEqual(h_T.shape, (N, H))

        # Combine both outputs in loss; grads should still work
        loss = h_seq.sum() + h_T.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn.parameters():
            self.assertIsNotNone(p.grad)

    def test_seq2one_return_state_true(self):
        """
        Keras-style seq2one + return_state: forward() returns (h_T, h_T) or (out, h_T)
        depending on your implementation. The contract we assert is:
          - it returns a tuple (out, h_T)
          - both are shape (N, H)
        """
        T, N, D, H = 6, 2, 4, 4
        rnn = RNN(
            D,
            H,
            bias=True,
            return_sequences=False,
            return_state=True,
            keras_compat=True,
        )

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        out, h_T = rnn.forward(x)
        self.assertIsInstance(out, Tensor)
        self.assertIsInstance(h_T, Tensor)
        self.assertEqual(out.shape, (N, H))
        self.assertEqual(h_T.shape, (N, H))

        loss = out.sum() + h_T.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn.parameters():
            self.assertIsNotNone(p.grad)

    # -------------------------
    # Legacy mode: backward-compatible defaults
    # -------------------------

    def test_legacy_default_returns_hseq_and_hT(self):
        """
        Legacy mode should continue returning (h_seq, h_T).
        This is your backward-compat contract for existing tests/callers.
        """
        T, N, D, H = 3, 2, 3, 5
        rnn = RNN(D, H, bias=True)  # legacy defaults

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x)
        self.assertEqual(h_seq.shape, (T, N, H))
        self.assertEqual(h_T.shape, (N, H))

        # seq2seq loss
        h_seq.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
