import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._rnn_module import RNN
from src.keydnn.infrastructure._linear import (
    Linear,
)


def _tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool
) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(arr.shape, device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _zero_all_grads(*modules):
    for m in modules:
        for p in m.parameters():
            # Parameter inherits Tensor in your infra; zero_grad exists on both in your design.
            if hasattr(p, "zero_grad"):
                p.zero_grad()


class TestRNNStackingAndHeads(unittest.TestCase):
    """
    Additional integration tests:
      - 2 consecutive RNN layers (seq2seq -> seq2seq)
      - 2 consecutive RNN layers (seq2seq -> seq2one)
      - RNN -> Linear head (seq2one classification-style)
      - RNN -> Linear head (seq2seq per-timestep projection)

    Assumptions:
      - RNN.forward returns either:
          * legacy: (h_seq, h_T)
          * keras_compat: Tensor or tuple depending on flags
      - Linear.forward supports input shape (N, in_features) -> (N, out_features)
      - Tensor supports basic ops used below: sum(), mean(), reshape(), etc.
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_two_rnn_layers_seq2seq_to_seq2seq_shape_and_grad(self):
        """
        Layer1 produces full sequence (T,N,H1) which is fed into Layer2 as input_size=H1.
        Loss depends on the full output sequence to ensure BPTT through both layers.
        """
        T, N, D = 6, 3, 4
        H1, H2 = 5, 7

        rnn1 = RNN(D, H1, bias=True)  # legacy returns (h_seq, h_T)
        rnn2 = RNN(H1, H2, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h1_seq, _h1_T = rnn1.forward(x)
        self.assertEqual(h1_seq.shape, (T, N, H1))

        h2_seq, _h2_T = rnn2.forward(h1_seq)
        self.assertEqual(h2_seq.shape, (T, N, H2))

        # Loss on all timesteps
        loss = h2_seq.sum()
        loss.backward()

        # Grad should reach the original input
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Params in both layers should get grads
        for p in rnn1.parameters():
            self.assertIsNotNone(p.grad, "rnn1 parameter grad should exist")
        for p in rnn2.parameters():
            self.assertIsNotNone(p.grad, "rnn2 parameter grad should exist")

    def test_two_rnn_layers_seq2seq_to_seq2one_final_state_loss(self):
        """
        Second layer's loss depends only on its final state h_T.
        This checks that gradient still flows through time and through the first layer.
        """
        T, N, D = 5, 2, 3
        H1, H2 = 4, 6

        rnn1 = RNN(D, H1, bias=False)
        rnn2 = RNN(H1, H2, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h1_seq, _ = rnn1.forward(x)
        h2_seq, h2_T = rnn2.forward(h1_seq)

        self.assertEqual(h2_seq.shape, (T, N, H2))
        self.assertEqual(h2_T.shape, (N, H2))

        loss = h2_T.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn1.parameters():
            self.assertIsNotNone(p.grad)
        for p in rnn2.parameters():
            self.assertIsNotNone(p.grad)

    def test_rnn_then_linear_seq2one_head(self):
        """
        Typical sequence-to-one: use h_T then Linear head to logits/pred.
        """
        T, N, D = 7, 4, 3
        H = 5
        C = 2  # "num classes" or regression outputs

        rnn = RNN(D, H, bias=True)
        head = Linear(in_features=H, out_features=C, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x)
        self.assertEqual(h_T.shape, (N, H))

        y = head.forward(h_T)
        self.assertEqual(y.shape, (N, C))

        # Simple scalar objective
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn.parameters():
            self.assertIsNotNone(p.grad, "RNN params should get grads")
        for p in head.parameters():
            self.assertIsNotNone(p.grad, "Linear head params should get grads")

    def test_rnn_then_linear_seq2seq_head_per_timestep_projection(self):
        """
        Sequence-to-sequence: apply a Linear head at each timestep.
        This uses a reshape trick (flatten time+batch), assuming Tensor.reshape exists.
        If your framework doesn't have reshape, you can replace with Flatten-like op.
        """
        T, N, D = 4, 3, 2
        H = 6
        K = 5  # per-timestep output dimension

        rnn = RNN(D, H, bias=True)
        head = Linear(in_features=H, out_features=K, bias=False)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        h_seq, _h_T = rnn.forward(x)
        self.assertEqual(h_seq.shape, (T, N, H))

        # Flatten (T,N,H) -> (T*N, H) -> apply Linear -> (T*N, K) -> reshape back
        # If your Tensor uses .reshape(...) or a dedicated reshape op, use it.
        if not hasattr(h_seq, "reshape"):
            self.skipTest(
                "Tensor.reshape not available; implement reshape or use Flatten module."
            )
        h_flat = h_seq.reshape((T * N, H))
        y_flat = head.forward(h_flat)
        self.assertEqual(y_flat.shape, (T * N, K))

        y_seq = y_flat.reshape((T, N, K))
        self.assertEqual(y_seq.shape, (T, N, K))

        loss = y_seq.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        for p in rnn.parameters():
            self.assertIsNotNone(p.grad)
        for p in head.parameters():
            self.assertIsNotNone(p.grad)

    def test_two_rnn_layers_and_linear_head_end2end_training_step(self):
        """
        A very small "one step" training sanity check:
          x -> rnn1 -> rnn2 -> h_T -> linear -> loss=sum -> backward
        Ensures the whole composed graph is differentiable.

        (We do not perform optimizer updates here; you already have separate train tests.)
        """
        T, N, D = 5, 2, 3
        H1, H2, C = 4, 4, 1

        rnn1 = RNN(D, H1, bias=True)
        rnn2 = RNN(H1, H2, bias=True)
        head = Linear(in_features=H2, out_features=C, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, device=self.device, requires_grad=True)

        _zero_all_grads(rnn1, rnn2, head)

        h1_seq, _ = rnn1.forward(x)
        h2_seq, h2_T = rnn2.forward(h1_seq)
        y = head.forward(h2_T)

        loss = y.sum()
        loss.backward()

        # Check grads exist
        self.assertIsNotNone(x.grad)
        for p in (
            list(rnn1.parameters()) + list(rnn2.parameters()) + list(head.parameters())
        ):
            self.assertIsNotNone(p.grad)
            g = p.grad.to_numpy()
            self.assertFalse(np.any(np.isnan(g)))
            self.assertFalse(np.any(np.isinf(g)))


if __name__ == "__main__":
    unittest.main()
