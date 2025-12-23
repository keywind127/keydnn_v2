from __future__ import annotations

import unittest
from unittest import TestCase
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.recurrent._rnn_module import RNN
from src.keydnn.infrastructure.recurrent._bidirectional import Bidirectional


# -----------------------------
# Helpers (match your style)
# -----------------------------
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


def _zero_grads(module) -> None:
    for p in module.parameters():
        pt = _unwrap_param_tensor(p)
        if hasattr(pt, "zero_grad"):
            pt.zero_grad()


# -----------------------------
# Tests
# -----------------------------
class TestBidirectionalRNNForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_forward_return_sequences_true_shape(self):
        np.random.seed(0)
        T, N, D, H = 5, 3, 4, 6

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=False)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        y = bi.forward(x)

        self.assertIsInstance(y, Tensor)
        self.assertEqual(y.shape, (T, N, 2 * H))

    def test_forward_return_sequences_false_shape(self):
        np.random.seed(1)
        T, N, D, H = 7, 2, 3, 5

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=False, return_state=False)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        y = bi.forward(x)

        self.assertIsInstance(y, Tensor)
        self.assertEqual(y.shape, (N, 2 * H))

    def test_forward_return_state_true_shapes(self):
        np.random.seed(2)
        T, N, D, H = 4, 2, 3, 5

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=True)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        out = bi.forward(x)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)

        y, h_f, h_b = out
        self.assertEqual(y.shape, (T, N, 2 * H))
        self.assertEqual(h_f.shape, (N, H))
        self.assertEqual(h_b.shape, (N, H))

    def test_forward_rejects_non_3d_input(self):
        base = RNN(input_size=3, hidden_size=4, bias=True)
        bi = Bidirectional(base)

        x = Tensor(shape=(2, 3), device=self.device, requires_grad=False)  # not (T,N,D)
        with self.assertRaises(ValueError):
            _ = bi.forward(x)

    def test_get_config_and_from_config_roundtrip(self):
        base = RNN(input_size=3, hidden_size=4, bias=False)
        bi1 = Bidirectional(base, return_sequences=False, return_state=True)

        cfg = bi1.get_config()
        bi2 = Bidirectional.from_config(cfg)

        self.assertEqual(bi2.merge_mode, "concat")
        self.assertEqual(bi2.return_sequences, False)
        self.assertEqual(bi2.return_state, True)

        # Ensure inner RNN hyperparams survive roundtrip
        self.assertEqual(bi2.forward_rnn.cell.input_size, 3)
        self.assertEqual(bi2.forward_rnn.cell.hidden_size, 4)
        self.assertEqual(bi2.forward_rnn.cell.bias, False)
        self.assertEqual(bi2.backward_rnn.cell.input_size, 3)
        self.assertEqual(bi2.backward_rnn.cell.hidden_size, 4)
        self.assertEqual(bi2.backward_rnn.cell.bias, False)


class TestBidirectionalRNNAutograd(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_backward_populates_x_grad_return_sequences_true(self):
        """
        x -> BiRNN(return_sequences=True) -> sum -> backward
        ensures grad flows to x (via __getitem__ + stack + concat graph).
        """
        np.random.seed(10)
        T, N, D, H = 6, 2, 3, 4

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=False)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        y = bi.forward(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Expected x.grad to be populated")
        self.assertEqual(x.grad.shape, x.shape)

        g = x.grad.to_numpy()
        self.assertTrue(np.isfinite(g).all())
        self.assertFalse(np.all(g == 0.0), "Expected non-zero grads for x")

    def test_backward_populates_x_grad_return_sequences_false(self):
        """
        x -> BiRNN(return_sequences=False) -> sum -> backward
        ensures grad flows to x even when only final merged output is returned.
        """
        np.random.seed(11)
        T, N, D, H = 5, 3, 2, 4

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=False, return_state=False)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        y = bi.forward(x)
        self.assertEqual(y.shape, (N, 2 * H))

        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())

    def test_backward_populates_parameter_grads_both_directions(self):
        """
        Sanity: backward should populate some parameter grads in BOTH directions.
        """
        np.random.seed(12)
        T, N, D, H = 4, 2, 3, 5

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=False)

        # Ensure all params require grad
        for p in bi.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        y = bi.forward(x)
        y.sum().backward()

        # Check at least one grad exists for forward and backward RNN params
        f_any = False
        for p in bi.forward_rnn.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                f_any = True
                break
        b_any = False
        for p in bi.backward_rnn.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                b_any = True
                break

        self.assertTrue(f_any, "Expected at least one forward-direction param grad")
        self.assertTrue(b_any, "Expected at least one backward-direction param grad")

    def test_return_state_true_backward_runs(self):
        """
        x -> BiRNN(return_state=True) -> use all outputs -> backward
        """
        np.random.seed(13)
        T, N, D, H = 5, 2, 3, 4

        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=True)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        y, h_f, h_b = bi.forward(x)

        # touch all outputs
        loss = y.sum() + h_f.sum() + h_b.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())

    def test_forward_matches_numpy_reference_deterministic_params(self):
        """
        Deterministic numerical correctness test.

        We set known weights/biases for BOTH directions, run Bidirectional forward,
        and compare to a NumPy reference implementation of:
            h_t = tanh(x_t @ Wih + h_{t-1} @ Whh + b_ih + b_hh)

        Then merge_mode="concat" => concat([h_f, h_b], axis=-1).
        """
        np.random.seed(123)

        T, N, D, H = 4, 2, 3, 2
        base = RNN(input_size=D, hidden_size=H, bias=True)
        bi = Bidirectional(base, return_sequences=True, return_state=False)

        # ---- deterministic parameters (different for forward/backward to catch mixups) ----
        Wih_f = np.array(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=np.float32
        )  # (D,H)
        Whh_f = np.array([[0.2, 0.1], [-0.3, 0.05]], dtype=np.float32)  # (H,H)
        bih_f = np.array([0.01, -0.02], dtype=np.float32)  # (H,)
        bhh_f = np.array([0.03, 0.04], dtype=np.float32)  # (H,)

        Wih_b = np.array([[-0.05, 0.15], [0.25, -0.35], [0.45, 0.05]], dtype=np.float32)
        Whh_b = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
        bih_b = np.array([0.02, 0.01], dtype=np.float32)
        bhh_b = np.array([-0.01, 0.05], dtype=np.float32)

        # ---- assign into forward cell ----
        fcell = bi.forward_rnn.cell
        _unwrap_param_tensor(fcell.W_ih).copy_from_numpy(Wih_f)
        _unwrap_param_tensor(fcell.W_hh).copy_from_numpy(Whh_f)
        _unwrap_param_tensor(fcell.b_ih).copy_from_numpy(bih_f)
        _unwrap_param_tensor(fcell.b_hh).copy_from_numpy(bhh_f)

        # ---- assign into backward cell ----
        bcell = bi.backward_rnn.cell
        _unwrap_param_tensor(bcell.W_ih).copy_from_numpy(Wih_b)
        _unwrap_param_tensor(bcell.W_hh).copy_from_numpy(Whh_b)
        _unwrap_param_tensor(bcell.b_ih).copy_from_numpy(bih_b)
        _unwrap_param_tensor(bcell.b_hh).copy_from_numpy(bhh_b)

        # ---- input ----
        x_np = (np.arange(T * N * D, dtype=np.float32).reshape(T, N, D) / 10.0) - 0.2
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        # ---- KeyDNN output ----
        y = bi.forward(x).to_numpy()  # (T,N,2H)

        # ---- NumPy reference ----
        def rnn_seq_numpy(
            x_seq: np.ndarray,
            Wih: np.ndarray,
            Whh: np.ndarray,
            bih: np.ndarray,
            bhh: np.ndarray,
        ) -> np.ndarray:
            T_, N_, D_ = x_seq.shape
            H_ = Whh.shape[0]
            h_prev = np.zeros((N_, H_), dtype=np.float32)
            hs = []
            for t in range(T_):
                a = x_seq[t] @ Wih + h_prev @ Whh
                a = (
                    a
                    + bih.reshape(1, -1).repeat(N_, axis=0)
                    + bhh.reshape(1, -1).repeat(N_, axis=0)
                )
                h_prev = np.tanh(a).astype(np.float32)
                hs.append(h_prev)
            return np.stack(hs, axis=0)  # (T,N,H)

        # forward direction: time 0..T-1
        h_f = rnn_seq_numpy(x_np, Wih_f, Whh_f, bih_f, bhh_f)

        # backward direction: process reversed time, then align back to original order
        x_rev = x_np[::-1].copy()  # (T,N,D)
        h_b_rev = rnn_seq_numpy(
            x_rev, Wih_b, Whh_b, bih_b, bhh_b
        )  # corresponds to times T-1..0
        h_b = h_b_rev[::-1].copy()  # align to original time order

        ref = np.concatenate([h_f, h_b], axis=2)  # (T,N,2H)

        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
