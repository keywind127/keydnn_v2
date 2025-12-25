from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.recurrent._lstm_module import LSTM
from src.keydnn.infrastructure.recurrent._bidirectional import Bidirectional


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


def _unwrap_out(ret):
    """
    LSTM/LSTM-like modules may return:
      - Tensor
      - (out, state)
      - (out, h_T, c_T)
      - (out, (h_T, c_T))
    For manual references we need only `out`.
    """
    return ret[0] if isinstance(ret, tuple) else ret


def _reverse_time(x: Tensor) -> Tensor:
    T = x.shape[0]
    xs = [x[t] for t in range(T - 1, -1, -1)]
    return Tensor.stack(xs, axis=0)


def _set_lstm_deterministic_params(lstm: LSTM, *, D: int, H: int, seed: int) -> None:
    """
    Assumes cell params:
      W_ih: (D, 4H)
      W_hh: (H, 4H)
      b_ih: (4H,) optional
      b_hh: (4H,) optional
    """
    rng = np.random.default_rng(seed)
    cell = lstm.cell

    Wih = rng.standard_normal((D, 4 * H)).astype(np.float32) * 0.05
    Whh = rng.standard_normal((H, 4 * H)).astype(np.float32) * 0.05
    _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
    _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)

    if getattr(cell, "bias", True):
        bih = rng.standard_normal((4 * H,)).astype(np.float32) * 0.02
        bhh = rng.standard_normal((4 * H,)).astype(np.float32) * 0.02
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)


class TestBidirectionalWithLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_bidirectional_lstm_forward_matches_manual_concat_reference(self):
        """
        Deterministic numerical correctness:

          y = concat( y_fwd(x),
                      reverse_time( y_bwd(reverse_time(x)) ),
                      axis=-1 )
        """
        np.random.seed(0)
        T, N, D, H = 5, 2, 3, 4

        # Template layer used by Bidirectional; wrapper will clone internally
        layer = LSTM(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        # Bidirectional clones forward/backward layers internally.
        _set_lstm_deterministic_params(bi.forward_layer, D=D, H=H, seed=123)
        _set_lstm_deterministic_params(bi.backward_layer, D=D, H=H, seed=456)

        x_np = np.random.randn(T, N, D).astype(np.float32) * 0.1
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        y = _unwrap_out(bi.forward(x))
        y_np = y.to_numpy()

        # Manual reference using the SAME cloned sublayers
        y_f = _unwrap_out(bi.forward_layer.forward(x))
        x_rev = _reverse_time(x)
        y_b_rev = _unwrap_out(bi.backward_layer.forward(x_rev))
        y_b = _reverse_time(y_b_rev)

        y_ref = Tensor.concat([y_f, y_b], axis=-1).to_numpy()

        self.assertEqual(y.shape, (T, N, 2 * H))
        np.testing.assert_allclose(y_np, y_ref, rtol=1e-6, atol=1e-6)

    def test_bidirectional_lstm_return_state_structure(self):
        """
        Expect: (out, state_f, state_b)
        where each state is (h_T, c_T).
        """
        np.random.seed(1)
        T, N, D, H = 4, 2, 3, 5

        layer = LSTM(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=False,
            return_state=True,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=False,
            return_state=True,
        )

        _set_lstm_deterministic_params(bi.forward_layer, D=D, H=H, seed=111)
        _set_lstm_deterministic_params(bi.backward_layer, D=D, H=H, seed=222)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D).astype(np.float32),
            self.device,
            requires_grad=False,
        )

        ret = bi.forward(x)
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 3)

        out, state_f, state_b = ret
        # out should already be the merged final output tensor for return_sequences=False
        self.assertEqual(out.shape, (N, 2 * H))

        self.assertIsInstance(state_f, tuple)
        self.assertIsInstance(state_b, tuple)
        self.assertEqual(len(state_f), 2)
        self.assertEqual(len(state_b), 2)

        h_f, c_f = state_f
        h_b, c_b = state_b

        self.assertEqual(h_f.shape, (N, H))
        self.assertEqual(c_f.shape, (N, H))
        self.assertEqual(h_b.shape, (N, H))
        self.assertEqual(c_b.shape, (N, H))

    def test_bidirectional_lstm_backward_runs(self):
        """
        Smoke test: x -> BiLSTM -> sum -> backward should run and populate x.grad.
        """
        np.random.seed(2)
        T, N, D, H = 6, 2, 3, 4

        layer = LSTM(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        _set_lstm_deterministic_params(bi.forward_layer, D=D, H=H, seed=1)
        _set_lstm_deterministic_params(bi.backward_layer, D=D, H=H, seed=2)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D).astype(np.float32), self.device, requires_grad=True
        )

        y = _unwrap_out(bi.forward(x))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
