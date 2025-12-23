from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.recurrent._gru_module import GRU, GRUCell


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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    ex = np.exp(x[neg], dtype=np.float32)
    out[neg] = ex / (1.0 + ex)
    return out


def _gru_cell_numpy_reference(
    x: np.ndarray,
    h_prev: np.ndarray,
    W_ih: np.ndarray,
    W_hh: np.ndarray,
    b_ih: np.ndarray | None,
    b_hh: np.ndarray | None,
) -> np.ndarray:
    """
    NumPy reference for one GRU step, matching the implementation described:

      W_ih: (D, 3H) -> [W_iz | W_ir | W_in]
      W_hh: (H, 3H) -> [W_hz | W_hr | W_hn]
      b_ih, b_hh: (3H,)

    gates:
      z = sigmoid(x W_iz + h W_hz + b_z)
      r = sigmoid(x W_ir + h W_hr + b_r)
      n = tanh(   x W_in + (r*h) W_hn + b_n)
      h = (1-z)*n + z*h_prev
    """
    x = x.astype(np.float32, copy=False)
    h_prev = h_prev.astype(np.float32, copy=False)
    W_ih = W_ih.astype(np.float32, copy=False)
    W_hh = W_hh.astype(np.float32, copy=False)

    H = h_prev.shape[1]
    W_iz = W_ih[:, 0:H]
    W_ir = W_ih[:, H : 2 * H]
    W_in = W_ih[:, 2 * H : 3 * H]

    W_hz = W_hh[:, 0:H]
    W_hr = W_hh[:, H : 2 * H]
    W_hn = W_hh[:, 2 * H : 3 * H]

    if b_ih is not None and b_hh is not None:
        b_ih = b_ih.astype(np.float32, copy=False)
        b_hh = b_hh.astype(np.float32, copy=False)
        b_z = (b_ih[0:H] + b_hh[0:H]).reshape(1, -1)
        b_r = (b_ih[H : 2 * H] + b_hh[H : 2 * H]).reshape(1, -1)
        b_n = (b_ih[2 * H : 3 * H] + b_hh[2 * H : 3 * H]).reshape(1, -1)
    else:
        b_z = b_r = b_n = None

    a_z = x @ W_iz + h_prev @ W_hz
    a_r = x @ W_ir + h_prev @ W_hr
    if b_z is not None:
        a_z = a_z + np.repeat(b_z, x.shape[0], axis=0)
        a_r = a_r + np.repeat(b_r, x.shape[0], axis=0)

    z = _sigmoid_np(a_z)
    r = _sigmoid_np(a_r)

    rh = r * h_prev
    a_n = x @ W_in + rh @ W_hn
    if b_n is not None:
        a_n = a_n + np.repeat(b_n, x.shape[0], axis=0)

    n = np.tanh(a_n).astype(np.float32)
    h = ((1.0 - z) * n + z * h_prev).astype(np.float32)
    return h


def _set_gru_deterministic_params(cell: GRUCell, *, D: int, H: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    Wih = (rng.standard_normal((D, 3 * H)).astype(np.float32)) * 0.05
    Whh = (rng.standard_normal((H, 3 * H)).astype(np.float32)) * 0.05
    _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
    _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)

    if getattr(cell, "bias", True):
        bih = (rng.standard_normal((3 * H,)).astype(np.float32)) * 0.02
        bhh = (rng.standard_normal((3 * H,)).astype(np.float32)) * 0.02
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)


class TestGRUCellForward(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_grucell_forward_shape_and_dtype(self):
        np.random.seed(0)
        N, D, H = 4, 3, 5
        cell = GRUCell(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=False
        )

        out = cell.forward(x, h_prev)

        self.assertEqual(out.shape, (N, H))
        self.assertEqual(out.to_numpy().dtype, np.float32)

    def test_grucell_forward_matches_numpy_reference_deterministic_params(self):
        np.random.seed(1)
        N, D, H = 2, 3, 4
        cell = GRUCell(input_size=D, hidden_size=H, bias=True)
        _set_gru_deterministic_params(cell, D=D, H=H, seed=123)

        # Inputs
        x_np = (np.arange(N * D, dtype=np.float32).reshape(N, D) / 10.0) - 0.2
        h_np = (np.arange(N * H, dtype=np.float32).reshape(N, H) / 12.0) + 0.1

        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(h_np, self.device, requires_grad=False)

        y = cell.forward(x, h_prev).to_numpy()

        W_ih = _unwrap_param_tensor(cell.W_ih).to_numpy()
        W_hh = _unwrap_param_tensor(cell.W_hh).to_numpy()
        b_ih = _unwrap_param_tensor(cell.b_ih).to_numpy()
        b_hh = _unwrap_param_tensor(cell.b_hh).to_numpy()

        ref = _gru_cell_numpy_reference(x_np, h_np, W_ih, W_hh, b_ih, b_hh)
        np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)

    def test_grucell_forward_no_bias_path(self):
        np.random.seed(2)
        N, D, H = 3, 4, 5
        cell = GRUCell(input_size=D, hidden_size=H, bias=False)
        _set_gru_deterministic_params(cell, D=D, H=H, seed=7)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=False
        )

        y = cell.forward(x, h_prev)
        self.assertEqual(y.shape, (N, H))
        self.assertIsNone(getattr(cell, "b_ih", None))
        self.assertIsNone(getattr(cell, "b_hh", None))


class TestGRUCellBackward(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_backward_populates_grads_for_x_h_and_params_bias_true(self):
        np.random.seed(3)
        N, D, H = 4, 3, 5
        cell = GRUCell(input_size=D, hidden_size=H, bias=True)
        _set_gru_deterministic_params(cell, D=D, H=H, seed=999)

        # ensure requires_grad set
        for p in cell.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        out = cell.forward(x, h_prev)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h_prev.grad.shape, h_prev.shape)

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        bih = _unwrap_param_tensor(cell.b_ih)
        bhh = _unwrap_param_tensor(cell.b_hh)

        self.assertIsNotNone(Wih.grad)
        self.assertIsNotNone(Whh.grad)
        self.assertIsNotNone(bih.grad)
        self.assertIsNotNone(bhh.grad)

        self.assertEqual(Wih.grad.shape, Wih.shape)
        self.assertEqual(Whh.grad.shape, Whh.shape)
        self.assertEqual(bih.grad.shape, bih.shape)
        self.assertEqual(bhh.grad.shape, bhh.shape)

        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(h_prev.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(Wih.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(Whh.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(bih.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(bhh.grad.to_numpy()).all())

    def test_backward_runs_when_bias_false(self):
        np.random.seed(4)
        N, D, H = 2, 3, 4
        cell = GRUCell(input_size=D, hidden_size=H, bias=False)
        _set_gru_deterministic_params(cell, D=D, H=H, seed=1234)

        for p in cell.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        y = cell.forward(x, h_prev)
        y.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertIsNotNone(_unwrap_param_tensor(cell.W_ih).grad)
        self.assertIsNotNone(_unwrap_param_tensor(cell.W_hh).grad)

        self.assertIsNone(getattr(cell, "b_ih", None))
        self.assertIsNone(getattr(cell, "b_hh", None))


class TestGRUSequenceModule(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_gru_rejects_non_3d_input(self):
        gru = GRU(input_size=3, hidden_size=4, bias=True)
        x = Tensor(shape=(2, 3), device=self.device, requires_grad=False, ctx=None)
        with self.assertRaises(ValueError):
            gru.forward(x)

    def test_gru_forward_shapes_keras_compat_sequences(self):
        np.random.seed(10)
        T, N, D, H = 5, 2, 3, 4
        gru = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )
        _set_gru_deterministic_params(gru.cell, D=D, H=H, seed=101)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        y = gru.forward(x)

        self.assertIsInstance(y, Tensor)
        self.assertEqual(y.shape, (T, N, H))

    def test_gru_forward_shapes_keras_compat_state(self):
        np.random.seed(11)
        T, N, D, H = 6, 3, 2, 5
        gru = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=False,
            return_state=True,
            keras_compat=True,
        )
        _set_gru_deterministic_params(gru.cell, D=D, H=H, seed=202)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        out = gru.forward(x)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        y, h_T = out
        self.assertEqual(y.shape, (N, H))
        self.assertEqual(h_T.shape, (N, H))

    def test_gru_bptt_populates_x_grad(self):
        """
        E2E sanity: x -> GRU -> sum -> backward should populate x.grad.
        """
        np.random.seed(12)
        T, N, D, H = 7, 2, 3, 4
        gru = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=True,
            keras_compat=True,
        )
        _set_gru_deterministic_params(gru.cell, D=D, H=H, seed=303)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        y_seq, h_T = gru.forward(x)

        loss = y_seq.sum() + h_T.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())

        # also check some parameter grad exists
        any_param_grad = False
        for p in gru.parameters():
            pt = _unwrap_param_tensor(p)
            if getattr(pt, "grad", None) is not None:
                any_param_grad = True
                break
        self.assertTrue(any_param_grad)

    def test_get_config_from_config_roundtrip(self):
        gru1 = GRU(
            input_size=3,
            hidden_size=4,
            bias=False,
            return_sequences=False,
            return_state=True,
            keras_compat=True,
        )
        cfg = gru1.get_config()
        gru2 = GRU.from_config(cfg)

        self.assertEqual(gru2.cell.input_size, gru1.cell.input_size)
        self.assertEqual(gru2.cell.hidden_size, gru1.cell.hidden_size)
        self.assertEqual(gru2.cell.bias, gru1.cell.bias)
        self.assertEqual(gru2.return_sequences, gru1.return_sequences)
        self.assertEqual(gru2.return_state, gru1.return_state)
        self.assertEqual(gru2.keras_compat, gru1.keras_compat)


if __name__ == "__main__":
    unittest.main()
