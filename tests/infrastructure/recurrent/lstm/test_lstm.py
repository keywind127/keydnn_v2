from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.recurrent._lstm_module import LSTMCell


# -----------------------------
# Helpers (match Conv2d/RNN style)
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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid, float32
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    expx = np.exp(x[neg], dtype=np.float32)
    out[neg] = expx / (1.0 + expx)
    return out


def _lstmcell_numpy_reference(
    x: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    Wih: np.ndarray,
    Whh: np.ndarray,
    bih: np.ndarray | None,
    bhh: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy reference for one-step LSTMCell forward matching the implementation:

      gates = x @ Wih + h_prev @ Whh + b_ih + b_hh
      split -> i,f,g,o (each N,H)
      i,f,o sigmoid, g tanh
      c_t = f*c_prev + i*g
      h_t = o*tanh(c_t)
    """
    gates = x @ Wih + h_prev @ Whh
    if bih is not None and bhh is not None:
        N = x.shape[0]
        gates = (
            gates
            + np.repeat(bih.reshape(1, -1), N, axis=0)
            + np.repeat(bhh.reshape(1, -1), N, axis=0)
        )

    H = h_prev.shape[1]
    ai = gates[:, 0:H]
    af = gates[:, H : 2 * H]
    ag = gates[:, 2 * H : 3 * H]
    ao = gates[:, 3 * H : 4 * H]

    i = _sigmoid_np(ai)
    f = _sigmoid_np(af)
    g = np.tanh(ag).astype(np.float32)
    o = _sigmoid_np(ao)

    c_t = (f * c_prev + i * g).astype(np.float32)
    h_t = (o * np.tanh(c_t).astype(np.float32)).astype(np.float32)
    return h_t, c_t


# -----------------------------
# Tests
# -----------------------------
class TestLSTMCellForward(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_lstmcell_forward_shape_and_dtype(self):
        np.random.seed(0)
        N, D, H = 4, 3, 5
        cell = LSTMCell(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=False
        )
        c_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=False
        )

        h_t, c_t = cell.forward(x, h_prev, c_prev)

        self.assertEqual(h_t.shape, (N, H))
        self.assertEqual(c_t.shape, (N, H))
        self.assertEqual(h_t.to_numpy().dtype, np.float32)
        self.assertEqual(c_t.to_numpy().dtype, np.float32)

    def test_lstmcell_forward_matches_numpy_reference_with_fixed_params(self):
        np.random.seed(123)
        N, D, H = 2, 3, 4
        cell = LSTMCell(input_size=D, hidden_size=H, bias=True)

        # Deterministic params
        Wih = (np.arange(D * 4 * H, dtype=np.float32).reshape(D, 4 * H) / 50.0) - 0.2
        Whh = (np.arange(H * 4 * H, dtype=np.float32).reshape(H, 4 * H) / 70.0) + 0.1
        bih = np.linspace(-0.3, 0.3, 4 * H, dtype=np.float32)
        bhh = np.linspace(0.2, -0.2, 4 * H, dtype=np.float32)

        _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
        _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)

        x_np = (np.arange(N * D, dtype=np.float32).reshape(N, D) / 5.0) - 0.1
        h_np = (np.arange(N * H, dtype=np.float32).reshape(N, H) / 7.0) + 0.05
        c_np = (np.arange(N * H, dtype=np.float32).reshape(N, H) / 9.0) - 0.2

        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(h_np, self.device, requires_grad=False)
        c_prev = _tensor_from_numpy(c_np, self.device, requires_grad=False)

        h_t, c_t = cell.forward(x, h_prev, c_prev)
        h_out = h_t.to_numpy()
        c_out = c_t.to_numpy()

        h_ref, c_ref = _lstmcell_numpy_reference(
            x_np, h_np, c_np, Wih, Whh, bih=bih, bhh=bhh
        )

        np.testing.assert_allclose(h_out, h_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(c_out, c_ref, rtol=1e-6, atol=1e-6)


class TestLSTMCellBackward(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_backward_populates_grads_for_x_h_c_and_params_bias_true(self):
        np.random.seed(1)
        N, D, H = 3, 4, 5
        cell = LSTMCell(input_size=D, hidden_size=H, bias=True)

        # Ensure params require grad
        for p in cell.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )
        c_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        h_t, c_t = cell.forward(x, h_prev, c_prev)

        # scalar loss using BOTH outputs to ensure both ctx paths contribute
        loss = h_t.sum() + c_t.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertIsNotNone(c_prev.grad)

        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h_prev.grad.shape, h_prev.shape)
        self.assertEqual(c_prev.grad.shape, c_prev.shape)

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        bih = _unwrap_param_tensor(cell.b_ih)
        bhh = _unwrap_param_tensor(cell.b_hh)

        self.assertIsNotNone(Wih.grad)
        self.assertIsNotNone(Whh.grad)
        self.assertIsNotNone(bih.grad)
        self.assertIsNotNone(bhh.grad)

        self.assertEqual(Wih.grad.shape, Wih.shape)  # (D,4H)
        self.assertEqual(Whh.grad.shape, Whh.shape)  # (H,4H)
        self.assertEqual(bih.grad.shape, bih.shape)  # (4H,)
        self.assertEqual(bhh.grad.shape, bhh.shape)  # (4H,)

        # sanity: grads finite
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(h_prev.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(c_prev.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(Wih.grad.to_numpy()).all())
        self.assertTrue(np.isfinite(Whh.grad.to_numpy()).all())

    def test_backward_runs_when_bias_false(self):
        np.random.seed(2)
        N, D, H = 2, 3, 4
        cell = LSTMCell(input_size=D, hidden_size=H, bias=False)

        for p in cell.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )
        c_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        h_t, c_t = cell.forward(x, h_prev, c_prev)
        (h_t.sum() + c_t.sum()).backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertIsNotNone(c_prev.grad)

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        self.assertIsNotNone(Wih.grad)
        self.assertIsNotNone(Whh.grad)

        self.assertIsNone(cell.b_ih)
        self.assertIsNone(cell.b_hh)

    def test_backward_batch1_edge_case(self):
        np.random.seed(3)
        N, D, H = 1, 3, 2
        cell = LSTMCell(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )
        c_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        h_t, c_t = cell.forward(x, h_prev, c_prev)
        (h_t.sum() + c_t.sum()).backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(h_prev.grad)
        self.assertIsNotNone(c_prev.grad)

        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(h_prev.grad.shape, h_prev.shape)
        self.assertEqual(c_prev.grad.shape, c_prev.shape)


class TestLSTMCellConstruction(unittest.TestCase):
    def test_constructs_parameters_with_bias(self):
        cell = LSTMCell(input_size=3, hidden_size=4, bias=True)

        self.assertIsNotNone(cell.W_ih)
        self.assertIsNotNone(cell.W_hh)
        self.assertIsNotNone(cell.b_ih)
        self.assertIsNotNone(cell.b_hh)

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        bih = _unwrap_param_tensor(cell.b_ih)
        bhh = _unwrap_param_tensor(cell.b_hh)

        self.assertEqual(Wih.shape, (3, 16))
        self.assertEqual(Whh.shape, (4, 16))
        self.assertEqual(bih.shape, (16,))
        self.assertEqual(bhh.shape, (16,))

    def test_constructs_parameters_without_bias(self):
        cell = LSTMCell(input_size=3, hidden_size=4, bias=False)

        self.assertIsNotNone(cell.W_ih)
        self.assertIsNotNone(cell.W_hh)
        self.assertIsNone(cell.b_ih)
        self.assertIsNone(cell.b_hh)

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        self.assertEqual(Wih.shape, (3, 16))
        self.assertEqual(Whh.shape, (4, 16))


if __name__ == "__main__":
    unittest.main()
