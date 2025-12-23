from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure.recurrent._rnn_module import RNNCell


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    """
    Same pattern as your Conv2d tests: treat Parameter as Tensor-like
    or unwrap .data/.tensor if you later change design.
    """
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


# NOTE: Removed _ensure_parameter_from_numpy and all Parameter.from_numpy usage.
# After updating RNNCell.__post_init__ to use explicit Parameter(...) + copy_from_numpy,
# tests should not patch or depend on from_numpy at all.


class TestRNNCellForward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_rnncell_forward_shape_and_dtype(self):
        np.random.seed(0)
        N, D, H = 4, 3, 5
        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=False
        )

        out = cell.forward(x, h_prev)

        self.assertEqual(out.shape, (N, H))
        self.assertEqual(out.to_numpy().dtype, np.float32)

    def test_rnncell_forward_matches_numpy_reference_with_fixed_params(self):
        np.random.seed(123)
        N, D, H = 2, 3, 4
        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

        # Deterministic params
        Wih = np.arange(D * H, dtype=np.float32).reshape(D, H) / 10.0
        Whh = (np.arange(H * H, dtype=np.float32).reshape(H, H) / 20.0) - 0.1
        bih = np.linspace(-0.2, 0.2, H, dtype=np.float32)
        bhh = np.linspace(0.1, -0.1, H, dtype=np.float32)

        _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
        _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)

        x_np = np.arange(N * D, dtype=np.float32).reshape(N, D) / 5.0
        h_np = (np.arange(N * H, dtype=np.float32).reshape(N, H) / 7.0) - 0.3

        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h_prev = _tensor_from_numpy(h_np, self.device, requires_grad=False)

        out = cell.forward(x, h_prev).to_numpy()

        # Reference (matches your bias-expansion rule)
        a = x_np @ Wih + h_np @ Whh
        a = (
            a
            + np.repeat(bih.reshape(1, -1), N, axis=0)
            + np.repeat(bhh.reshape(1, -1), N, axis=0)
        )
        ref = np.tanh(a).astype(np.float32)

        self.assertTrue(np.allclose(out, ref, atol=1e-6))


class TestRNNCellBackward(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_rnncell_backward_populates_grads_bias_true(self):
        np.random.seed(1)
        N, D, H = 3, 4, 5
        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

        # Ensure params require grad (Parameter defaults True, but be explicit)
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

    def test_rnncell_backward_no_bias_path(self):
        np.random.seed(2)
        N, D, H = 2, 3, 4
        cell = RNNCell(input_size=D, hidden_size=H, bias=False)

        for p in cell.parameters():
            _unwrap_param_tensor(p).requires_grad = True

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        out = cell.forward(x, h_prev)
        out.sum().backward()

        Wih = _unwrap_param_tensor(cell.W_ih)
        Whh = _unwrap_param_tensor(cell.W_hh)
        self.assertIsNotNone(Wih.grad)
        self.assertIsNotNone(Whh.grad)

        self.assertIsNone(cell.b_ih)
        self.assertIsNone(cell.b_hh)

    def test_rnncell_backward_batch1_edge_case(self):
        np.random.seed(3)
        N, D, H = 1, 3, 2
        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

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


class TestRNNCellConstruction(TestCase):
    """
    Extra edge-case coverage:
    - Confirms RNNCell parameters are Tensor-like and on CPU device string.
    - Confirms bias flags control existence of bias parameters.
    """

    def test_rnncell_constructs_parameters_with_bias(self):
        cell = RNNCell(input_size=3, hidden_size=4, bias=True)

        self.assertIsNotNone(cell.W_ih)
        self.assertIsNotNone(cell.W_hh)
        self.assertIsNotNone(cell.b_ih)
        self.assertIsNotNone(cell.b_hh)

        self.assertEqual(_unwrap_param_tensor(cell.W_ih).shape, (3, 4))
        self.assertEqual(_unwrap_param_tensor(cell.W_hh).shape, (4, 4))
        self.assertEqual(_unwrap_param_tensor(cell.b_ih).shape, (4,))
        self.assertEqual(_unwrap_param_tensor(cell.b_hh).shape, (4,))

    def test_rnncell_constructs_parameters_without_bias(self):
        cell = RNNCell(input_size=3, hidden_size=4, bias=False)

        self.assertIsNotNone(cell.W_ih)
        self.assertIsNotNone(cell.W_hh)
        self.assertIsNone(cell.b_ih)
        self.assertIsNone(cell.b_hh)


if __name__ == "__main__":
    unittest.main()
