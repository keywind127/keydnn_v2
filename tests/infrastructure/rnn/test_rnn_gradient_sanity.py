import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._rnn_module import RNN


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    """
    Same convention as your Conv2d tests:
    - If Parameter is Tensor-like, use it directly.
    - Otherwise, unwrap common wrapper fields.
    """
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


def _zero_module_grads(m) -> None:
    """Clear all parameter grads (idempotent)."""
    for p in m.parameters():
        _unwrap_param_tensor(p).zero_grad()


class TestRNNEndToEndGradients(unittest.TestCase):
    """
    End-to-end gradient tests for RNN (sequence-level), intended to validate
    BPTT wiring through:
      - x[t] via Tensor.__getitem__
      - Tensor.stack backward distributing grads to per-timestep outputs
      - parameter grad accumulation over timesteps
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def _build_tiny_rnn_and_input(self, *, T=2, N=1, D=2, H=2):
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        # Make everything deterministic (so finite-diff is stable)
        # Deterministic parameters (small magnitudes reduce tanh saturation)
        Wih = (np.arange(D * H, dtype=np.float32).reshape(D, H) - 1.5) / 10.0
        Whh = (np.arange(H * H, dtype=np.float32).reshape(H, H) - 1.0) / 10.0
        bih = np.linspace(-0.05, 0.05, H, dtype=np.float32)
        bhh = np.linspace(0.02, -0.02, H, dtype=np.float32)

        # Your RNN owns a single cell
        cell = rnn.cell
        _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
        _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)

        # Tiny input
        x_np = (np.arange(T * N * D, dtype=np.float32).reshape(T, N, D) - 1.0) / 5.0
        x = _tensor_from_numpy(x_np, self.device, requires_grad=True)

        return rnn, x

    def test_rnn_bptt_gradcheck_wih_single_element(self):
        """
        Finite-difference check for one element of W_ih under loss = sum(h_seq).
        This catches missing timestep accumulation and transpose mistakes.
        """
        rnn, x = self._build_tiny_rnn_and_input(T=3, N=1, D=2, H=2)

        cell = rnn.cell
        Wih = _unwrap_param_tensor(cell.W_ih)

        eps = 1e-3
        i, j = 0, 1  # check a single entry

        # ---- Analytic gradient ----
        _zero_module_grads(rnn)
        x.zero_grad()

        h_seq, _hT = rnn.forward(x)
        loss = h_seq.sum()
        loss.backward()

        self.assertIsNotNone(Wih.grad, "W_ih.grad should exist after backward()")
        grad_analytic = float(Wih.grad.to_numpy()[i, j])

        # ---- Numerical gradient (central difference) ----
        W_base = Wih.to_numpy().copy()

        W_plus = W_base.copy()
        W_plus[i, j] += eps
        Wih.copy_from_numpy(W_plus)
        _zero_module_grads(rnn)
        x.zero_grad()
        h_seq_p, _ = rnn.forward(x)
        loss_p = float(
            h_seq_p.sum().to_numpy()
        )  # scalar tensor -> numpy scalar -> float

        W_minus = W_base.copy()
        W_minus[i, j] -= eps
        Wih.copy_from_numpy(W_minus)
        _zero_module_grads(rnn)
        x.zero_grad()
        h_seq_m, _ = rnn.forward(x)
        loss_m = float(h_seq_m.sum().to_numpy())

        # Restore
        Wih.copy_from_numpy(W_base)

        grad_num = (loss_p - loss_m) / (2.0 * eps)

        self.assertTrue(
            np.allclose(grad_analytic, grad_num, atol=5e-3, rtol=5e-2),
            msg=f"Gradcheck failed: analytic={grad_analytic:.6f}, numerical={grad_num:.6f}",
        )

    def test_rnn_bptt_produces_input_gradient(self):
        """
        Basic end-to-end wiring check: x.grad should be populated for a sequence input.
        """
        rnn, x = self._build_tiny_rnn_and_input(T=4, N=2, D=3, H=3)

        _zero_module_grads(rnn)
        x.zero_grad()

        h_seq, _hT = rnn.forward(x)
        loss = h_seq.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "x.grad should be populated after backward()")
        self.assertEqual(x.grad.shape, x.shape)

        gx = x.grad.to_numpy()
        self.assertEqual(gx.shape, (4, 2, 3))
        self.assertTrue(np.isfinite(gx).all(), "x.grad should be finite")


class TestRNNTinyTrainingSanity(unittest.TestCase):
    """
    A tiny training sanity test: run a few SGD steps and confirm the scalar loss decreases.

    This intentionally uses a simple differentiable scalar loss that should work with
    your existing ops: loss = sum(h_seq). (No elementwise square/subtraction required.)
    """

    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(123)

    def test_rnn_tiny_training_loss_decreases(self):
        T, N, D, H = 5, 2, 3, 4
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        # Input sequence (requires_grad can be False for "parameter-only" training)
        x_np = np.random.randn(T, N, D).astype(np.float32) * 0.1
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        lr = 1e-2
        steps = 30

        def forward_loss() -> float:
            h_seq, _ = rnn.forward(x)
            loss_t = h_seq.sum()
            return float(loss_t.to_numpy())

        # Initial loss
        _zero_module_grads(rnn)
        loss0 = forward_loss()

        # Training loop: minimize loss = sum(h_seq)
        for _ in range(steps):
            _zero_module_grads(rnn)

            h_seq, _ = rnn.forward(x)
            loss = h_seq.sum()
            loss.backward()

            # SGD step over all parameters
            for p in rnn.parameters():
                pt = _unwrap_param_tensor(p)
                if not getattr(pt, "requires_grad", False):
                    continue
                if pt.grad is None:
                    continue
                p_np = pt.to_numpy()
                g_np = pt.grad.to_numpy()
                pt.copy_from_numpy((p_np - lr * g_np).astype(np.float32))

        loss1 = forward_loss()

        # We expect descent to reduce the scalar objective value.
        # (If you later change the objective, update this accordingly.)
        self.assertLess(
            loss1,
            loss0,
            msg=f"Expected loss to decrease after SGD steps, got loss0={loss0:.6f}, loss1={loss1:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
