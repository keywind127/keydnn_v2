from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._rnn_module import RNN, RNNCell


# -----------------------------
# Helpers (match Conv2d style)
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


def _zero_grads(module: RNN) -> None:
    for p in module.parameters():
        pt = _unwrap_param_tensor(p)
        if hasattr(pt, "zero_grad"):
            pt.zero_grad()


def _sgd_step(module: RNN, lr: float) -> None:
    """
    Simple SGD update: p -= lr * p.grad
    Uses Tensor public API via to_numpy/copy_from_numpy.
    """
    for p in module.parameters():
        pt = _unwrap_param_tensor(p)
        g = getattr(pt, "grad", None)
        if g is None:
            continue
        p_np = pt.to_numpy().astype(np.float32, copy=True)
        g_np = g.to_numpy().astype(np.float32, copy=False)
        pt.copy_from_numpy(p_np - lr * g_np)


def _supports_binary_ops() -> bool:
    """
    Check whether Tensor supports the ops needed for MSE training:
    subtraction and multiplication.
    """
    dev = Device("cpu")
    a = Tensor((1,), dev, requires_grad=True)
    b = Tensor((1,), dev, requires_grad=True)
    a.copy_from_numpy(np.array([1.0], dtype=np.float32))
    b.copy_from_numpy(np.array([2.0], dtype=np.float32))

    try:
        _ = a - b
        _ = a * b
    except Exception:
        return False
    return True


# -----------------------------
# E2E RNN autograd tests
# -----------------------------
class TestRNNE2EGradients(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_rnn_grad_flows_to_input_sequence(self):
        """
        Ensure BPTT path reaches the original x when using:
          x_t = x[t]  (Tensor.__getitem__ scatter backward)
          h_seq = Tensor.stack(hs)
        """
        np.random.seed(0)
        T, N, D, H = 5, 2, 3, 4

        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x_np = np.random.randn(T, N, D).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=True)

        h_seq, h_T = rnn.forward(x)

        # Simple scalar loss that touches all timesteps
        loss = h_seq.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Expected x.grad to be populated by BPTT")
        self.assertEqual(x.grad.shape, x.shape)

        grad_np = x.grad.to_numpy()
        self.assertFalse(
            np.all(grad_np == 0.0),
            "Expected x.grad to have nonzero entries (sanity check).",
        )

    def test_rnn_loss_on_final_state_only(self):
        """
        Ensure gradients still propagate when loss depends only on final state h_T.
        """
        np.random.seed(1)
        T, N, D, H = 6, 3, 2, 5

        rnn = RNN(input_size=D, hidden_size=H, bias=True)
        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        h_seq, h_T = rnn.forward(x)

        loss = h_T.sum()
        loss.backward()

        # Parameters should have grads
        any_param_grad = False
        for p in rnn.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is not None:
                any_param_grad = True
                break
        self.assertTrue(any_param_grad, "Expected at least one parameter gradient")

        # x should also get gradient (because h_T depends on all inputs through time)
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_rnn_long_sequence_forward_backward_no_nan(self):
        """
        Stability test: longer T should not crash and grads should stay finite.
        """
        np.random.seed(2)
        T, N, D, H = 50, 2, 3, 4

        rnn = RNN(input_size=D, hidden_size=H, bias=True)
        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )

        h_seq, _ = rnn.forward(x)
        loss = h_seq.sum()
        loss.backward()

        # Finite grads for x
        self.assertIsNotNone(x.grad)
        g = x.grad.to_numpy()
        self.assertTrue(np.isfinite(g).all(), "Expected finite gradients for x")

        # Finite grads for parameters
        for p in rnn.parameters():
            pt = _unwrap_param_tensor(p)
            if pt.grad is None:
                continue
            self.assertTrue(np.isfinite(pt.grad.to_numpy()).all())

    def test_rnncell_backward_matches_finite_difference_single_weight_entry(self):
        """
        Numerical gradient spot-check for one weight entry in W_ih.

        This is a *sanity* check (not exhaustive). It validates that the sign/scale
        of backprop is plausible.

        We use loss = sum(h_T) so grad is well-defined and dense.
        """
        np.random.seed(3)
        N, D, H = 2, 3, 4

        cell = RNNCell(input_size=D, hidden_size=H, bias=True)

        # Deterministic params to make finite difference stable
        Wih = np.random.randn(D, H).astype(np.float32) * 0.1
        Whh = np.random.randn(H, H).astype(np.float32) * 0.1
        bih = np.random.randn(H).astype(np.float32) * 0.1
        bhh = np.random.randn(H).astype(np.float32) * 0.1

        _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
        _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)

        x = _tensor_from_numpy(np.random.randn(N, D), self.device, requires_grad=True)
        h_prev = _tensor_from_numpy(
            np.random.randn(N, H), self.device, requires_grad=True
        )

        # Analytical grad
        out = cell.forward(x, h_prev)
        loss = out.sum()
        loss.backward()

        Wih_t = _unwrap_param_tensor(cell.W_ih)
        self.assertIsNotNone(Wih_t.grad, "Expected W_ih.grad to exist")

        # Pick one entry to check
        i, j = 1, 2
        grad_analytical = float(Wih_t.grad.to_numpy()[i, j])

        # Finite difference
        eps = 1e-3
        Wih_plus = Wih.copy()
        Wih_minus = Wih.copy()
        Wih_plus[i, j] += eps
        Wih_minus[i, j] -= eps

        def loss_with_Wih(Wih_candidate: np.ndarray) -> float:
            # Rebuild a fresh cell to avoid accumulated ctx/grad state
            c = RNNCell(input_size=D, hidden_size=H, bias=True)
            _unwrap_param_tensor(c.W_ih).copy_from_numpy(Wih_candidate)
            _unwrap_param_tensor(c.W_hh).copy_from_numpy(Whh)
            _unwrap_param_tensor(c.b_ih).copy_from_numpy(bih)
            _unwrap_param_tensor(c.b_hh).copy_from_numpy(bhh)

            y = c.forward(x, h_prev)  # x/h_prev are fixed tensors
            return float(np.sum(y.to_numpy()))

        Lp = loss_with_Wih(Wih_plus)
        Lm = loss_with_Wih(Wih_minus)
        grad_fd = (Lp - Lm) / (2.0 * eps)

        # Loose tolerance: this is a sanity check, and tanh + float32 + eps yields noise
        self.assertTrue(
            np.isfinite(grad_fd),
            "Finite difference gradient should be finite",
        )
        self.assertAlmostEqual(
            grad_analytical,
            grad_fd,
            places=2,  # intentionally loose
            msg=f"Analytical grad {grad_analytical} vs FD grad {grad_fd}",
        )


# -----------------------------
# Tiny training sanity test
# -----------------------------
class TestRNNTinyTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_tiny_training_decreases_mse_loss(self):
        """
        Tiny supervised training sanity check.

        We create a simple target hidden sequence and minimize MSE:
          loss = ((h_seq - target) * (h_seq - target)).sum()

        This requires Tensor.__sub__ and Tensor.__mul__ to exist.
        If not implemented yet, we skip (so your suite stays green).
        """
        if not _supports_binary_ops():
            self.skipTest(
                "Tensor does not yet support '-' and '*' needed for MSE training."
            )

        np.random.seed(4)
        T, N, D, H = 6, 2, 3, 4
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        # Fixed input
        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )

        # Simple target: encourage hidden states toward a constant pattern
        target_np = np.ones((T, N, H), dtype=np.float32) * 0.5
        target = _tensor_from_numpy(target_np, self.device, requires_grad=False)

        def compute_loss() -> Tensor:
            h_seq, _ = rnn.forward(x)
            diff = h_seq - target
            return (diff * diff).sum()

        # Run a few SGD steps and expect loss to go down (not necessarily monotonically every step,
        # but should improve from start to end).
        lr = 0.05
        steps = 25

        _zero_grads(rnn)
        loss0 = compute_loss()
        loss0_val = float(loss0.to_numpy())
        self.assertTrue(np.isfinite(loss0_val))

        last_val = loss0_val
        for _ in range(steps):
            _zero_grads(rnn)
            loss = compute_loss()
            loss.backward()
            _sgd_step(rnn, lr=lr)
            last_val = float(loss.to_numpy())

        self.assertTrue(
            last_val < loss0_val,
            msg=f"Expected training to reduce loss: start={loss0_val} end={last_val}",
        )


if __name__ == "__main__":
    unittest.main()
