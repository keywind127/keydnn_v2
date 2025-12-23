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


class TestRNNFiniteDifference(TestCase):
    """
    Mathematical gradient check via finite differences.

    This validates that BPTT (as wired by your autograd + slicing + stack)
    produces approximately correct gradients w.r.t. h0.
    """

    def setUp(self) -> None:
        self.device = Device("cpu")

    def _set_deterministic_params(self, rnn: RNN, D: int, H: int) -> None:
        """
        Make parameters deterministic so the finite-difference check is stable.
        """
        # Simple deterministic patterns
        Wih = (np.arange(D * H, dtype=np.float32).reshape(D, H) / 10.0) - 0.2
        Whh = (np.arange(H * H, dtype=np.float32).reshape(H, H) / 15.0) - 0.1

        rnn.cell.W_ih.copy_from_numpy(Wih)
        rnn.cell.W_hh.copy_from_numpy(Whh)

        if rnn.cell.bias:
            bih = np.linspace(-0.05, 0.05, H, dtype=np.float32)
            bhh = np.linspace(0.03, -0.03, H, dtype=np.float32)
            rnn.cell.b_ih.copy_from_numpy(bih)
            rnn.cell.b_hh.copy_from_numpy(bhh)

    def _loss_value(self, rnn: RNN, x_np: np.ndarray, h0_np: np.ndarray) -> float:
        """
        Compute scalar loss L = sum(h_T) using pure forward (no backward).
        """
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h0 = _tensor_from_numpy(h0_np, self.device, requires_grad=False)
        _, h_T = rnn.forward(x, h0=h0)
        return float(h_T.to_numpy().sum())

    def test_rnn_h0_grad_matches_finite_difference(self):
        np.random.seed(123)

        # Keep small for numeric grad check
        T, N, D, H = 3, 2, 2, 2
        eps = 1e-3  # finite-diff step (float32-friendly)

        rnn = RNN(input_size=D, hidden_size=H, bias=True)
        self._set_deterministic_params(rnn, D=D, H=H)

        # Fixed input
        x_np = (np.random.randn(T, N, D)).astype(np.float32)
        h0_np = (np.random.randn(N, H)).astype(np.float32)

        # Autograd gradient
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)
        h0 = _tensor_from_numpy(h0_np, self.device, requires_grad=True)

        # Clear any stale grads on parameters if your engine accumulates
        for p in rnn.parameters():
            p.zero_grad()

        _, h_T = rnn.forward(x, h0=h0)
        loss = h_T.sum()
        loss.backward()

        self.assertIsNotNone(h0.grad, "h0.grad must be populated for grad check")
        g_auto = h0.grad.to_numpy().astype(np.float32)

        # Finite differences gradient
        g_num = np.zeros_like(h0_np, dtype=np.float32)

        base = self._loss_value(
            rnn, x_np, h0_np
        )  # not strictly needed, but useful for debugging

        for i in range(N):
            for j in range(H):
                h_pos = h0_np.copy()
                h_neg = h0_np.copy()
                h_pos[i, j] += eps
                h_neg[i, j] -= eps

                L_pos = self._loss_value(rnn, x_np, h_pos)
                L_neg = self._loss_value(rnn, x_np, h_neg)

                g_num[i, j] = (L_pos - L_neg) / (2.0 * eps)

        # Compare
        # Tolerances: float32 + tanh + finite diff -> allow a bit of slack
        self.assertTrue(
            np.allclose(g_auto, g_num, atol=2e-3, rtol=2e-2),
            msg=(
                "Finite-difference check failed for h0.\n"
                f"base loss={base}\n"
                f"autograd grad:\n{g_auto}\n"
                f"numeric grad:\n{g_num}\n"
                f"diff:\n{g_auto - g_num}\n"
            ),
        )


if __name__ == "__main__":
    unittest.main()
