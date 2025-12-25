from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.recurrent._rnn_module import RNN


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    """
    Match your Conv2d tests: treat Parameter as Tensor-like
    or unwrap .data/.tensor if you later change design.
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


def _sgd_step(module, lr: float) -> None:
    """
    Minimal SGD update:
      p <- p - lr * p.grad
    Uses only to_numpy/copy_from_numpy.
    """
    for p in module.parameters():
        pt = _unwrap_param_tensor(p)

        # Frozen params or params without grad: skip
        if not getattr(pt, "requires_grad", False):
            continue
        if getattr(pt, "grad", None) is None:
            continue

        w = pt.to_numpy()
        g = pt.grad.to_numpy()
        pt.copy_from_numpy((w - lr * g).astype(np.float32))


class TestRNNTinyTraining(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_rnn_tiny_training_loss_decreases(self):
        """
        Tiny training sanity test:
        - fixed small input sequence (time-major)
        - objective: minimize sum of all hidden states (h_seq.sum())
        This is a "does SGD move the loss in the right direction?" test.
        """
        np.random.seed(0)

        T, N, D, H = 6, 4, 3, 5
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        # Positive-ish inputs to make initial loss usually > 0 (more stable)
        x_np = np.random.uniform(0.0, 0.2, size=(T, N, D)).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        lr = 5e-2
        steps = 120

        # initial loss
        _zero_grads(rnn)
        h_seq0, _ = rnn.forward(x)
        loss0 = h_seq0.sum()
        loss0_val = float(loss0.to_numpy())

        # train
        last_val = loss0_val
        for _ in range(steps):
            _zero_grads(rnn)
            h_seq, _ = rnn.forward(x)
            loss = h_seq.sum()
            loss.backward()
            _sgd_step(rnn, lr)
            last_val = float(loss.to_numpy())

        # final loss
        _zero_grads(rnn)
        h_seqF, _ = rnn.forward(x)
        lossF = h_seqF.sum()
        lossF_val = float(lossF.to_numpy())

        # Assert: loss decreased by a meaningful margin.
        # Use a relative threshold, but guard against tiny initial value.
        self.assertTrue(
            lossF_val < loss0_val - 1e-4 or lossF_val < 0.8 * loss0_val,
            msg=f"Expected loss to decrease. loss0={loss0_val:.6f}, lossF={lossF_val:.6f}, last_step={last_val:.6f}",
        )

    def test_rnn_tiny_training_updates_parameters(self):
        """
        Ensure training actually changes parameters (not just computes grads).
        We snapshot parameters before and after a few SGD steps and expect at least one
        parameter tensor to change.
        """
        np.random.seed(1)

        T, N, D, H = 4, 2, 3, 4
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x_np = np.random.uniform(0.0, 0.2, size=(T, N, D)).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        # Snapshot params
        before = []
        for p in rnn.parameters():
            pt = _unwrap_param_tensor(p)
            before.append(pt.to_numpy().copy())

        # Train a bit
        lr = 5e-2
        steps = 10
        for _ in range(steps):
            _zero_grads(rnn)
            h_seq, _ = rnn.forward(x)
            loss = h_seq.sum()
            loss.backward()
            _sgd_step(rnn, lr)

        # Snapshot after
        after = []
        for p in rnn.parameters():
            pt = _unwrap_param_tensor(p)
            after.append(pt.to_numpy().copy())

        # At least one parameter must change
        any_changed = False
        for b, a in zip(before, after):
            if not np.array_equal(b, a):
                any_changed = True
                break

        self.assertTrue(
            any_changed,
            msg="Expected at least one parameter to update after SGD steps.",
        )

    def test_rnn_tiny_training_grads_are_finite(self):
        """
        Simple numerical stability check: gradients shouldn't be NaN/Inf.
        """
        np.random.seed(2)

        T, N, D, H = 5, 3, 2, 4
        rnn = RNN(input_size=D, hidden_size=H, bias=True)

        x_np = np.random.uniform(0.0, 0.2, size=(T, N, D)).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        _zero_grads(rnn)
        h_seq, _ = rnn.forward(x)
        loss = h_seq.sum()
        loss.backward()

        for p in rnn.parameters():
            pt = _unwrap_param_tensor(p)
            if getattr(pt, "grad", None) is None:
                continue
            g = pt.grad.to_numpy()
            self.assertTrue(
                np.isfinite(g).all(), msg="Found NaN/Inf in parameter gradients."
            )


if __name__ == "__main__":
    unittest.main()
