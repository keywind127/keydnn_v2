from __future__ import annotations

import unittest
import numpy as np


class TestSequentialXORTraining(unittest.TestCase):
    def test_xor_training_one_hidden_layer(self):
        # Only skip when components truly cannot be imported.
        try:
            from src.keydnn.infrastructure._models import Sequential
            from src.keydnn.infrastructure._linear import Linear
            from src.keydnn.infrastructure._activations import Sigmoid
            from src.keydnn.infrastructure.tensor._tensor import Tensor
            from src.keydnn.domain.device._device import Device
            from src.keydnn.infrastructure._optimizers import SGD
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"XOR training test skipped (missing import): {e}")

        # ---------------- Dataset ----------------
        x_np = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        y_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

        device = Device("cpu")

        x = Tensor(shape=x_np.shape, device=device)
        x.copy_from_numpy(x_np)

        y = Tensor(shape=y_np.shape, device=device)
        y.copy_from_numpy(y_np)

        # Make sure our inputs are actually tensors (helps catch silent numpy fallbacks)
        self.assertTrue(hasattr(x, "copy_from_numpy"))
        self.assertTrue(hasattr(y, "copy_from_numpy"))

        # ---------------- Model ----------------
        hidden_dim = 8
        model = Sequential(
            Linear(2, hidden_dim),
            Sigmoid(),
            Linear(hidden_dim, 1),
            Sigmoid(),
        )

        # ---------------- Loss (MSE) ----------------
        def mse(pred, target):
            diff = pred - target
            sq = diff * diff

            # Critical: ensure loss remains a Tensor-like object, not a python float.
            if hasattr(sq, "mean"):
                return sq.mean()
            if hasattr(sq, "sum"):
                return sq.sum() * (1.0 / target.shape[0])

            raise AttributeError("Tensor must implement mean() or sum()")

        # ---------------- Optimizer ----------------
        opt = SGD(model.parameters(), lr=1.0)
        self.assertTrue(hasattr(opt, "step"), "SGD must implement step().")

        # ---------------- Training loop ----------------
        epochs = 1000

        for _ in range(epochs):
            pred = model(x)
            loss = mse(pred, y)

            # Do NOT skip here; fail with a useful message.
            self.assertTrue(
                hasattr(loss, "backward"),
                f"Autograd entry point missing: Tensor.backward() not found on type={type(loss)}. "
                f"Implement Tensor.backward() (and graph traversal) to enable training.",
            )

            # print("Tensor has backward:", hasattr(type(loss), "backward"))
            # print("Instance has backward:", hasattr(loss, "backward"))
            # print(
            #     "Tensor methods sample:", [m for m in dir(loss) if "back" in m.lower()]
            # )

            loss.backward()
            opt.step()

            # Clear gradients
            # Clear gradients
            if hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                for p in model.parameters():
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()

        # ---------------- Evaluation ----------------
        pred = model(x)

        self.assertTrue(
            hasattr(pred, "to_numpy"),
            f"Expected prediction to support to_numpy(); got type={type(pred)}",
        )

        pred_np = pred.to_numpy()
        y_hat = (pred_np >= 0.5).astype(np.float32)
        acc = float((y_hat == y_np).mean())

        # print(f"XOR accuracy: {acc:.3f}, predictions={pred_np.reshape(-1).tolist()}")

        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low: {acc:.3f}, predictions={pred_np.reshape(-1).tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
