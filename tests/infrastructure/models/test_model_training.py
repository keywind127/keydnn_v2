from __future__ import annotations

import unittest
import numpy as np


class TestSequentialXORTraining(unittest.TestCase):
    def test_xor_training_one_hidden_layer(self):

        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
            from src.keydnn.infrastructure.activations._modules import Sigmoid
            from src.keydnn.infrastructure.tensor._tensor import Tensor
            from src.keydnn.domain.device._device import Device
            from src.keydnn.infrastructure.optimizers._sgd import SGD
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"XOR training test skipped (missing import): {e}")

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

        self.assertTrue(hasattr(x, "copy_from_numpy"))
        self.assertTrue(hasattr(y, "copy_from_numpy"))

        hidden_dim = 8
        model = Sequential(
            Linear(2, hidden_dim),
            Sigmoid(),
            Linear(hidden_dim, 1),
            Sigmoid(),
        )

        def mse(pred, target):
            diff = pred - target
            sq = diff * diff

            if hasattr(sq, "mean"):
                return sq.mean()
            if hasattr(sq, "sum"):
                return sq.sum() * (1.0 / target.shape[0])

            raise AttributeError("Tensor must implement mean() or sum()")

        opt = SGD(model.parameters(), lr=1.0)
        self.assertTrue(hasattr(opt, "step"), "SGD must implement step().")

        epochs = 800

        model.build(x[:1])

        for _ in range(epochs):
            pred = model(x)
            loss = mse(pred, y)

            self.assertTrue(
                hasattr(loss, "backward"),
                f"Autograd entry point missing: Tensor.backward() not found on type={type(loss)}. "
                f"Implement Tensor.backward() (and graph traversal) to enable training.",
            )

            loss.backward()
            opt.step()

            if hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                for p in model.parameters():
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()

        pred = model(x)

        self.assertTrue(
            hasattr(pred, "to_numpy"),
            f"Expected prediction to support to_numpy(); got type={type(pred)}",
        )

        pred_np = pred.to_numpy()
        y_hat = (pred_np >= 0.5).astype(np.float32)
        acc = float((y_hat == y_np).mean())

        self.assertGreaterEqual(
            acc,
            0.99,
            f"XOR accuracy too low: {acc:.3f}, predictions={pred_np.reshape(-1).tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
