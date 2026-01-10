from __future__ import annotations

import unittest
import numpy as np


def _cuda_available() -> bool:
    """
    Best-effort CUDA availability check.
    We only return True if your native CUDA DLL/wrappers can be imported and loaded.
    """
    try:
        # Pick any known-good loader you already have in the repo.
        # If you have multiple, choose the most stable one.
        from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


@unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
class TestSequentialXORTrainingCUDA(unittest.TestCase):
    def test_xor_training_one_hidden_layer_cuda(self):
        # Only skip when components truly cannot be imported.
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
            from src.keydnn.infrastructure._activations import Sigmoid
            from src.keydnn.infrastructure.tensor._tensor import Tensor
            from src.keydnn.domain.device._device import Device
            from src.keydnn.infrastructure._optimizers import SGD
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"XOR CUDA training test skipped (missing import): {e}")

        # ---------------- Dataset ----------------
        x_np = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        y_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

        device = Device("cuda:0")

        x = Tensor(shape=x_np.shape, device=device)
        x.copy_from_numpy(x_np)

        y = Tensor(shape=y_np.shape, device=device)
        y.copy_from_numpy(y_np)

        # --- Guardrails: ensure we're truly on CUDA and not silently CPU ---
        self.assertTrue(
            hasattr(x, "device") and str(x.device) == "cuda:0",
            f"x is not on cuda:0 (silent CPU fallback?). x.device={getattr(x, 'device', None)}",
        )
        self.assertTrue(
            hasattr(y, "device") and str(y.device) == "cuda:0",
            f"y is not on cuda:0 (silent CPU fallback?). y.device={getattr(y, 'device', None)}",
        )

        # Optional stronger checks if your Tensor exposes device pointer / cuda alloc
        # (won't fail if you don't have these attributes)
        if hasattr(x, "_cuda_ptr"):
            self.assertNotEqual(
                getattr(x, "_cuda_ptr"), 0, "x CUDA pointer not allocated"
            )
        if hasattr(y, "_cuda_ptr"):
            self.assertNotEqual(
                getattr(y, "_cuda_ptr"), 0, "y CUDA pointer not allocated"
            )

        # ---------------- Model ----------------
        hidden_dim = 8
        model: Sequential = Sequential(
            Linear(2, hidden_dim, device=device),
            Sigmoid(),
            Linear(hidden_dim, 1, device=device),
            Sigmoid(),
        )

        # If your framework requires explicit device move for modules/params, do it here.
        # Try model.to(device) if you have it; otherwise do nothing and let ops enforce device.
        # if hasattr(model, "to"):
        #     model.to(device)  # type: ignore[attr-defined]

        # for p in model.layers():
        #     p.to(device)

        # Verify parameters (if available) are on CUDA too (catches partial moves).
        if hasattr(model, "parameters"):
            for i, p in enumerate(model.parameters()):
                if hasattr(p, "device"):
                    self.assertEqual(
                        str(p.device),
                        "cuda:0",
                        f"param[{i}] not on cuda:0; got {p.device}",
                    )

        # ---------------- Loss (MSE) ----------------
        def mse(pred, target):
            diff = pred - target
            sq = diff * diff

            # ensure loss remains Tensor-like, not python float
            if hasattr(sq, "mean"):
                return sq.mean()
            if hasattr(sq, "sum"):
                return sq.sum() * (1.0 / target.shape[0])

            raise AttributeError("Tensor must implement mean() or sum()")

        # ---------------- Optimizer ----------------
        opt = SGD(model.parameters(), lr=1.0)
        self.assertTrue(hasattr(opt, "step"), "SGD must implement step().")

        # ---------------- Training loop ----------------
        # CUDA can be numerically a bit different; allow more epochs if needed.
        epochs = 2000

        for _ in range(epochs):
            pred = model(x)

            # Guardrail: pred should stay on CUDA
            if hasattr(pred, "device"):
                self.assertEqual(
                    str(pred.device),
                    "cuda:0",
                    f"pred not on cuda:0 (silent CPU fallback?). pred.device={pred.device}",
                )

            loss = mse(pred, y)

            self.assertTrue(
                hasattr(loss, "backward"),
                f"Autograd entry point missing: Tensor.backward() not found on type={type(loss)}.",
            )

            loss.backward()
            opt.step()

            # Clear gradients
            if hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                for p in model.parameters():
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()

        # ---------------- Evaluation ----------------
        pred = model(x)

        # We evaluate on CPU numpy (common pattern even for CUDA backends).
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
            f"XOR CUDA accuracy too low: {acc:.3f}, predictions={pred_np.reshape(-1).tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
