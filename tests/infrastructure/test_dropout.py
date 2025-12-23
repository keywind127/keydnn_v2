import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.layers._dropout import Dropout
from src.keydnn.infrastructure._tensor import Tensor


def _tensor_supports_numpy_load() -> bool:
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        pass

    t = Tensor((1, 1), Device("cpu"))
    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        return True
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        return True

    return False


def _make_tensor_from_numpy(arr: np.ndarray, device: Device) -> Tensor:
    arr = arr.astype(np.float32, copy=False)

    try:
        return Tensor(data=arr, device=device)
    except TypeError:
        t = Tensor(arr.shape, device)
        if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
            t.from_numpy(arr)
            return t
        if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
            t.copy_from_numpy(arr)
            return t

        raise AssertionError(
            "Tensor cannot be loaded with NumPy data via public APIs. "
            "Implement Tensor(data=...) OR Tensor.from_numpy()/copy_from_numpy()."
        )


class TestDropout(TestCase):
    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            Dropout(p=-0.1)
        with self.assertRaises(ValueError):
            Dropout(p=1.0)

        Dropout(p=0.0)
        Dropout(p=0.999)

    def test_eval_mode_is_identity(self):
        device = Device("cpu")
        x_np = np.random.randn(4, 5).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, device)

        d = Dropout(p=0.75)
        d.training = False

        y = d(x)

        # by design, dropout returns x directly in eval mode
        self.assertIs(y, x)
        np.testing.assert_allclose(y.to_numpy(), x_np, rtol=0.0, atol=0.0)

    def test_p_zero_is_noop_even_in_train(self):
        device = Device("cpu")
        x_np = np.random.randn(3, 3).astype(np.float32)
        x = _make_tensor_from_numpy(x_np, device)

        d = Dropout(p=0.0)
        d.training = True

        y = d(x)

        self.assertIs(y, x)
        np.testing.assert_allclose(y.to_numpy(), x_np, rtol=0.0, atol=0.0)

    def test_forward_is_deterministic_with_seed(self):
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric Dropout tests because Tensor cannot be loaded from NumPy.",
        )

        device = Device("cpu")
        x = _make_tensor_from_numpy(np.ones((6, 6), dtype=np.float32), device)

        d = Dropout(p=0.5)
        d.training = True

        np.random.seed(42)
        y1 = d(x).to_numpy().copy()

        np.random.seed(42)
        y2 = d(x).to_numpy().copy()

        np.testing.assert_allclose(y1, y2, rtol=0.0, atol=0.0)

    def test_backward_masked_and_scaled(self):
        device = Device("cpu")

        p = 0.5
        scale = 1.0 / (1.0 - p)

        x = _make_tensor_from_numpy(np.ones((8, 8), dtype=np.float32), device)
        x.requires_grad = True

        d = Dropout(p=p)
        d.training = True

        np.random.seed(7)
        y = d(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

        g = x.grad.to_numpy()
        uniq = np.unique(g)

        for val in uniq:
            ok = np.isclose(val, 0.0) or np.isclose(val, scale)
            self.assertTrue(
                ok, msg=f"Unexpected grad value {val}; expected 0 or {scale}"
            )

        # likely both dropped and kept are present
        self.assertTrue(np.any(np.isclose(g, 0.0)))
        self.assertTrue(np.any(np.isclose(g, scale)))

    def test_expected_mean_is_approximately_one_for_ones_input(self):
        device = Device("cpu")

        p = 0.2
        x = _make_tensor_from_numpy(np.ones((200, 200), dtype=np.float32), device)

        d = Dropout(p=p)
        d.training = True

        np.random.seed(1234)
        y = d(x).to_numpy()

        self.assertTrue(abs(float(y.mean()) - 1.0) < 0.03)

    def test_get_config_and_from_config_roundtrip(self):
        d1 = Dropout(p=0.3)
        cfg = d1.get_config()

        self.assertIn("p", cfg)
        self.assertAlmostEqual(cfg["p"], 0.3)

        d2 = Dropout.from_config(cfg)
        self.assertIsInstance(d2, Dropout)
        self.assertAlmostEqual(d2.p, 0.3)

        # runtime state not serialized
        self.assertTrue(d2.training)


if __name__ == "__main__":
    unittest.main()
