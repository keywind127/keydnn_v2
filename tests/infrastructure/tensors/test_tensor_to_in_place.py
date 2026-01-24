import unittest
import numpy as np

from keydnn.domain.device._device import Device
from keydnn.infrastructure.tensor._tensor import Tensor


class TestTensorToInPlace(unittest.TestCase):
    def test_tensor_to_in_place(self):
        # Create a CPU tensor with known contents
        x = Tensor(
            shape=(2, 3, 4),
            device=Device("cpu"),
            requires_grad=True,
            ctx=None,
            dtype=np.float32,
        )
        data = np.random.randn(*x.shape).astype(np.float32)
        x.copy_from_numpy(data)

        x_id = id(x)
        x_vals_before = x.to_numpy().copy()
        x_dev_before = str(x.device)

        # Same-device no-op should preserve identity, device, and values
        y = x.to_(Device("cpu"))
        self.assertIs(y, x)
        self.assertEqual(id(x), x_id)
        self.assertEqual(str(x.device), x_dev_before)
        np.testing.assert_allclose(x.to_numpy(), x_vals_before, rtol=0, atol=0)

        # Same-device copy=True should still preserve identity and values
        y2 = x.to_(Device("cpu"), copy=True)
        self.assertIs(y2, x)
        self.assertEqual(id(x), x_id)
        self.assertEqual(str(x.device), x_dev_before)
        np.testing.assert_allclose(x.to_numpy(), x_vals_before, rtol=0, atol=0)

        # --- CUDA portion (skip unless CUDA transfer is truly functional) ---
        cuda0 = Device("cuda:0")

        # Probe CUDA by attempting an actual tiny transfer.
        # If this fails (no GPU, driver issue, DLL issue, etc.), skip CUDA checks.
        try:
            probe = x.to(cuda0, copy=True)  # real H2D path
            # Also sanity check D2H round-trip quickly
            _ = probe.to(Device("cpu"), copy=True)
        except Exception as e:
            self.skipTest(f"CUDA backend not functional for Tensor.to(): {e!r}")

        # In-place CPU -> CUDA should preserve identity and update device
        x_before_cuda_vals = x.to_numpy().copy()
        x.to_(cuda0)
        self.assertEqual(id(x), x_id)
        self.assertTrue(x.device.is_cuda())
        self.assertEqual(str(x.device), str(cuda0))

        # CUDA -> CPU should preserve identity and restore values
        x.to_(Device("cpu"))
        self.assertEqual(id(x), x_id)
        self.assertTrue(x.device.is_cpu())
        np.testing.assert_allclose(x.to_numpy(), x_before_cuda_vals, rtol=0, atol=0)

        # Graph break assertions (only if fields exist)
        if hasattr(x, "ctx"):
            self.assertIsNone(getattr(x, "ctx"))
        if hasattr(x, "_ctx"):
            self.assertIsNone(getattr(x, "_ctx"))


if __name__ == "__main__":
    unittest.main()
