import unittest
import numpy as np

from keydnn.domain.device._device import Device
from keydnn.infrastructure.tensor._tensor import Tensor


class TestTensorToInPlace(unittest.TestCase):
    def test_tensor_to_in_place(self):

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

        y = x.to_(Device("cpu"))
        self.assertIs(y, x)
        self.assertEqual(id(x), x_id)
        self.assertEqual(str(x.device), x_dev_before)
        np.testing.assert_allclose(x.to_numpy(), x_vals_before, rtol=0, atol=0)

        y2 = x.to_(Device("cpu"), copy=True)
        self.assertIs(y2, x)
        self.assertEqual(id(x), x_id)
        self.assertEqual(str(x.device), x_dev_before)
        np.testing.assert_allclose(x.to_numpy(), x_vals_before, rtol=0, atol=0)

        cuda0 = Device("cuda:0")

        try:
            probe = x.to(cuda0, copy=True)

            _ = probe.to(Device("cpu"), copy=True)
        except Exception as e:
            self.skipTest(f"CUDA backend not functional for Tensor.to(): {e!r}")

        x_before_cuda_vals = x.to_numpy().copy()
        x.to_(cuda0)
        self.assertEqual(id(x), x_id)
        self.assertTrue(x.device.is_cuda())
        self.assertEqual(str(x.device), str(cuda0))

        x.to_(Device("cpu"))
        self.assertEqual(id(x), x_id)
        self.assertTrue(x.device.is_cpu())
        np.testing.assert_allclose(x.to_numpy(), x_before_cuda_vals, rtol=0, atol=0)

        if hasattr(x, "ctx"):
            self.assertIsNone(getattr(x, "ctx"))
        if hasattr(x, "_ctx"):
            self.assertIsNone(getattr(x, "_ctx"))


if __name__ == "__main__":
    unittest.main()
