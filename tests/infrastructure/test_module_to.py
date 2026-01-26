import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure.tensor._tensor import Tensor


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


class TestModuleToAndToInPlace(TestCase):
    def test_module_to_moves_parameters_and_rebinds(self):

        device_cpu = Device("cpu")
        device_cuda = Device("cuda:0")

        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run Module.to() tests because Tensor cannot be loaded from NumPy "
            "(need Tensor(data=...) or from_numpy/copy_from_numpy).",
        )

        class Child(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter((2, 2), device_cpu, requires_grad=True)
                self.register_parameter("w", self.w)

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.p = Parameter((1, 3), device_cpu, requires_grad=True)
                self.register_parameter("p", self.p)
                self.child = Child()
                self.register_module("child", self.child)

            def forward(self, x):
                return x

        m = Parent()

        m.p.fill(1.0)
        m.child.w.fill(2.0)

        p_before = m.p
        w_before = m.child.w
        p_before_id = id(p_before)
        w_before_id = id(w_before)

        try:
            _ = m.p.to(device_cuda, copy=True).to(device_cpu, copy=True)
        except Exception as e:
            self.skipTest(f"CUDA backend not functional for Tensor.to(): {e!r}")

        m.to(device_cuda)

        self.assertTrue(m.p.device.is_cuda())
        self.assertTrue(m.child.w.device.is_cuda())

        self.assertNotEqual(id(m.p), p_before_id)
        self.assertNotEqual(id(m.child.w), w_before_id)

        self.assertIs(m._parameters["p"], m.p)
        self.assertIs(m.child._parameters["w"], m.child.w)

        np.testing.assert_allclose(
            m.p.to(Device("cpu"), copy=True).to_numpy(),
            np.ones((1, 3), dtype=np.float32),
        )
        np.testing.assert_allclose(
            m.child.w.to(Device("cpu"), copy=True).to_numpy(),
            np.full((2, 2), 2.0, dtype=np.float32),
        )

    def test_module_to_in_place_preserves_parameter_identity(self):
        device_cpu = Device("cpu")
        device_cuda = Device("cuda:0")

        class Child(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter((2, 2), device_cpu, requires_grad=True)
                self.register_parameter("w", self.w)

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.p = Parameter((1, 3), device_cpu, requires_grad=True)
                self.register_parameter("p", self.p)
                self.child = Child()
                self.register_module("child", self.child)

            def forward(self, x):
                return x

        m = Parent()
        m.p.fill(3.0)
        m.child.w.fill(4.0)

        p_id = id(m.p)
        w_id = id(m.child.w)

        try:
            _ = m.p.to(device_cuda, copy=True).to(device_cpu, copy=True)
        except Exception as e:
            self.skipTest(f"CUDA backend not functional for Tensor.to(): {e!r}")

        m.to_(device_cuda)

        self.assertEqual(id(m.p), p_id)
        self.assertEqual(id(m.child.w), w_id)

        p_cuda_back = m.p.to(device_cpu, copy=True).to_numpy()
        w_cuda_back = m.child.w.to(device_cpu, copy=True).to_numpy()

        np.testing.assert_allclose(
            p_cuda_back, np.full((1, 3), 3.0, np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            w_cuda_back, np.full((2, 2), 4.0, np.float32), rtol=0, atol=0
        )

        m.to_(device_cpu)

        self.assertEqual(id(m.p), p_id)
        self.assertEqual(id(m.child.w), w_id)

        np.testing.assert_allclose(
            m.p.to_numpy(), np.full((1, 3), 3.0, np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            m.child.w.to_numpy(), np.full((2, 2), 4.0, np.float32), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
