import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure._linear import Linear
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure._tensor import Tensor


def _tensor_supports_data_ctor() -> bool:
    """Return True if Tensor can be constructed with data=..."""
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        return False


class TestModuleInfrastructure(TestCase):
    def test_register_parameter_and_parameters_contract(self):
        m = Module()
        p = Parameter((2, 2), Device("cpu"), requires_grad=True)

        # Register and enumerate via public API
        m.register_parameter("p", p)

        params = list(m.parameters())
        self.assertEqual(len(params), 1)
        self.assertIs(params[0], p)

    def test_register_parameter_none_is_noop(self):
        m = Module()

        m.register_parameter("bias", None)

        self.assertEqual(len(list(m.parameters())), 0)

    def test_named_parameters_contract(self):
        m = Module()
        p1 = Parameter((2, 2), Device("cpu"))
        p2 = Parameter((3, 1), Device("cpu"))

        m.register_parameter("p1", p1)
        m.register_parameter("p2", p2)

        named = dict(m.named_parameters())
        self.assertIn("p1", named)
        self.assertIn("p2", named)
        self.assertIs(named["p1"], p1)
        self.assertIs(named["p2"], p2)

    def test_call_delegates_to_forward(self):
        class Identity(Module):
            def forward(self, x):
                return x

        m = Identity()
        x = Tensor((2, 3), Device("cpu"))

        y = m(x)
        self.assertIs(y, x)


class TestLinearInfrastructure(TestCase):
    def test_linear_exposes_parameters(self):
        lin1 = Linear(3, 4, bias=True, device=Device("cpu"))
        params1 = list(lin1.parameters())
        self.assertEqual(len(params1), 2)  # weight + bias

        lin2 = Linear(3, 4, bias=False, device=Device("cpu"))
        params2 = list(lin2.parameters())
        self.assertEqual(len(params2), 1)  # weight only

    def test_linear_forward_rejects_non_2d_input(self):
        lin = Linear(3, 4, device=Device("cpu"))

        x = Tensor((3,), Device("cpu"))  # 1D
        with self.assertRaises(ValueError):
            lin.forward(x)

    def test_linear_forward_rejects_feature_mismatch(self):
        lin = Linear(3, 4, device=Device("cpu"))

        x = Tensor((2, 5), Device("cpu"))  # in_features mismatch
        with self.assertRaises(ValueError):
            lin.forward(x)

    @unittest.skipUnless(
        _tensor_supports_data_ctor(),
        "Tensor(data=..., device=...) not supported; skip Linear forward numeric test.",
    )
    def test_linear_forward_cpu_outputs_expected_shape_and_values(self):
        """
        If Tensor supports data construction, verify Linear forward numeric output.

        We force weight and bias to zeros for deterministic output:
        y = x @ 0 + 0 = 0
        """
        lin = Linear(3, 4, bias=True, device=Device("cpu"))

        # Deterministic parameters via public API (fill)
        lin.weight.fill(0.0)
        lin.bias.fill(0.0)

        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = Tensor(data=x_np, device=Device("cpu"))

        y = lin.forward(x)

        self.assertEqual(y.shape, (2, 4))
        y_np = y.to_numpy()
        self.assertTrue(np.all(y_np == 0.0))

    def test_linear_parameters_count(self):
        lin_bias = Linear(3, 4, bias=True, device=Device("cpu"))
        self.assertEqual(len(list(lin_bias.parameters())), 2)

        lin_nobias = Linear(3, 4, bias=False, device=Device("cpu"))
        self.assertEqual(len(list(lin_nobias.parameters())), 1)

    def test_linear_forward_shape_validation(self):
        lin = Linear(3, 4, device=Device("cpu"))

        with self.assertRaises(ValueError):
            lin.forward(Tensor((3,), Device("cpu")))  # not 2D

        with self.assertRaises(ValueError):
            lin.forward(Tensor((2, 5), Device("cpu")))  # feature mismatch


if __name__ == "__main__":
    unittest.main()
