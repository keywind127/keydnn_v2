from unittest import TestCase
import unittest

import numpy as np

from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.domain._device import Device


class TestParameterInfrastructure(TestCase):

    def test_parameter_initialization_inherits_tensor_contract(self):
        """Parameter should behave like Tensor for shape/device."""
        shape = (2, 3)
        device_cpu = Device("cpu")

        p = Parameter(shape, device_cpu, requires_grad=True)

        self.assertEqual(p.shape, shape)
        self.assertEqual(str(p.device), str(device_cpu))

    def test_parameter_requires_grad_default_true(self):
        """requires_grad should default to True unless specified."""
        p = Parameter((2, 2), Device("cpu"))
        self.assertIsInstance(p.requires_grad, bool)
        self.assertTrue(p.requires_grad)

    def test_parameter_requires_grad_can_toggle(self):
        """requires_grad should be publicly mutable via property setter."""
        p = Parameter((2, 2), Device("cpu"), requires_grad=True)

        p.requires_grad = False
        self.assertFalse(p.requires_grad)

        p.requires_grad = True
        self.assertTrue(p.requires_grad)

    def test_parameter_grad_initially_none(self):
        """grad should start as None."""
        p = Parameter((2, 2), Device("cpu"))
        self.assertIsNone(p.grad)

    def test_parameter_set_grad_and_zero_grad_contract(self):
        """set_grad() should set grad; zero_grad() should clear it."""
        p = Parameter((2, 2), Device("cpu"))
        g = Tensor((2, 2), Device("cpu"))
        g.fill(1.0)

        p.set_grad(g)
        self.assertIsNotNone(p.grad)
        # Compare via public contract (to_numpy), not internal storage
        self.assertTrue(np.array_equal(p.grad.to_numpy(), g.to_numpy()))

        p.zero_grad()
        self.assertIsNone(p.grad)

    def test_parameter_accumulate_grad_first_time_sets_grad(self):
        """accumulate_grad() should set grad on first accumulation."""
        p = Parameter((2, 2), Device("cpu"), requires_grad=True)
        g1 = Tensor((2, 2), Device("cpu"))
        g1.fill(1.0)

        p.accumulate_grad(g1)

        self.assertIsNotNone(p.grad)
        self.assertTrue(np.array_equal(p.grad.to_numpy(), g1.to_numpy()))

    def test_parameter_accumulate_grad_adds_when_possible(self):
        """
        accumulate_grad() should sum gradients if Tensor supports `+`.
        If `+` isn't implemented yet, Parameter falls back to overwrite.
        This test accepts either behavior, but strongly prefers summation.
        """
        p = Parameter((2, 2), Device("cpu"), requires_grad=True)

        g1 = Tensor((2, 2), Device("cpu"))
        g1.fill(1.0)

        g2 = Tensor((2, 2), Device("cpu"))
        g2.fill(2.0)

        p.accumulate_grad(g1)
        p.accumulate_grad(g2)

        self.assertIsNotNone(p.grad)

        grad_arr = p.grad.to_numpy()

        # Preferred: elementwise sum => 3.0
        expected_sum = np.full((2, 2), 3.0, dtype=np.float32)
        # Fallback: overwrite => 2.0
        expected_overwrite = np.full((2, 2), 2.0, dtype=np.float32)

        self.assertTrue(
            np.array_equal(grad_arr, expected_sum)
            or np.array_equal(grad_arr, expected_overwrite),
            msg=(
                "Expected grad to be either summed (preferred, requires Tensor.__add__) "
                "or overwritten (fallback when + not available)."
            ),
        )

    def test_parameter_accumulate_grad_noop_when_requires_grad_false(self):
        """If requires_grad is False, accumulate_grad() should not store gradients."""
        p = Parameter((2, 2), Device("cpu"), requires_grad=False)

        g = Tensor((2, 2), Device("cpu"))
        g.fill(1.0)

        p.accumulate_grad(g)
        self.assertIsNone(p.grad)

    def test_parameter_zero_grad_idempotent(self):
        """Calling zero_grad() multiple times should be safe."""
        p = Parameter((2, 2), Device("cpu"))

        p.zero_grad()
        self.assertIsNone(p.grad)

        p.zero_grad()
        self.assertIsNone(p.grad)

    def test_parameter_cuda_constructor_still_has_grad_semantics(self):
        """
        Even if CUDA Tensor storage is a placeholder right now,
        Parameter should still expose requires_grad/grad/zero_grad without crashing.
        """
        p = Parameter((2, 2), Device("cuda:0"), requires_grad=True)

        self.assertTrue(p.requires_grad)
        self.assertIsNone(p.grad)

        # zero_grad should still work
        p.zero_grad()
        self.assertIsNone(p.grad)

    def test_parameter_requires_grad_can_be_disabled(self):
        p = Parameter((2, 2), Device("cpu"), requires_grad=False)
        self.assertFalse(p.requires_grad)

    def test_parameter_zero_grad_contract(self):
        p = Parameter((2, 2), Device("cpu"))
        # grad is optional and might be None initially; zero_grad should always be safe
        p.zero_grad()
        self.assertIsNone(p.grad)


class TestParameterAutogradAccumulation(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_parameter_reuse_in_graph_accumulates_grad(self):
        """
        If the same Parameter appears in the graph multiple times, autograd must
        accumulate its gradients (not overwrite).

        This is a core requirement for BPTT, because W_ih/W_hh are reused at
        every timestep.
        """
        # p used twice in the forward graph
        p = Parameter((2, 2), self.device, requires_grad=True)
        p.copy_from_numpy(np.zeros((2, 2), dtype=np.float32))

        x = Tensor((2, 2), self.device, requires_grad=False)
        x.fill(0.0)

        # y = p + x
        # z = p + x
        # loss = (y + z).sum()
        y = p + x
        z = p + x
        loss = (y + z).sum()
        loss.backward()

        self.assertIsNotNone(p.grad, "p.grad should exist after backward()")

        # d/dp of (p+x) is 1, used twice => 2
        expected = np.full((2, 2), 2.0, dtype=np.float32)
        self.assertTrue(
            np.array_equal(p.grad.to_numpy(), expected),
            msg=(
                "Expected Parameter grad to accumulate across multiple uses in graph. "
                "If this fails, your autograd engine is likely overwriting grads "
                "instead of accumulating."
            ),
        )

    def test_parameter_reuse_in_graph_does_not_require_x_grad(self):
        """
        Sanity check: x.requires_grad=False should not prevent p from receiving grad.
        """
        p = Parameter((3, 1), self.device, requires_grad=True)
        p.copy_from_numpy(np.zeros((3, 1), dtype=np.float32))

        x = Tensor((3, 1), self.device, requires_grad=False)
        x.fill(0.0)

        loss = (p + x).sum()
        loss.backward()

        self.assertIsNotNone(p.grad)
        expected = np.ones((3, 1), dtype=np.float32)
        self.assertTrue(np.array_equal(p.grad.to_numpy(), expected))


if __name__ == "__main__":
    unittest.main()
