import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure._linear import Linear
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure.tensor._tensor import Tensor


def _tensor_supports_data_ctor() -> bool:
    """Return True if Tensor can be constructed with data=..."""
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        return False


def _tensor_supports_numpy_load() -> bool:
    """
    Return True if we can create a Tensor with known numeric contents
    using ONLY public APIs.
    """
    # Option 1: data ctor
    try:
        _ = Tensor(data=np.zeros((1, 1), dtype=np.float32), device=Device("cpu"))
        return True
    except TypeError:
        pass

    # Option 2: shape ctor + public load method
    t = Tensor((1, 1), Device("cpu"))
    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        return True
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        return True

    return False


def _make_tensor_from_numpy(arr: np.ndarray, device: Device) -> Tensor:
    """
    Construct a Tensor holding arr using public APIs only.
    Prefers data=..., otherwise uses from_numpy/copy_from_numpy.
    """
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


def _set_tensor_data_public(t: Tensor, arr: np.ndarray) -> None:
    arr = arr.astype(np.float32, copy=False)

    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        t.from_numpy(arr)
        return
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        t.copy_from_numpy(arr)
        return

    # If you can't load arbitrary arrays, correctness tests can't be deterministic.
    raise AssertionError(
        "Cannot set parameter tensor values via public API. "
        "Implement Tensor.from_numpy()/copy_from_numpy() (recommended) or allow Parameter(data=...)."
    )


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

    def test_linear_forward_cpu_outputs_expected_shape_and_values(self):
        """
        Verify Linear forward numeric output with deterministic params:
        y = x @ 0 + 0 = 0
        """
        device = Device("cpu")
        lin = Linear(3, 4, bias=True, device=device)

        lin.weight.fill(0.0)
        lin.bias.fill(0.0)

        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = _make_tensor_from_numpy(x_np, device)

        y = lin.forward(x)

        self.assertEqual(y.shape, (2, 4))
        np.testing.assert_allclose(
            y.to_numpy(), np.zeros((2, 4), dtype=np.float32), rtol=0, atol=0
        )

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

    def test_linear_forward_cpu_matches_numpy_reference(self):
        # Fail loudly if framework can't do numeric tests yet
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric Linear tests because Tensor cannot be loaded from NumPy "
            "(need Tensor(data=...) or from_numpy/copy_from_numpy).",
        )

        device = Device("cpu")
        lin = Linear(3, 2, bias=True, device=device)

        W = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        _set_tensor_data_public(lin.weight, W)
        _set_tensor_data_public(lin.bias, b)

        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _make_tensor_from_numpy(x_np, device)

        y = lin.forward(x)
        y_np = y.to_numpy()

        ref = x_np @ W.T + b
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y_np, ref, rtol=1e-6, atol=1e-6)

    def test_linear_forward_attaches_context_when_requires_grad(self):
        device = Device("cpu")
        lin = Linear(3, 4, bias=True, device=device)

        # Ensure params require grad (Linear init already sets requires_grad=True, but be explicit)
        lin.weight.requires_grad = True
        lin.bias.requires_grad = True

        x_np = np.ones((2, 3), dtype=np.float32)
        x = _make_tensor_from_numpy(x_np, device)
        x.requires_grad = True  # trigger ctx attach

        out = lin.forward(x)

        # 1) requires_grad should propagate to output
        self.assertTrue(out.requires_grad)

        # 2) ctx should be attached and retrievable via internal hook
        ctx = out._get_ctx()
        self.assertIsNotNone(ctx, "Expected Context to be attached to output Tensor.")
        self.assertTrue(callable(ctx.backward_fn))

        # 3) parents should be [x, weight, bias] in that exact order
        self.assertEqual(len(ctx.parents), 3)
        self.assertIs(ctx.parents[0], x)
        self.assertIs(ctx.parents[1], lin.weight)
        self.assertIs(ctx.parents[2], lin.bias)

        # 4) saved_tensors should be exactly (x, weight) in that exact order
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], x)
        self.assertIs(ctx.saved_tensors[1], lin.weight)

        # 5) (Optional) backward_fn returns grads aligned with parents & correct shapes
        grad_out = _make_tensor_from_numpy(np.ones(out.shape, dtype=np.float32), device)
        grads = ctx.backward_fn(grad_out)

        self.assertEqual(len(grads), 3)

        grad_x, grad_w, grad_b = grads
        self.assertIsNotNone(grad_x)
        self.assertIsNotNone(grad_w)
        self.assertIsNotNone(grad_b)

        self.assertEqual(grad_x.shape, x.shape)  # (batch, in_features)
        self.assertEqual(grad_w.shape, lin.weight.shape)  # (out_features, in_features)
        self.assertEqual(grad_b.shape, lin.bias.shape)  # (out_features,)

    def test_linear_forward_no_context_when_no_requires_grad(self):
        device = Device("cpu")
        lin = Linear(3, 4, bias=True, device=device)

        # Turn off grad tracking
        lin.weight.requires_grad = False
        lin.bias.requires_grad = False

        x = _make_tensor_from_numpy(np.ones((2, 3), dtype=np.float32), device)
        x.requires_grad = False

        out = lin.forward(x)

        self.assertFalse(out.requires_grad)
        self.assertIsNone(
            out._get_ctx(), "Expected no Context when nothing requires grad."
        )


if __name__ == "__main__":
    unittest.main()
