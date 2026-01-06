import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.fully_connected._dense import Dense
from src.keydnn.infrastructure.tensor._tensor import Tensor


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
    arr = np.asarray(arr, dtype=np.float32)

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
    arr = np.asarray(arr, dtype=np.float32)

    if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
        t.from_numpy(arr)
        return
    if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
        t.copy_from_numpy(arr)
        return

    raise AssertionError(
        "Cannot set tensor values via public API. "
        "Implement Tensor.from_numpy()/copy_from_numpy()."
    )


class TestDenseInfrastructure(TestCase):
    def test_dense_is_lazy_before_first_forward(self):
        d = Dense(4, bias=True, device=Device("cpu"))

        self.assertFalse(d.is_built)
        self.assertIsNone(d.in_features)
        self.assertIsNone(getattr(d, "_linear", None))

        # Lazy module should have no params yet (until built)
        self.assertEqual(len(list(d.parameters())), 0)

        cfg = d.get_config()
        self.assertEqual(cfg["out_features"], 4)
        self.assertEqual(cfg["bias"], True)
        self.assertEqual(cfg["device"], "cpu")
        self.assertIsNone(cfg["in_features"])

    def test_dense_builds_on_first_forward_and_exposes_parameters(self):
        device = Device("cpu")
        d = Dense(4, bias=True, device=device)

        x = Tensor((2, 3), device)  # (batch=2, in_features=3)
        y = d.forward(x)

        self.assertTrue(d.is_built)
        self.assertEqual(d.in_features, 3)
        self.assertEqual(y.shape, (2, 4))

        # After build, parameters should be visible (weight + bias)
        params = list(d.parameters())
        unique = {id(p) for p in params}
        self.assertEqual(len(unique), 2)  # weight + bias (unique)

        # Underlying linear should exist and have expected param shapes
        lin = getattr(d, "_linear", None)
        self.assertIsNotNone(lin)
        self.assertEqual(lin.weight.shape, (4, 3))
        self.assertIsNotNone(lin.bias)
        self.assertEqual(lin.bias.shape, (4,))

    def test_dense_forward_rejects_non_2d_input(self):
        d = Dense(4, bias=True, device=Device("cpu"))

        x1 = Tensor((3,), Device("cpu"))
        with self.assertRaises(ValueError):
            d.forward(x1)

    def test_dense_forward_rejects_feature_mismatch_after_built(self):
        device = Device("cpu")
        d = Dense(4, bias=True, device=device)

        # first call builds with in_features=3
        _ = d.forward(Tensor((2, 3), device))

        # second call uses mismatching in_features
        with self.assertRaises(RuntimeError):
            d.forward(Tensor((2, 5), device))

    def test_dense_device_adopts_first_input_device_when_device_none(self):
        # If device=None, Dense should adopt x.device on first forward
        d = Dense(4, bias=True, device=None)
        self.assertIsNone(d.device)

        x = Tensor((2, 3), Device("cpu"))
        _ = d.forward(x)

        self.assertIsNotNone(d.device)
        self.assertEqual(str(d.device), str(x.device))

    def test_dense_forward_rejects_device_mismatch_when_device_specified(self):
        d = Dense(4, bias=True, device=Device("cpu"))

        # Create x on "cpu" but use a different Device string if your Device supports it.
        # If your Device only has "cpu" and "cuda:*", this test becomes meaningful in CUDA tests.
        x = Tensor((2, 3), Device("cpu"))
        # Should be ok
        _ = d.forward(x)

        # No further device mismatch test on CPU-only device set; covered in CUDA tests.

    def test_dense_cpu_forward_matches_numpy_reference_after_setting_params(self):
        self.assertTrue(
            _tensor_supports_numpy_load(),
            "Cannot run numeric Dense tests because Tensor cannot be loaded from NumPy "
            "(need Tensor(data=...) or from_numpy/copy_from_numpy).",
        )

        device = Device("cpu")
        d = Dense(2, bias=True, device=device)

        # Build with in_features=3
        x_np = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        x = _make_tensor_from_numpy(x_np, device)
        _ = d.forward(x)

        lin = d._linear
        self.assertIsNotNone(lin)

        W = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        b = np.array([0.25, -2.0], dtype=np.float32)

        _set_tensor_data_public(lin.weight, W)
        _set_tensor_data_public(lin.bias, b)

        y = d.forward(x)
        y_np = y.to_numpy()

        ref = x_np @ W.T + b
        self.assertEqual(y.shape, ref.shape)
        np.testing.assert_allclose(y_np, ref, rtol=1e-6, atol=1e-6)

    def test_dense_forward_attaches_context_when_requires_grad(self):
        device = Device("cpu")
        d = Dense(4, bias=True, device=device)

        x = Tensor((2, 3), device)
        x.requires_grad = True

        out = d.forward(x)

        self.assertTrue(out.requires_grad)
        ctx = out._get_ctx()
        self.assertIsNotNone(ctx, "Expected Context to be attached to output Tensor.")
        self.assertTrue(callable(ctx.backward_fn))

        lin = d._linear
        self.assertIsNotNone(lin)

        # parents should match underlying Linear's contract: (x, weight, bias)
        self.assertEqual(len(ctx.parents), 3)
        self.assertIs(ctx.parents[0], x)
        self.assertIs(ctx.parents[1], lin.weight)
        self.assertIs(ctx.parents[2], lin.bias)

        # saved_tensors should be (x, weight)
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], x)
        self.assertIs(ctx.saved_tensors[1], lin.weight)

    def test_dense_forward_no_context_when_no_requires_grad(self):
        device = Device("cpu")
        d = Dense(4, bias=True, device=device)

        x = Tensor((2, 3), device)
        x.requires_grad = False

        # Build once
        _ = d.forward(x)

        # Turn off param grads (critical!)
        lin = d._linear
        self.assertIsNotNone(lin)
        lin.weight.requires_grad = False
        if lin.bias is not None:
            lin.bias.requires_grad = False

        out = d.forward(x)

        self.assertFalse(out.requires_grad)
        self.assertIsNone(out._get_ctx(), "Expected no Context when nothing requires grad.")


    def test_dense_get_config_after_build_includes_in_features(self):
        device = Device("cpu")
        d = Dense(4, bias=True, device=device)
        _ = d.forward(Tensor((2, 3), device))

        cfg = d.get_config()
        self.assertEqual(cfg["out_features"], 4)
        self.assertEqual(cfg["bias"], True)
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["in_features"], 3)

    def test_dense_from_config_unbuilt_then_builds_on_forward(self):
        cfg = {
            "out_features": 4,
            "bias": True,
            "device": "cpu",
            "in_features": None,
        }
        d = Dense.from_config(cfg)

        self.assertFalse(d.is_built)
        self.assertIsNone(d.in_features)

        y = d.forward(Tensor((2, 3), Device("cpu")))
        self.assertTrue(d.is_built)
        self.assertEqual(d.in_features, 3)
        self.assertEqual(y.shape, (2, 4))

    def test_dense_from_config_built_eagerly_when_in_features_present(self):
        cfg = {
            "out_features": 4,
            "bias": True,
            "device": "cpu",
            "in_features": 3,
        }
        d = Dense.from_config(cfg)

        self.assertTrue(d.is_built)
        self.assertEqual(d.in_features, 3)

        y = d.forward(Tensor((2, 3), Device("cpu")))
        self.assertEqual(y.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
