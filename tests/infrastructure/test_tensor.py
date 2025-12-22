from unittest import TestCase
import unittest
import numpy as np

from src.keydnn.infrastructure._tensor import Tensor, Context
from src.keydnn.domain._device import Device


class TestTensorInfrastructure(TestCase):

    def test_tensor_initialization_properties(self):
        shape = (2, 3)
        device_cpu = Device("cpu")
        device_cuda = Device("cuda:0")

        tensor_cpu = Tensor(shape, device_cpu)
        self.assertEqual(tensor_cpu.shape, shape)
        self.assertEqual(str(tensor_cpu.device), str(device_cpu))

        tensor_cuda = Tensor(shape, device_cuda)
        self.assertEqual(tensor_cuda.shape, shape)
        self.assertEqual(str(tensor_cuda.device), str(device_cuda))

    def test_cpu_tensor_to_numpy_contract(self):
        """CPU tensor should allocate float32 ndarray, correct shape, initialized to zeros."""
        shape = (2, 3)
        tensor = Tensor(shape, Device("cpu"))

        arr = tensor.to_numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(arr == 0.0))

    def test_cpu_tensor_fill_contract(self):
        """fill() should update CPU tensor values without direct storage access."""
        shape = (2, 2)
        tensor = Tensor(shape, Device("cpu"))

        tensor.fill(1.0)

        expected = np.ones(shape, dtype=np.float32)
        self.assertTrue(np.array_equal(tensor.to_numpy(), expected))

    def test_cuda_tensor_debug_repr_contract(self):
        """CUDA placeholder should have a stable debug representation."""
        shape = (2, 3)
        tensor = Tensor(shape, Device("cuda:0"))

        s = tensor.debug_storage_repr()
        self.assertIsInstance(s, str)
        self.assertIn("CUDA Tensor on device", s)
        self.assertIn("0", s)
        self.assertIn(str(shape), s)

    def test_to_numpy_on_cuda_raises(self):
        tensor = Tensor((2, 3), Device("cuda:0"))
        with self.assertRaises(RuntimeError):
            tensor.to_numpy()

    def test_fill_on_cuda_raises(self):
        tensor = Tensor((2, 3), Device("cuda:0"))
        with self.assertRaises(RuntimeError):
            tensor.fill(1.0)

    def test_invalid_shape_negative_dimension_raises(self):
        """numpy.zeros rejects negative dimensions (current behavior)."""
        with self.assertRaises(ValueError):
            Tensor((2, -1), Device("cpu"))

    def test_invalid_shape_non_int_dimension_raises(self):
        """numpy.zeros rejects non-integer shape entries (current behavior)."""
        with self.assertRaises((TypeError, ValueError)):
            Tensor((2, 3.5), Device("cpu"))

    def test_unsupported_device_type_raises(self):
        """Non-Device should raise ValueError from initialization match-case."""
        with self.assertRaises(ValueError):
            Tensor((2, 3), object())


class TestTensorAutogradFields(TestCase):
    def test_requires_grad_default_false(self):
        t = Tensor((2, 3), Device("cpu"))
        self.assertFalse(t.requires_grad)

    def test_requires_grad_can_be_set_in_constructor(self):
        t = Tensor((2, 3), Device("cpu"), requires_grad=True)
        self.assertTrue(t.requires_grad)

    def test_requires_grad_setter(self):
        t = Tensor((2, 3), Device("cpu"))
        t.requires_grad = True
        self.assertTrue(t.requires_grad)
        t.requires_grad = False
        self.assertFalse(t.requires_grad)

    def test_grad_default_none(self):
        t = Tensor((2, 3), Device("cpu"))
        self.assertIsNone(t.grad)

    def test_zero_grad_clears_grad(self):
        # See note in original test: we avoid setting private _grad.
        t = Tensor((2, 3), Device("cpu"), requires_grad=True)
        t.zero_grad()
        self.assertIsNone(t.grad)

    def test_ctx_default_none(self):
        t = Tensor((2, 3), Device("cpu"))
        self.assertIsNone(t._get_ctx())

    def test_ctx_can_be_attached_via_constructor(self):
        parent = Tensor((2, 3), Device("cpu"))
        ctx = Context(
            parents=[parent],
            backward_fn=lambda grad_out: (None,),
        )
        t = Tensor((2, 3), Device("cpu"), ctx=ctx)
        self.assertIs(t._get_ctx(), ctx)
        self.assertEqual(list(t._get_ctx().parents), [parent])

    def test_set_ctx_and_get_ctx_roundtrip(self):
        t = Tensor((2, 3), Device("cpu"))
        parent = Tensor((2, 3), Device("cpu"))
        ctx = Context(
            parents=[parent],
            backward_fn=lambda grad_out: (None,),
        )
        t._set_ctx(ctx)
        self.assertIs(t._get_ctx(), ctx)

        t._set_ctx(None)
        self.assertIsNone(t._get_ctx())

    def test_context_save_for_backward_appends(self):
        p1 = Tensor((2, 3), Device("cpu"))
        p2 = Tensor((2, 3), Device("cpu"))
        ctx = Context(parents=[p1], backward_fn=lambda grad_out: (None,))
        self.assertEqual(ctx.saved_tensors, [])

        ctx.save_for_backward(p1, p2)
        self.assertEqual(ctx.saved_tensors, [p1, p2])

    def test_context_saved_meta_default_empty_dict(self):
        p1 = Tensor((2, 3), Device("cpu"))
        ctx = Context(parents=[p1], backward_fn=lambda grad_out: (None,))
        self.assertIsInstance(ctx.saved_meta, dict)
        self.assertEqual(ctx.saved_meta, {})

        ctx.saved_meta["stride"] = 2
        self.assertEqual(ctx.saved_meta["stride"], 2)


class TestTensorGetItem(TestCase):
    def _make_arange(self, shape, requires_grad=False):
        t = Tensor(shape, Device("cpu"), requires_grad=requires_grad)
        t.copy_from_numpy(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        return t

    def test_getitem_forward_basic_slice(self):
        x = self._make_arange((3, 4), requires_grad=False)
        y = x[:, 1:3]
        expected = x.to_numpy()[:, 1:3]
        self.assertEqual(y.shape, expected.shape)
        self.assertTrue(np.array_equal(y.to_numpy(), expected))

    def test_getitem_forward_integer_index_reduces_dim(self):
        x = self._make_arange((3, 4), requires_grad=False)
        y = x[1]  # shape (4,)
        expected = x.to_numpy()[1]
        self.assertEqual(y.shape, expected.shape)
        self.assertTrue(np.array_equal(y.to_numpy(), expected))

    def test_getitem_forward_scalar_index(self):
        x = self._make_arange((3, 4), requires_grad=False)
        y = x[2, 1]  # scalar
        expected = x.to_numpy()[2, 1]
        self.assertEqual(y.shape, ())
        self.assertAlmostEqual(
            float(np.asarray(y.to_numpy())), float(expected), places=6
        )

    def test_getitem_backward_basic_slice_scatter(self):
        x = self._make_arange((3, 4), requires_grad=True)
        y = x[:, 1:3]  # (3,2)
        loss = y.sum()  # scalar
        loss.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[:, 1:3] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_backward_single_element(self):
        x = self._make_arange((2, 2), requires_grad=True)
        y = x[1, 0]  # scalar
        y.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((2, 2), dtype=np.float32)
        expected[1, 0] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_backward_negative_index(self):
        x = self._make_arange((3, 4), requires_grad=True)
        y = x[-1, -2]  # scalar -> x[2,2]
        y.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[2, 2] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_backward_step_slice(self):
        x = self._make_arange((6,), requires_grad=True)
        y = x[::2]  # indices 0,2,4
        loss = y.sum()
        loss.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((6,), dtype=np.float32)
        expected[0] = 1.0
        expected[2] = 1.0
        expected[4] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_backward_fancy_index_accumulates_repeats(self):
        # This ensures you used np.add.at (or equivalent) for fancy indexing.
        x = Tensor((5,), Device("cpu"), requires_grad=True)
        x.copy_from_numpy(np.zeros((5,), dtype=np.float32))

        y = x[[1, 1, 3]]  # repeated index 1
        loss = y.sum()
        loss.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((5,), dtype=np.float32)
        expected[1] = 2.0
        expected[3] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_backward_boolean_mask(self):
        x = Tensor((5,), Device("cpu"), requires_grad=True)
        x.copy_from_numpy(np.arange(5, dtype=np.float32))

        mask = np.array([True, False, True, False, True])
        y = x[mask]  # shape (3,)
        loss = y.sum()
        loss.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((5,), dtype=np.float32)
        expected[mask] = 1.0
        self.assertTrue(np.array_equal(grad, expected))

    def test_getitem_requires_grad_propagates(self):
        x = self._make_arange((3, 4), requires_grad=True)
        y = x[:, :2]
        self.assertTrue(y.requires_grad)

        x2 = self._make_arange((3, 4), requires_grad=False)
        y2 = x2[:, :2]
        self.assertFalse(y2.requires_grad)

    def test_getitem_on_cuda_raises(self):
        t = Tensor((2, 3), Device("cuda:0"))
        with self.assertRaises(Exception):
            _ = t[0]


class TestTensorStack(TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def _tensor_from_numpy(self, arr: np.ndarray, requires_grad: bool) -> Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        t = Tensor(arr.shape, self.device, requires_grad=requires_grad)
        t.copy_from_numpy(arr)
        return t

    def test_stack_forward_axis0(self):
        a = self._tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), requires_grad=False
        )
        b = self._tensor_from_numpy(
            np.zeros((2, 3), dtype=np.float32), requires_grad=False
        )

        s = Tensor.stack([a, b], axis=0)
        self.assertEqual(s.shape, (2, 2, 3))

        expected = np.stack([a.to_numpy(), b.to_numpy()], axis=0)
        self.assertTrue(np.array_equal(s.to_numpy(), expected))

    def test_stack_forward_axis1(self):
        a = self._tensor_from_numpy(
            np.ones((2, 3), dtype=np.float32), requires_grad=False
        )
        b = self._tensor_from_numpy(
            2.0 * np.ones((2, 3), dtype=np.float32), requires_grad=False
        )

        s = Tensor.stack([a, b], axis=1)
        self.assertEqual(s.shape, (2, 2, 3))

        expected = np.stack([a.to_numpy(), b.to_numpy()], axis=1)
        self.assertTrue(np.array_equal(s.to_numpy(), expected))

    def test_stack_backward_splits_grad_to_inputs_axis0(self):
        """
        For loss = stack([a,b], axis=0).sum(), grads to a and b should be all-ones
        with their original shapes.
        """
        a = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=True)
        b = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=True)

        s = Tensor.stack([a, b], axis=0)
        loss = s.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

        self.assertTrue(
            np.array_equal(a.grad.to_numpy(), np.ones((2, 3), dtype=np.float32))
        )
        self.assertTrue(
            np.array_equal(b.grad.to_numpy(), np.ones((2, 3), dtype=np.float32))
        )

    def test_stack_backward_splits_grad_to_inputs_axis1(self):
        """
        Same as axis0 test, but along axis=1.
        """
        a = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=True)
        b = self._tensor_from_numpy(np.random.randn(2, 3), requires_grad=True)

        s = Tensor.stack([a, b], axis=1)
        loss = s.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

        self.assertTrue(
            np.array_equal(a.grad.to_numpy(), np.ones((2, 3), dtype=np.float32))
        )
        self.assertTrue(
            np.array_equal(b.grad.to_numpy(), np.ones((2, 3), dtype=np.float32))
        )

    def test_stack_rejects_empty_list(self):
        with self.assertRaises(Exception):
            _ = Tensor.stack([], axis=0)

    def test_stack_rejects_mismatched_shapes(self):
        a = self._tensor_from_numpy(
            np.zeros((2, 3), dtype=np.float32), requires_grad=False
        )
        b = self._tensor_from_numpy(
            np.zeros((2, 4), dtype=np.float32), requires_grad=False
        )

        with self.assertRaises(Exception):
            _ = Tensor.stack([a, b], axis=0)

    def test_stack_on_cuda_raises_or_not_supported(self):
        """
        Your current Tensor ops are CPU-first. If stack isn't implemented for CUDA,
        it should raise.
        """
        a = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        b = Tensor((2, 3), Device("cuda:0"), requires_grad=False)

        with self.assertRaises(Exception):
            _ = Tensor.stack([a, b], axis=0)


class TestTensorReshape(TestCase):
    def _make_arange(self, shape, requires_grad=False):
        t = Tensor(shape, Device("cpu"), requires_grad=requires_grad)
        t.copy_from_numpy(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        return t

    def test_reshape_forward_preserves_values(self):
        x = self._make_arange((2, 3), requires_grad=False)
        y = x.reshape((3, 2))

        self.assertEqual(y.shape, (3, 2))
        self.assertTrue(np.array_equal(y.to_numpy(), x.to_numpy().reshape(3, 2)))

    def test_reshape_forward_supports_minus_one_inference(self):
        x = self._make_arange((2, 3, 4), requires_grad=False)
        y = x.reshape((-1, 4))

        self.assertEqual(y.shape, (6, 4))
        self.assertTrue(np.array_equal(y.to_numpy(), x.to_numpy().reshape(6, 4)))

    def test_reshape_scalar_roundtrip(self):
        x = Tensor((), Device("cpu"), requires_grad=False)
        x.copy_from_numpy(np.array(3.14, dtype=np.float32))

        y = x.reshape((1,))
        self.assertEqual(y.shape, (1,))
        self.assertTrue(np.allclose(y.to_numpy(), np.array([3.14], dtype=np.float32)))

        z = y.reshape(())
        self.assertEqual(z.shape, ())
        self.assertAlmostEqual(float(np.asarray(z.to_numpy())), 3.14, places=6)

    def test_reshape_requires_grad_propagates(self):
        x = self._make_arange((2, 3), requires_grad=True)
        y = x.reshape((6,))
        self.assertTrue(y.requires_grad)

        x2 = self._make_arange((2, 3), requires_grad=False)
        y2 = x2.reshape((6,))
        self.assertFalse(y2.requires_grad)

    def test_reshape_backward_routes_gradient_to_original_shape(self):
        x = self._make_arange((2, 3), requires_grad=True)
        y = x.reshape((3, 2))
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # d(sum)/dx = 1 everywhere
        expected = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_reshape_backward_chained_reshapes(self):
        x = self._make_arange((2, 3), requires_grad=True)
        y = x.reshape((6,)).reshape((3, 2))  # reshape chain
        loss = y.sum()
        loss.backward()

        expected = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.array_equal(x.grad.to_numpy(), expected))

    def test_reshape_after_getitem_keeps_grad_path(self):
        # x[t] uses __getitem__ (scatter backward), then reshape should reshape grad for that slice.
        x = Tensor((4, 2, 3), Device("cpu"), requires_grad=True)
        x.copy_from_numpy(np.arange(24, dtype=np.float32).reshape(4, 2, 3))

        t_pick = 2
        x_t = x[t_pick]  # shape (2,3)
        y = x_t.reshape((6,))  # shape (6,)
        loss = y.sum()
        loss.backward()

        grad = x.grad.to_numpy()
        expected = np.zeros((4, 2, 3), dtype=np.float32)
        expected[t_pick, :, :] = 1.0

        self.assertTrue(np.array_equal(grad, expected))

    def test_reshape_invalid_shape_raises(self):
        x = self._make_arange((2, 3), requires_grad=False)

        # 2*3=6 cannot reshape to 5
        with self.assertRaises(Exception):
            _ = x.reshape((5,))

    def test_reshape_on_cuda_raises(self):
        x = Tensor((2, 3), Device("cuda:0"), requires_grad=False)
        with self.assertRaises(Exception):
            _ = x.reshape((3, 2))


if __name__ == "__main__":
    unittest.main()
