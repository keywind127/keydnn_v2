import unittest
import numpy as np

from src.keydnn.infrastructure.utils.weight_initializer._base import (
    WeightInitializer,
)


class DummyTensor:
    def __init__(self, shape=(4, 4), dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.data = None

    def copy_from_numpy(self, arr):
        self.data = np.asarray(arr, dtype=self.dtype)


class TestWeightInitializerRegistry(unittest.TestCase):
    def test_available_contains_known_initializers(self):
        names = WeightInitializer.available()
        self.assertIn("xavier", names)
        self.assertIn("kaiming", names)

    def test_get_returns_callable(self):
        fn = WeightInitializer.get("xavier")
        self.assertTrue(callable(fn))

    def test_unknown_initializer_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            WeightInitializer("___does_not_exist___")
        msg = str(ctx.exception)
        self.assertIn("Unsupported initializer name", msg)
        self.assertIn("Available:", msg)

    def test_register_initializer_no_overwrite_by_default(self):
        # Register a throwaway initializer name unique to this test.
        name = "__unit_test_initializer__"

        @WeightInitializer.register_initializer(name, overwrite=True)
        def init_a(tensor):
            return tensor

        with self.assertRaises(ValueError):

            @WeightInitializer.register_initializer(name)  # overwrite=False default
            def init_b(tensor):
                return tensor

    def test_register_initializer_overwrite_true(self):
        name = "__unit_test_initializer_overwrite__"

        @WeightInitializer.register_initializer(name, overwrite=True)
        def init_a(tensor):
            return tensor

        @WeightInitializer.register_initializer(name, overwrite=True)
        def init_b(tensor):
            return tensor

        # Ensure the registry now points to init_b
        self.assertIs(WeightInitializer.get(name), init_b)

    def test_dispatch_calls_initializer(self):
        name = "__unit_test_dispatch__"
        called = {"ok": False}

        @WeightInitializer.register_initializer(name, overwrite=True)
        def init(tensor):
            called["ok"] = True
            tensor.copy_from_numpy(np.zeros(tensor.shape, dtype=tensor.dtype))
            return tensor

        t = DummyTensor(shape=(3, 5))
        WeightInitializer(name)(t)
        self.assertTrue(called["ok"])
        self.assertEqual(t.data.shape, (3, 5))
        self.assertTrue((t.data == 0).all())


if __name__ == "__main__":
    unittest.main()
