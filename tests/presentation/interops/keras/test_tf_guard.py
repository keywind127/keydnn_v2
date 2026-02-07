import unittest
from unittest import TestCase
from unittest.mock import patch

from src.keydnn.presentation.interops.keras.tf_guard import require_tensorflow


class TestTFGuard(TestCase):
    def test_require_tensorflow_success(self):
        # If TF is installed in the env, this should succeed.
        # If not installed, skip this test.
        try:
            tf = require_tensorflow()
        except ImportError:
            self.skipTest("TensorFlow not installed; skipping success-path test.")

        self.assertIsNotNone(tf)
        self.assertTrue(hasattr(tf, "__name__"))

    def test_require_tensorflow_raises_with_helpful_message(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise ImportError("No module named tensorflow")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError) as cm:
                _ = require_tensorflow()

        msg = str(cm.exception)
        self.assertIn("Keras interop requires TensorFlow", msg)
        self.assertIn("pip install keydnn[keras]", msg)


if __name__ == "__main__":
    unittest.main()
