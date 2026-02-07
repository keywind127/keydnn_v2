import unittest
from unittest import TestCase

import numpy as np

from src.keydnn.presentation.interops.keras.context import KerasImportContext


class TestKerasImportContext(TestCase):
    def test_defaults(self):
        ctx = KerasImportContext(device="cpu")
        self.assertEqual(ctx.device, "cpu")
        self.assertEqual(ctx.dtype, np.float32)
        self.assertTrue(ctx.strict)
        self.assertFalse(ctx.allow_non_linear_activation)

    def test_with_overrides(self):
        ctx = KerasImportContext(device="cpu")
        ctx2 = ctx.with_overrides(device="cuda:0", strict=False)

        # original remains unchanged (frozen dataclass behavior)
        self.assertEqual(ctx.device, "cpu")
        self.assertTrue(ctx.strict)

        # new context reflects overrides
        self.assertEqual(ctx2.device, "cuda:0")
        self.assertFalse(ctx2.strict)

        # unchanged fields preserved
        self.assertEqual(ctx2.dtype, np.float32)
        self.assertFalse(ctx2.allow_non_linear_activation)

    def test_with_overrides_all_fields(self):
        ctx = KerasImportContext(device="cpu")
        ctx2 = ctx.with_overrides(
            device="cuda:0",
            dtype=np.float16,
            strict=False,
            allow_non_linear_activation=True,
        )
        self.assertEqual(ctx2.device, "cuda:0")
        self.assertEqual(ctx2.dtype, np.float16)
        self.assertFalse(ctx2.strict)
        self.assertTrue(ctx2.allow_non_linear_activation)


if __name__ == "__main__":
    unittest.main()
