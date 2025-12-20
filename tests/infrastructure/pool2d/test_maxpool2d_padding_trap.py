# tests/infrastructure/test_maxpool2d_padding_trap.py

import unittest
import numpy as np

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
)


class TestMaxPool2dPaddingTrap(unittest.TestCase):
    def test_maxpool_padding_does_not_win_and_grad_stays_in_bounds(self):
        """
        Classic MaxPool trap: padding must not become the max.

        If padding is incorrectly filled with 0 instead of -inf, then for
        all-negative inputs near the border, the pooled output may incorrectly
        select padding values (0), producing wrong forward results and
        misrouted gradients.

        This test uses a strictly negative input and positive padding. The
        correct max must always come from real input values, not padding.
        """
        # Strictly negative input
        x = np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]], dtype=np.float32)

        # Pooling with padding; windows at borders include padded region.
        # Correct implementation pads with -inf (so padding can never win).
        k = (2, 2)
        s = (1, 1)
        p = (1, 1)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=k, stride=s, padding=p)

        # Forward: values must remain negative (cannot become 0 from padding)
        self.assertTrue(np.all(y < 0.0), msg=f"Expected all outputs < 0, got:\n{y}")

        # Backward: make sure gradient doesn't "leak" into padding (i.e., get dropped)
        grad_out = np.ones_like(y, dtype=np.float32)
        grad_x = maxpool2d_backward_cpu(
            grad_out, argmax_idx, x_shape=x.shape, kernel_size=k, stride=s, padding=p
        )

        # If padding ever won, that grad would go to padded area and vanish after cropping,
        # reducing sum(grad_x) below sum(grad_out). Correctly, every grad_out element should
        # map to some real input position, so sums must match.
        self.assertAlmostEqual(
            float(grad_x.sum()),
            float(grad_out.sum()),
            places=6,
            msg=f"Gradient sum mismatch: sum(grad_x)={grad_x.sum()} vs sum(grad_out)={grad_out.sum()}",
        )

        # Sanity: gradients should be finite
        self.assertTrue(np.all(np.isfinite(grad_x)))


if __name__ == "__main__":
    unittest.main()
