import unittest
import numpy as np

from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    maxpool2d_backward_cpu,
)


class TestMaxPool2dTieBreaking(unittest.TestCase):
    def test_maxpool_tie_breaking_routes_grad_to_first_argmax(self):
        """
        Catch a common MaxPool bug: when multiple maxima exist in the pooling window,
        backward must route gradient consistently to the same location chosen in forward.

        Implementation detail:
        - This codebase uses np.argmax, which returns the first maximum in flattened order.
        """
        # 1x1x2x2 window with three equal maxima (5.0)
        # Flatten order: [ (0,0), (0,1), (1,0), (1,1) ]
        # First max is at (0,0).
        x = np.array([[[[5.0, 5.0], [5.0, 1.0]]]], dtype=np.float32)

        y, argmax_idx = maxpool2d_forward_cpu(x, kernel_size=2, stride=2, padding=0)
        self.assertEqual(y.shape, (1, 1, 1, 1))
        self.assertTrue(np.allclose(y, np.array([[[[5.0]]]], dtype=np.float32)))

        grad_out = np.array([[[[7.0]]]], dtype=np.float32)
        grad_x = maxpool2d_backward_cpu(
            grad_out,
            argmax_idx,
            x_shape=x.shape,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        expected = np.array([[[[7.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)

        self.assertEqual(grad_x.shape, x.shape)
        self.assertTrue(
            np.allclose(grad_x, expected, atol=1e-6, rtol=1e-6),
            msg=f"Expected grad routed to first argmax only.\nGot:\n{grad_x}\nExpected:\n{expected}",
        )


if __name__ == "__main__":
    unittest.main()
