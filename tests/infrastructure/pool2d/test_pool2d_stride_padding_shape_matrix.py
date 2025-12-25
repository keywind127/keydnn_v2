import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor

from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d, AvgPool2d
from src.keydnn.infrastructure.ops.pool2d_cpu import (
    maxpool2d_forward_cpu,
    avgpool2d_forward_cpu,
)


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestPool2dStridePaddingShapeMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_pool2d_shape_matrix_matches_cpu_reference(self):
        """
        Catch common pooling bugs around output shape calculation and padding/stride handling
        by validating module forward shapes against the CPU reference ops over a matrix of
        configurations.
        """
        cases = [
            # (x_shape, kernel, stride, padding)
            ((2, 3, 8, 8), (2, 2), (2, 2), (0, 0)),
            ((1, 1, 7, 6), (3, 2), (1, 2), (1, 0)),
            ((1, 4, 9, 5), (2, 3), (2, 1), (0, 1)),
            ((1, 2, 5, 5), (3, 3), (1, 1), (1, 1)),
        ]

        for x_shape, k, s, p in cases:
            with self.subTest(x_shape=x_shape, kernel=k, stride=s, padding=p):
                x_np = np.random.randn(*x_shape).astype(np.float32)
                x = tensor_from_numpy(x_np, self.device, requires_grad=False)

                # ---- MaxPool2d ----
                maxpool = MaxPool2d(kernel_size=k, stride=s, padding=p)
                y_max = maxpool.forward(x)
                y_max_ref, _ = maxpool2d_forward_cpu(
                    x_np, kernel_size=k, stride=s, padding=p
                )
                self.assertEqual(
                    y_max.shape,
                    y_max_ref.shape,
                    msg=f"MaxPool2d shape mismatch: module={y_max.shape} ref={y_max_ref.shape}",
                )

                # ---- AvgPool2d ----
                avgpool = AvgPool2d(kernel_size=k, stride=s, padding=p)
                y_avg = avgpool.forward(x)
                y_avg_ref = avgpool2d_forward_cpu(
                    x_np, kernel_size=k, stride=s, padding=p
                )
                self.assertEqual(
                    y_avg.shape,
                    y_avg_ref.shape,
                    msg=f"AvgPool2d shape mismatch: module={y_avg.shape} ref={y_avg_ref.shape}",
                )


if __name__ == "__main__":
    unittest.main()
