import unittest
import numpy as np

from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure.ops.conv2d_cpu import conv2d_forward_cpu


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


class TestConv2dStridePaddingShapes(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def _assert_shape_matches_cpu_ref(
        self,
        *,
        x_shape: tuple[int, int, int, int],
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        x_np = np.random.randn(*x_shape).astype(np.float32)
        x = tensor_from_numpy(x_np, self.device, requires_grad=False)

        conv = Conv2d(
            in_channels=x_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            device=self.device,
        )

        y = conv.forward(x)

        w_np = conv.weight.to_numpy()
        b_np = conv.bias.to_numpy() if conv.bias is not None else None
        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=stride, padding=padding)

        self.assertEqual(y.shape, y_ref.shape)

    def test_stride_padding_variants_shapes(self):
        cases = [
            # stride=(2,2), padding=(0,0)
            ((2, 3, 9, 9), 4, (3, 3), (2, 2), (0, 0)),
            # stride=(1,2), padding=(1,0)
            ((1, 2, 8, 7), 3, (3, 2), (1, 2), (1, 0)),
            # stride=(2,1), padding=(0,1)
            ((1, 1, 10, 6), 2, (3, 3), (2, 1), (0, 1)),
        ]

        for x_shape, c_out, k, s, p in cases:
            with self.subTest(
                x_shape=x_shape, out_channels=c_out, kernel=k, stride=s, padding=p
            ):
                self._assert_shape_matches_cpu_ref(
                    x_shape=x_shape,
                    out_channels=c_out,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                )


if __name__ == "__main__":
    unittest.main()
