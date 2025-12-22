import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure.ops.conv2d_cpu import (
    conv2d_forward_cpu,
    conv2d_backward_cpu,
)


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    """
    Return the underlying Tensor for a Parameter-like object.

    Supports:
      - Parameter is Tensor-like (has to_numpy/copy_from_numpy)
      - Parameter wraps Tensor in `.data` or `.tensor`

    Adjust this helper if your Parameter uses a different attribute name.
    """
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


class TestConv2dModule(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_conv2d_module_forward_shape(self):
        conv = Conv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 2),
            bias=True,
            device=self.device,
        )

        x = Tensor(shape=(2, 3, 10, 9), device=self.device, requires_grad=False)
        y = conv.forward(x)

        # Compute expected shape the same way the CPU op does (reference)
        x_np = x.to_numpy()
        w_np = _unwrap_param_tensor(conv.weight).to_numpy()
        b_np = (
            _unwrap_param_tensor(conv.bias).to_numpy()
            if conv.bias is not None
            else None
        )
        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=(2, 1), padding=(1, 2))

        self.assertEqual(y.shape, y_ref.shape)

    def test_conv2d_module_backward_autograd_matches_cpu(self):
        """
        Deterministically set weight/bias, run:
          y = conv(x)
          loss = y.sum()
          loss.backward()
        Compare x.grad and parameter grads with CPU reference.
        """
        np.random.seed(42)

        conv = Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 2),
            stride=(1, 2),
            padding=(1, 0),
            bias=True,
            device=self.device,
        )

        # Deterministic parameters
        w_np = np.random.randn(3, 2, 3, 2).astype(np.float32)
        b_np = np.random.randn(3).astype(np.float32)

        w_t = _unwrap_param_tensor(conv.weight)
        w_t.copy_from_numpy(w_np)

        b_t = _unwrap_param_tensor(conv.bias)
        b_t.copy_from_numpy(b_np)

        # Input
        x_np = np.random.randn(1, 2, 5, 4).astype(np.float32)
        x = _tensor_from_numpy(x_np, self.device, requires_grad=True)

        # Forward + backward
        y = conv.forward(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "x.grad should be populated after backward()")

        # Parameter grads: depends on whether Parameter is Tensor-like or wraps a Tensor
        w_grad_holder = _unwrap_param_tensor(conv.weight)
        b_grad_holder = _unwrap_param_tensor(conv.bias)

        self.assertIsNotNone(w_grad_holder.grad, "weight.grad should be populated")
        self.assertIsNotNone(b_grad_holder.grad, "bias.grad should be populated")

        # CPU reference under L = sum(y)
        y_ref = conv2d_forward_cpu(x_np, w_np, b_np, stride=(1, 2), padding=(1, 0))
        grad_out_np = np.ones_like(y_ref, dtype=np.float32)

        gx_ref, gw_ref, gb_ref = conv2d_backward_cpu(
            x_np, w_np, b_np, grad_out_np, stride=(1, 2), padding=(1, 0)
        )

        self.assertTrue(np.allclose(x.grad.to_numpy(), gx_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(
            np.allclose(w_grad_holder.grad.to_numpy(), gw_ref, atol=1e-4, rtol=1e-4)
        )
        self.assertTrue(
            np.allclose(b_grad_holder.grad.to_numpy(), gb_ref, atol=1e-5, rtol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
