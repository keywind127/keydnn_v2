import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
from src.keydnn.infrastructure._activations import Sigmoid


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def forward_loss_np(
    x_np: np.ndarray, w_np: np.ndarray, b_np: np.ndarray | None
) -> float:
    """
    Compute L = sum(sigmoid(conv2d(x,w,b))) using the KeyDNN modules (but with numpy inputs).
    This keeps the numeric comparison apples-to-apples with your actual implementation.
    """
    device = Device("cpu")
    x = tensor_from_numpy(x_np.astype(np.float32), device, requires_grad=False)

    conv = Conv2d(
        in_channels=x_np.shape[1],
        out_channels=w_np.shape[0],
        kernel_size=(w_np.shape[2], w_np.shape[3]),
        stride=1,
        padding=1,
        bias=(b_np is not None),
        device=device,
    )
    conv.weight.copy_from_numpy(w_np.astype(np.float32))
    if b_np is not None:
        assert conv.bias is not None
        conv.bias.copy_from_numpy(b_np.astype(np.float32))

    act = Sigmoid()
    y = act.forward(conv.forward(x))
    return float(np.asarray(y.sum().to_numpy()))


class TestConv2dSigmoidFiniteDiffChain(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_finite_diff_matches_autograd_for_chain(self):
        """
        Compare autograd gradients with finite-difference gradients for:
            L = sum(sigmoid(conv2d(x, w, b)))
        using a tiny input/kernel for speed.
        """
        # Tiny shapes for stable finite-diff
        x_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
        w_np = np.random.randn(1, 1, 3, 3).astype(np.float32)
        b_np = np.random.randn(1).astype(np.float32)

        # Build graph with autograd
        x = tensor_from_numpy(x_np, self.device, requires_grad=True)

        conv = Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )
        conv.weight.copy_from_numpy(w_np)
        assert conv.bias is not None
        conv.bias.copy_from_numpy(b_np)

        act = Sigmoid()
        y = act.forward(conv.forward(x))
        loss = y.sum()
        loss.backward()

        # Autograd grads
        gx = x.grad.to_numpy().copy()
        gw = conv.weight.grad.to_numpy().copy()
        gb = conv.bias.grad.to_numpy().copy()

        # Finite differences (sample a few entries to keep test fast)
        eps = 1e-3
        atol = 2e-2
        rtol = 2e-2

        # --- check a few x entries ---
        x_indices = [(0, 0, 0, 0), (0, 0, 1, 2), (0, 0, 3, 3)]
        for idx in x_indices:
            x_pos = x_np.copy()
            x_neg = x_np.copy()
            x_pos[idx] += eps
            x_neg[idx] -= eps
            f_pos = forward_loss_np(x_pos, w_np, b_np)
            f_neg = forward_loss_np(x_neg, w_np, b_np)
            fd = (f_pos - f_neg) / (2.0 * eps)
            self.assertTrue(
                np.isclose(gx[idx], fd, atol=atol, rtol=rtol),
                msg=f"x grad mismatch at {idx}: autograd={gx[idx]} fd={fd}",
            )

        # --- check a few weight entries ---
        w_indices = [(0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 2, 2)]
        for idx in w_indices:
            w_pos = w_np.copy()
            w_neg = w_np.copy()
            w_pos[idx] += eps
            w_neg[idx] -= eps
            f_pos = forward_loss_np(x_np, w_pos, b_np)
            f_neg = forward_loss_np(x_np, w_neg, b_np)
            fd = (f_pos - f_neg) / (2.0 * eps)
            self.assertTrue(
                np.isclose(gw[idx], fd, atol=atol, rtol=rtol),
                msg=f"w grad mismatch at {idx}: autograd={gw[idx]} fd={fd}",
            )

        # --- check bias entry ---
        b_pos = b_np.copy()
        b_neg = b_np.copy()
        b_pos[0] += eps
        b_neg[0] -= eps
        f_pos = forward_loss_np(x_np, w_np, b_pos)
        f_neg = forward_loss_np(x_np, w_np, b_neg)
        fd_b = (f_pos - f_neg) / (2.0 * eps)
        self.assertTrue(
            np.isclose(gb[0], fd_b, atol=atol, rtol=rtol),
            msg=f"b grad mismatch: autograd={gb[0]} fd={fd_b}",
        )


if __name__ == "__main__":
    unittest.main()
