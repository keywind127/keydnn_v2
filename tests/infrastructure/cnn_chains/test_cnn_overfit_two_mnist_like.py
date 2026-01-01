import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from src.keydnn.infrastructure.flatten._flatten_module import Flatten
from src.keydnn.infrastructure._activations import ReLU, Softmax
from src.keydnn.infrastructure._linear import Linear


def tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr.astype(np.float32, copy=False))
    return t


def sgd_step(params: list[Tensor], lr: float) -> None:
    """
    Minimal SGD update for tests.

    Assumes CPU tensors and gradients stored in `p.grad`.
    """
    for p in params:
        if p.grad is None:
            continue
        p.to_numpy()[...] = p.to_numpy() - lr * p.grad.to_numpy()
        p.zero_grad()


def make_mnist_like_one() -> np.ndarray:
    """
    Simple 28x28 '1' pattern, returned as (1, 1, 28, 28).
    """
    img = np.zeros((28, 28), dtype=np.float32)
    col = 14
    img[4:24, col - 1 : col + 1] = 1.0
    img[23:25, 12:17] = 1.0
    return img[None, None, :, :]


def make_mnist_like_zero() -> np.ndarray:
    """
    Simple 28x28 '0' pattern (a hollow ring), returned as (1, 1, 28, 28).
    """
    img = np.zeros((28, 28), dtype=np.float32)

    # outer rectangle-ish ring
    img[6:22, 8] = 1.0
    img[6:22, 19] = 1.0
    img[6, 8:20] = 1.0
    img[21, 8:20] = 1.0

    # slightly thicken to make it easier
    img[7:21, 9] = 1.0
    img[7:21, 18] = 1.0
    img[7, 9:19] = 1.0
    img[20, 9:19] = 1.0

    return img[None, None, :, :]


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    labels: shape (N,), int
    returns: (N, num_classes)
    """
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    for i, lab in enumerate(labels.tolist()):
        y[i, int(lab)] = 1.0
    return y


class TestCNNOverfitTwoMNISTLike(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_overfit_two_samples_zero_vs_one(self):
        """
        Overfit a CNN on two MNIST-like samples (digit 1 vs digit 0).

        Model:
            Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> Softmax

        Loss:
            Cross-entropy implemented as: -(target * log(probs)).sum()

        Pass criteria:
        - loss decreases significantly
        - final predictions match the two training labels
        """
        # ---- data (N=2) ----
        x1 = make_mnist_like_one()  # label 1
        x0 = make_mnist_like_zero()  # label 0

        x_np = np.concatenate([x1, x0], axis=0).astype(np.float32)  # (2, 1, 28, 28)
        labels = np.array([1, 0], dtype=np.int64)
        t_np = one_hot(labels, num_classes=10)  # (2, 10)

        x = tensor_from_numpy(x_np, self.device, requires_grad=True)
        target = tensor_from_numpy(t_np, self.device, requires_grad=False)

        # ---- model ----
        conv = Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,  # keep 28x28
            bias=True,
            device=self.device,
        )
        relu = ReLU()
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)  # 28x28 -> 14x14
        flatten = Flatten()

        # after pool: (2, 6, 14, 14) => flatten dim = 6*14*14 = 1176
        fc = Linear(in_features=6 * 14 * 14, out_features=10, device=self.device)
        softmax = Softmax()

        params: list[Tensor] = [conv.weight, fc.weight]
        if conv.bias is not None:
            params.append(conv.bias)
        if fc.bias is not None:
            params.append(fc.bias)

        # ---- training ----
        lr = 0.15
        steps = 32
        losses: list[float] = []

        for _ in range(steps):
            # forward
            y = conv.forward(x)
            y = relu.forward(y)
            y = pool.forward(y)
            y = flatten.forward(y)
            logits = fc.forward(y)
            probs = softmax.forward(logits)  # (2, 10)

            # cross-entropy
            loss = -(target * probs.log()).sum()

            # backward
            loss.backward()
            losses.append(float(np.asarray(loss.to_numpy())))

            # update
            sgd_step(params, lr=lr)

            # clear input grad
            x.zero_grad()

        # ---- assertions ----
        self.assertGreater(len(losses), 2)
        self.assertTrue(np.all(np.isfinite(np.array(losses, dtype=np.float32))))

        initial = losses[0]
        final = losses[-1]

        # should drop noticeably when memorizing 2 samples
        self.assertLess(final, initial * 0.35)

        # final predictions should match labels
        y = conv.forward(x)
        y = relu.forward(y)
        y = pool.forward(y)
        y = flatten.forward(y)
        logits = fc.forward(y)
        probs = softmax.forward(logits)

        preds = np.argmax(probs.to_numpy(), axis=1).astype(np.int64)
        self.assertTrue(np.array_equal(preds, labels))


if __name__ == "__main__":
    unittest.main()
