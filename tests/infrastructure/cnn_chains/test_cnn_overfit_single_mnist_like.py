import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure._conv2d_module import Conv2d
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
    Minimal SGD step for tests.

    Assumes params are CPU tensors and grads are accumulated in `p.grad`.
    Updates in-place and clears grads.
    """
    for p in params:
        if p.grad is None:
            continue
        # In-place update on underlying numpy storage
        p.to_numpy()[...] = p.to_numpy() - lr * p.grad.to_numpy()
        p.zero_grad()


def make_mnist_like_one() -> np.ndarray:
    """
    Create a simple MNIST-like '1' digit image (28x28) with a vertical stroke.
    Returns a float32 array in range [0, 1] with shape (1, 1, 28, 28).
    """
    img = np.zeros((28, 28), dtype=np.float32)

    # vertical stroke around the center
    col = 14
    img[4:24, col - 1 : col + 1] = 1.0

    # small base to look like '1'
    img[23:25, 12:17] = 1.0

    # (N, C, H, W)
    return img[None, None, :, :]


def one_hot(label: int, num_classes: int = 10) -> np.ndarray:
    y = np.zeros((1, num_classes), dtype=np.float32)
    y[0, label] = 1.0
    return y


class TestCNNOverfitSingleMNISTLike(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")
        np.random.seed(0)

    def test_overfit_single_digit_one(self):
        """
        Overfit a CNN on a single MNIST-like sample.

        Model:
            Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> Softmax

        Loss:
            Cross-entropy implemented as: -(target * log(probs)).sum()

        Pass criteria:
        - loss decreases significantly
        - predicted class becomes 1
        """
        # ---- data ----
        x_np = make_mnist_like_one()  # (1, 1, 28, 28)
        t_np = one_hot(label=1, num_classes=10)  # (1, 10)

        x = tensor_from_numpy(x_np, self.device, requires_grad=True)
        target = tensor_from_numpy(t_np, self.device, requires_grad=False)

        # ---- model ----
        conv = Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,  # keep 28x28
            bias=True,
            device=self.device,
        )
        relu = ReLU()
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)  # 28x28 -> 14x14
        flatten = Flatten()

        # After pooling: (1, 4, 14, 14) => flatten dim = 4*14*14 = 784
        fc = Linear(in_features=4 * 14 * 14, out_features=10, device=self.device)
        softmax = Softmax()

        params: list[Tensor] = [conv.weight, fc.weight]
        if conv.bias is not None:
            params.append(conv.bias)
        if fc.bias is not None:
            params.append(fc.bias)

        # ---- train loop ----
        lr = 0.2
        steps = 16

        losses: list[float] = []

        for _ in range(steps):
            # forward
            y = conv.forward(x)
            y = relu.forward(y)
            y = pool.forward(y)
            y = flatten.forward(y)
            y = fc.forward(y)
            probs = softmax.forward(y)  # (1, 10)

            # cross-entropy: -sum(target * log(probs))
            # NOTE: this assumes probs are in (0,1] and Tensor.log exists (you have it).
            loss = -(target * probs.log()).sum()

            # backward
            loss.backward()

            # record
            losses.append(float(np.asarray(loss.to_numpy())))

            # update params + clear grads
            sgd_step(params, lr=lr)

            # clear input grad to avoid accumulation across steps
            x.zero_grad()

        # ---- assertions ----
        self.assertGreater(len(losses), 2)
        self.assertTrue(np.all(np.isfinite(np.array(losses, dtype=np.float32))))

        initial = losses[0]
        final = losses[-1]

        # should drop a lot when overfitting one sample
        self.assertLess(final, initial * 0.25)

        # final prediction should be class 1
        # run final forward to check argmax
        y = conv.forward(x)
        y = relu.forward(y)
        y = pool.forward(y)
        y = flatten.forward(y)
        y = fc.forward(y)
        probs = softmax.forward(y)

        pred = int(np.argmax(probs.to_numpy(), axis=1)[0])
        self.assertEqual(pred, 1)


if __name__ == "__main__":
    unittest.main()
