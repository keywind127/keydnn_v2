import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution._conv2d_module import Conv2d
from src.keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from src.keydnn.infrastructure.flatten._flatten_module import Flatten
from src.keydnn.infrastructure.activations._modules import ReLU, Softmax
from src.keydnn.infrastructure.fully_connected._linear import Linear


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

        p.to_numpy()[...] = p.to_numpy() - lr * p.grad.to_numpy()
        p.zero_grad()


def make_mnist_like_one() -> np.ndarray:
    """
    Create a simple MNIST-like '1' digit image (28x28) with a vertical stroke.
    Returns a float32 array in range [0, 1] with shape (1, 1, 28, 28).
    """
    img = np.zeros((28, 28), dtype=np.float32)

    col = 14
    img[4:24, col - 1 : col + 1] = 1.0

    img[23:25, 12:17] = 1.0

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

        x_np = make_mnist_like_one()
        t_np = one_hot(label=1, num_classes=10)

        x = tensor_from_numpy(x_np, self.device, requires_grad=True)
        target = tensor_from_numpy(t_np, self.device, requires_grad=False)

        conv = Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=self.device,
        )
        relu = ReLU()
        pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        flatten = Flatten()

        fc = Linear(in_features=4 * 14 * 14, out_features=10, device=self.device)
        softmax = Softmax()

        params: list[Tensor] = [conv.weight, fc.weight]
        if conv.bias is not None:
            params.append(conv.bias)
        if fc.bias is not None:
            params.append(fc.bias)

        lr = 0.2
        steps = 16

        losses: list[float] = []

        for _ in range(steps):

            y = conv.forward(x)
            y = relu.forward(y)
            y = pool.forward(y)
            y = flatten.forward(y)
            y = fc.forward(y)
            probs = softmax.forward(y)

            loss = -(target * probs.log()).sum()

            loss.backward()

            losses.append(float(np.asarray(loss.to_numpy())))

            sgd_step(params, lr=lr)

            x.zero_grad()

        self.assertGreater(len(losses), 2)
        self.assertTrue(np.all(np.isfinite(np.array(losses, dtype=np.float32))))

        initial = losses[0]
        final = losses[-1]

        self.assertLess(final, initial * 0.25)

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
