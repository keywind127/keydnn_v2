import math
import unittest
import numpy as np

from src.keydnn.infrastructure.utils.weight_initializer._base import (
    WeightInitializer,
)


class DummyTensor:
    """
    Minimal tensor stub for initializer unit tests.

    The initializers under test only require:
      - .shape
      - .dtype
      - .copy_from_numpy(np.ndarray)
    """

    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.data = None

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        assert arr.shape == self.shape
        self.data = np.asarray(arr, dtype=self.dtype)


def _fan_in_out_for_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        fan_out, fan_in = shape
        return fan_in, fan_out
    receptive = 1
    for d in shape[2:]:
        receptive *= int(d)
    fan_in = int(shape[1]) * receptive
    fan_out = int(shape[0]) * receptive
    return fan_in, fan_out


class TestXavierInitializers(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def _assert_mean_std_close(
        self, x: np.ndarray, *, expected_std: float, atol_mean: float, rtol_std: float
    ):
        self.assertIsNotNone(x)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        self.assertLessEqual(abs(mean), atol_mean, msg=f"mean={mean} too large")
        self.assertTrue(
            math.isclose(std, expected_std, rel_tol=rtol_std, abs_tol=0.0),
            msg=f"std={std} not close to expected_std={expected_std}",
        )

    def test_xavier_normal_linear(self):
        init = WeightInitializer("xavier")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_xavier_uniform_linear_bounds_and_variance(self):
        init = WeightInitializer("xavier_uniform")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        bound = math.sqrt(6.0 / (fan_in + fan_out))

        self.assertLessEqual(float(t.data.max()), bound * 1.001)
        self.assertGreaterEqual(float(t.data.min()), -bound * 1.001)

        # For U(-b, b): Var = b^2 / 3
        expected_std = math.sqrt((bound * bound) / 3.0)
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_xavier_relu_gain(self):
        init = WeightInitializer("xavier_relu")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        gain = math.sqrt(2.0)
        expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_xavier_tanh_gain(self):
        init = WeightInitializer("xavier_tanh")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        gain = 5.0 / 3.0
        expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_xavier_leaky_relu_registered_0_01(self):
        init = WeightInitializer("xavier_leaky_relu_0.01")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        a = 0.01
        gain = math.sqrt(2.0 / (1.0 + a * a))
        expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_xavier_leaky_relu_registered_0_2(self):
        init = WeightInitializer("xavier_leaky_relu_0.2")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in, fan_out = _fan_in_out_for_shape(t.shape)
        a = 0.2
        gain = math.sqrt(2.0 / (1.0 + a * a))
        expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )


if __name__ == "__main__":
    unittest.main()
