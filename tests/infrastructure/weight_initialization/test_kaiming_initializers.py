import math
import unittest
import numpy as np

from src.keydnn.infrastructure.utils.weight_initializer._base import (
    WeightInitializer,
)


class DummyTensor:
    """Minimal tensor stub for initializer unit tests."""

    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.data = None

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        assert arr.shape == self.shape
        self.data = np.asarray(arr, dtype=self.dtype)


def _fan_in_for_shape(shape: tuple[int, ...]) -> int:
    if len(shape) == 0:
        return 1
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2:
        return shape[1]
    receptive = 1
    for d in shape[2:]:
        receptive *= int(d)
    return int(shape[1]) * receptive


class TestKaimingInitializers(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def _assert_mean_std_close(
        self, x: np.ndarray, *, expected_std: float, atol_mean: float, rtol_std: float
    ):
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        self.assertLessEqual(abs(mean), atol_mean, msg=f"mean={mean} too large")
        self.assertTrue(
            math.isclose(std, expected_std, rel_tol=rtol_std, abs_tol=0.0),
            msg=f"std={std} not close to expected_std={expected_std}",
        )

    def test_kaiming_alias_matches_relu_formula(self):
        init = WeightInitializer("kaiming")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in = _fan_in_for_shape(t.shape)
        expected_std = math.sqrt(2.0 / fan_in)
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_kaiming_relu(self):
        init = WeightInitializer("kaiming_relu")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in = _fan_in_for_shape(t.shape)
        expected_std = math.sqrt(2.0 / fan_in)
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_kaiming_leaky_relu_0_01(self):
        init = WeightInitializer("kaiming_leaky_relu_0.01")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in = _fan_in_for_shape(t.shape)
        a = 0.01
        expected_std = math.sqrt(2.0 / ((1.0 + a * a) * fan_in))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )

    def test_kaiming_leaky_relu_0_2(self):
        init = WeightInitializer("kaiming_leaky_relu_0.2")
        t = DummyTensor(shape=(512, 1024), dtype=np.float32)
        init(t)
        fan_in = _fan_in_for_shape(t.shape)
        a = 0.2
        expected_std = math.sqrt(2.0 / ((1.0 + a * a) * fan_in))
        self._assert_mean_std_close(
            t.data, expected_std=expected_std, atol_mean=5e-3, rtol_std=0.10
        )
