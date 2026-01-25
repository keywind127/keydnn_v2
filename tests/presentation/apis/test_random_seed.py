import unittest
import random
import numpy as np

from src.keydnn.presentation.apis.utils import random as kd_random


class TestRandomSeeding(unittest.TestCase):
    def tearDown(self) -> None:
        # Ensure one test doesn't influence another
        kd_random.seed(0)

    def test_python_random_same_seed_same_sequence(self) -> None:
        kd_random.seed(123)
        a = [random.random() for _ in range(5)]

        kd_random.seed(123)
        b = [random.random() for _ in range(5)]

        self.assertEqual(a, b)

    def test_numpy_random_same_seed_same_sequence(self) -> None:
        kd_random.seed(123)
        a = np.random.rand(3, 4)

        kd_random.seed(123)
        b = np.random.rand(3, 4)

        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_numpy_sequence(self) -> None:
        # Use randint to make "different" essentially guaranteed.
        kd_random.seed(1)
        a = np.random.randint(0, 2**31 - 1, size=10)

        kd_random.seed(2)
        b = np.random.randint(0, 2**31 - 1, size=10)

        # For deterministic RNGs, this should never match elementwise.
        np.testing.assert_array_equal(a == b, np.zeros_like(a, dtype=bool))

    def test_seed_is_idempotent_for_next_value(self) -> None:
        kd_random.seed(999)
        x1 = random.randint(0, 10**9)
        y1 = np.random.randint(0, 10**9)

        kd_random.seed(999)
        x2 = random.randint(0, 10**9)
        y2 = np.random.randint(0, 10**9)

        self.assertEqual(x1, x2)
        self.assertEqual(y1, y2)

    def test_get_seed_tracks_last_seed(self) -> None:
        kd_random.seed(7)
        self.assertEqual(kd_random.get_seed(), 7)

        kd_random.seed(42)
        self.assertEqual(kd_random.get_seed(), 42)

    def test_seed_requires_int(self) -> None:
        with self.assertRaises(TypeError):
            kd_random.seed(3.14)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            kd_random.seed("123")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
