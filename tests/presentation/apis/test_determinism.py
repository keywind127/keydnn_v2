import os
import unittest

from keydnn.presentation.apis.utils.determinism import (
    set_deterministic,
    get_deterministic,
)


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class TestDeterminismConfig(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot environment to avoid leaking across tests
        self._old_env = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}

    def tearDown(self) -> None:
        # Restore environment
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_enabled_true_sets_default_threads_to_one(self) -> None:
        state = set_deterministic(True)
        self.assertTrue(state.enabled)
        self.assertEqual(state.cpu_threads, 1)

        for k in _THREAD_ENV_VARS:
            self.assertEqual(os.environ.get(k), "1")

    def test_custom_cpu_threads_sets_env_vars(self) -> None:
        state = set_deterministic(True, cpu_threads=8)
        self.assertTrue(state.enabled)
        self.assertEqual(state.cpu_threads, 8)

        for k in _THREAD_ENV_VARS:
            self.assertEqual(os.environ.get(k), "8")

    def test_cpu_threads_none_does_not_modify_env(self) -> None:
        # Put known values in env first
        os.environ["OMP_NUM_THREADS"] = "16"
        os.environ["MKL_NUM_THREADS"] = "16"
        os.environ["OPENBLAS_NUM_THREADS"] = "16"
        os.environ["NUMEXPR_NUM_THREADS"] = "16"

        before = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}

        state = set_deterministic(True, cpu_threads=None)
        self.assertTrue(state.enabled)
        self.assertIsNone(state.cpu_threads)

        after = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
        self.assertEqual(before, after)

    def test_get_deterministic_tracks_last_call(self) -> None:
        set_deterministic(True, cpu_threads=2)
        st = get_deterministic()
        self.assertIsNotNone(st)
        self.assertTrue(st.enabled)
        self.assertEqual(st.cpu_threads, 2)

        set_deterministic(False, cpu_threads=4)
        st2 = get_deterministic()
        self.assertIsNotNone(st2)
        self.assertFalse(st2.enabled)
        self.assertEqual(st2.cpu_threads, 4)

    def test_enabled_requires_bool(self) -> None:
        with self.assertRaises(TypeError):
            set_deterministic(1)  # type: ignore[arg-type]

    def test_cpu_threads_validation(self) -> None:
        with self.assertRaises(ValueError):
            set_deterministic(True, cpu_threads=0)

        with self.assertRaises(ValueError):
            set_deterministic(True, cpu_threads=-3)

        with self.assertRaises(ValueError):
            set_deterministic(True, cpu_threads=3.14)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
