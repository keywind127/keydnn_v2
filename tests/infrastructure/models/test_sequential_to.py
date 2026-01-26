import unittest
from unittest import TestCase

from src.keydnn.domain.device._device import Device


def _cuda_available() -> bool:
    try:
        from src.keydnn.presentation.apis.backend.ops import cuda_available

        return bool(cuda_available())
    except Exception:
        return False


class TestSequentialToAndToInPlace(TestCase):
    def test_sequential_to_transfers_parameters(self):
        """
        Sequential.to(device) should move all registered parameters recursively.

        This test is written to be robust across environments:
        - Always validates CPU->CPU (no-op) behavior.
        - Validates CPU->CUDA only when CUDA is available.
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Sequential.to test skipped (missing import): {e}")

        model = Sequential(
            Linear(2, 3, device=Device("cpu")),
            Linear(3, 4, device=Device("cpu")),
        )

        for p in model.parameters():
            self.assertEqual(str(p.device), "cpu")

        model.to(Device("cpu"))
        for p in model.parameters():
            self.assertEqual(str(p.device), "cpu")

        if not _cuda_available():
            self.skipTest("CUDA not available; skipping CPU->CUDA transfer assertion.")

        model.to(Device("cuda:0"))
        for p in model.parameters():
            self.assertEqual(str(p.device), "cuda:0")

    def test_sequential_to_in_place_transfers_parameters_and_preserves_identity(self):
        """
        Sequential.to_(device) should migrate parameters recursively, in-place when supported.

        We verify:
        - parameter objects remain the same identity after to_() (best-effort contract)
        - devices reflect the requested target device
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
        except (ModuleNotFoundError, ImportError) as e:
            self.skipTest(f"Sequential.to_ test skipped (missing import): {e}")

        model = Sequential(
            Linear(2, 3, device=Device("cpu")),
            Linear(3, 4, device=Device("cpu")),
        )

        params_before = list(model.parameters())
        ids_before = [id(p) for p in params_before]

        model.to_(Device("cpu"))
        params_after_cpu = list(model.parameters())
        ids_after_cpu = [id(p) for p in params_after_cpu]
        self.assertEqual(ids_after_cpu, ids_before)

        for p in params_after_cpu:
            self.assertEqual(str(p.device), "cpu")

        if not _cuda_available():
            self.skipTest(
                "CUDA not available; skipping CPU->CUDA in-place transfer assertions."
            )

        model.to_(Device("cuda:0"))
        params_after_cuda = list(model.parameters())
        ids_after_cuda = [id(p) for p in params_after_cuda]

        self.assertEqual(
            ids_after_cuda,
            ids_before,
            "Expected Sequential.to_() to preserve parameter object identity.",
        )

        for p in params_after_cuda:
            self.assertEqual(str(p.device), "cuda:0")


if __name__ == "__main__":
    unittest.main()
