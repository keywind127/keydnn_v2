import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.keydnn.infrastructure.module._serialization_weights import (
    extract_state_payload,
)


class TestModelJsonSaveLoad(unittest.TestCase):
    def _build_known_serializable_model(self):
        """
        Try to build a small known-serializable model.
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
        except Exception as e:
            self.skipTest(
                f"Could not import Sequential/Linear for JSON save/load tests: {e}"
            )

        model = Sequential(
            Linear(4, 3, bias=True),
        )
        return model

    def _set_all_weights_deterministically(self, model) -> None:
        """
        Best-effort: write deterministic values into parameters using supported APIs.
        """

        state = extract_state_payload(model)

        new_state: Dict[str, Dict[str, Any]] = {}
        for k, payload in state.items():

            pass

        named_params = getattr(model, "named_parameters", None)
        if not callable(named_params):
            self.skipTest("Model does not implement named_parameters()")

        rng = np.random.default_rng(0)
        for name, p in named_params():

            sh = getattr(p, "shape", None)
            if sh is None:
                t = getattr(p, "data", None)
                if isinstance(t, np.ndarray):
                    sh = t.shape
                else:
                    sh = getattr(t, "shape", None)
            if sh is None:
                self.skipTest(f"Could not determine shape for param {name}")

            arr = rng.standard_normal(size=tuple(sh), dtype=np.float32)
            cfn = getattr(p, "copy_from_numpy", None)
            if callable(cfn):
                cfn(arr)
            else:
                t = getattr(p, "data", None)
                if isinstance(t, np.ndarray):
                    t[...] = arr
                else:
                    cfn2 = getattr(t, "copy_from_numpy", None)
                    if callable(cfn2):
                        cfn2(arr)
                    else:
                        self.skipTest(f"Could not write weights for param {name}")

    def test_save_json_and_load_json_roundtrip_preserves_weights(self) -> None:
        model = self._build_known_serializable_model()
        self._set_all_weights_deterministically(model)

        state_before = extract_state_payload(model)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.json"
            model.save_json(path)

            loaded = type(model).load_json(path)
            state_after = extract_state_payload(loaded)

        self.assertEqual(state_before.keys(), state_after.keys())

        self.assertEqual(state_before, state_after)

    def test_in_memory_payload_restore_restores_weights_in_place(self) -> None:
        model_a = self._build_known_serializable_model()
        model_b = self._build_known_serializable_model()

        self._set_all_weights_deterministically(model_a)

        named_params = getattr(model_b, "named_parameters", None)
        if not callable(named_params):
            self.skipTest("Model does not implement named_parameters()")

        rng = np.random.default_rng(123)
        for name, p in named_params():
            sh = getattr(p, "shape", None)
            if sh is None:
                t = getattr(p, "data", None)
                if isinstance(t, np.ndarray):
                    sh = t.shape
                else:
                    sh = getattr(t, "shape", None)
            if sh is None:
                self.skipTest(f"Could not determine shape for param {name}")

            arr = rng.standard_normal(size=tuple(sh), dtype=np.float32)
            cfn = getattr(p, "copy_from_numpy", None)
            if callable(cfn):
                cfn(arr)
            else:
                t = getattr(p, "data", None)
                if isinstance(t, np.ndarray):
                    t[...] = arr
                else:
                    cfn2 = getattr(t, "copy_from_numpy", None)
                    if callable(cfn2):
                        cfn2(arr)
                    else:
                        self.skipTest(f"Could not write weights for param {name}")

        payload = model_a.to_json_payload()
        model_b.from_json_payload_(payload)

        state_a = extract_state_payload(model_a)
        state_b = extract_state_payload(model_b)
        self.assertEqual(state_a, state_b)


if __name__ == "__main__":
    unittest.main()
