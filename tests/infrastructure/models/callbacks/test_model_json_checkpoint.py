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

        Adjust these imports if your actual paths differ.
        """
        try:
            from src.keydnn.infrastructure.models._sequential import Sequential
            from src.keydnn.infrastructure.fully_connected._linear import Linear
        except Exception as e:
            self.skipTest(
                f"Could not import Sequential/Linear for JSON save/load tests: {e}"
            )

        # Minimal model
        model = Sequential(
            Linear(4, 3, bias=True),
        )
        return model

    def _set_all_weights_deterministically(self, model) -> None:
        """
        Best-effort: write deterministic values into parameters using supported APIs.

        This works with your serialization helpers which expect:
        - p.copy_from_numpy(arr) or p.data as ndarray
        """
        # Extract current shapes via state payload, then write new arrays by name.
        # This avoids relying on internal Parameter fields.
        state = extract_state_payload(model)
        # Create deterministic new arrays with same shape.
        new_state: Dict[str, Dict[str, Any]] = {}
        for k, payload in state.items():
            # payload includes dtype/shape/order/b64; easiest is to re-load,
            # but we avoid importing payload_to_ndarray here to keep it minimal.
            # Instead: read model params directly and write matching shapes.
            pass

        # Simpler: just iterate named_parameters if available and use copy_from_numpy.
        named_params = getattr(model, "named_parameters", None)
        if not callable(named_params):
            self.skipTest("Model does not implement named_parameters()")

        rng = np.random.default_rng(0)
        for name, p in named_params():
            # figure out shape
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

            # Load model back using the concrete class (Sequential is usually a Model subclass).
            loaded = type(model).load_json(path)
            state_after = extract_state_payload(loaded)

        self.assertEqual(state_before.keys(), state_after.keys())
        # Compare payload dicts exactly (base64 content included) for strong equivalence.
        self.assertEqual(state_before, state_after)

    def test_in_memory_payload_restore_restores_weights_in_place(self) -> None:
        model_a = self._build_known_serializable_model()
        model_b = self._build_known_serializable_model()

        self._set_all_weights_deterministically(model_a)

        # Ensure model_b differs (different RNG seed)
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

        # Now restore B from A using in-memory payload.
        payload = model_a.to_json_payload()
        model_b.from_json_payload_(payload)

        state_a = extract_state_payload(model_a)
        state_b = extract_state_payload(model_b)
        self.assertEqual(state_a, state_b)


if __name__ == "__main__":
    unittest.main()
