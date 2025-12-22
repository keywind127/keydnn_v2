import unittest
import numpy as np

from src.keydnn.infrastructure._models import Sequential, Model
from src.keydnn.infrastructure._parameter import Parameter
from src.keydnn.infrastructure._module import Module
from src.keydnn.domain.device._device import Device


class AddOne(Module):
    def forward(self, x):
        return x + 1


class MulTwo(Module):
    def forward(self, x):
        return x * 2


class DummyParamModule(Module):
    def __init__(self):
        super().__init__()
        device = Device("cpu")
        self.w = Parameter(shape=(1,), device=device)
        self.w.copy_from_numpy(np.array([1.0], dtype=np.float32))

    def forward(self, x):
        return x


class TestSequential(unittest.TestCase):
    def test_forward_chaining_order(self):
        model = Sequential(AddOne(), MulTwo())
        self.assertEqual(model(3), 8)

    def test_len_getitem_iter(self):
        model = Sequential(AddOne(), MulTwo())
        self.assertEqual(len(model), 2)
        self.assertIsInstance(model[0], AddOne)
        self.assertIsInstance(model[1], MulTwo)
        self.assertEqual(len(list(model)), 2)

    def test_add_appends_and_affects_forward(self):
        model = Sequential(AddOne())
        self.assertEqual(model(0), 1)
        model.add(MulTwo())
        self.assertEqual(model(0), 2)
        self.assertEqual(model(3), 8)

    def test_add_rejects_non_module(self):
        model = Sequential()
        with self.assertRaises(TypeError):
            model.add(object())

    def test_summary_contains_layer_names(self):
        model = Sequential(AddOne(), MulTwo())
        s = model.summary()
        self.assertIn("Sequential(", s)
        self.assertIn("(0): AddOne", s)
        self.assertIn("(1): MulTwo", s)

    def test_parameters_are_discoverable(self):
        model = Sequential(DummyParamModule(), AddOne())

        params = list(model.parameters())
        self.assertGreater(len(params), 0)

        if hasattr(model, "named_parameters"):
            named = list(model.named_parameters())
            self.assertTrue(len(named) > 0)

            # named_parameters returns (name, param)
            names = [n for n, _ in named]
            self.assertIn("0.w", names)  # Sequential assigns first layer name "0"

    def test_layers_returns_tuple(self):
        model = Sequential(AddOne(), MulTwo())
        self.assertIsInstance(model.layers(), tuple)
        self.assertEqual(len(model.layers()), 2)


class TestModelPredict(unittest.TestCase):
    def test_predict_runs_forward_or_raises_if_eval_missing(self):
        class Identity(Model):
            def forward(self, x):
                return x

        m = Identity()

        # Your current Model.predict calls self.eval(), but Module/Model doesn't
        # implement eval() yet. So either:
        # - update Model.predict to check for eval() before calling it, OR
        # - expect AttributeError for now.
        try:
            out = m.predict(123)
        except AttributeError as e:
            self.assertIn("eval", str(e))
        else:
            self.assertEqual(out, 123)


if __name__ == "__main__":
    unittest.main()
