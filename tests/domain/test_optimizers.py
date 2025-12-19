import unittest

from src.keydnn.domain._optimizers import IOptimizer
from src.keydnn.domain._device import Device
from src.keydnn.infrastructure._optimizers import SGD, Adam
from src.keydnn.infrastructure._parameter import Parameter


class TestOptimizerProtocol(unittest.TestCase):
    def test_sgd_conforms_to_ioptimizer(self):
        p = Parameter(shape=(1,), device=Device("cpu"), requires_grad=True)
        opt = SGD([p], lr=1e-3)
        self.assertIsInstance(opt, IOptimizer)

    def test_adam_conforms_to_ioptimizer(self):
        p = Parameter(shape=(1,), device=Device("cpu"), requires_grad=True)
        opt = Adam([p], lr=1e-3)
        self.assertIsInstance(opt, IOptimizer)


if __name__ == "__main__":
    unittest.main()
