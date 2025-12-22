import unittest


from src.keydnn.domain.device._device import Device
from src.keydnn.domain._module import IModule
from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure._linear import Linear


class TestIModuleProtocolCompatibility(unittest.TestCase):

    def test_linear_is_compatible_with_imodule_protocol(self):
        m = Linear(3, 4, device=Device("cpu"))
        self.assertIsInstance(m, IModule)

    def test_module_base_is_compatible_with_imodule_protocol(self):
        # Module implements IModule but forward() isn't implemented;
        # Protocol compatibility only checks member presence.
        m = Module()
        self.assertIsInstance(m, IModule)


if __name__ == "__main__":
    unittest.main()
