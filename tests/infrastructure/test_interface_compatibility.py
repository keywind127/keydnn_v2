import inspect
import unittest


def _make_cpu_device():
    """
    Try to construct a CPU Device without assuming one specific API.
    Edit this if your Device API is strict.
    """
    from src.keydnn.domain.device._device import Device

    if hasattr(Device, "cpu") and callable(getattr(Device, "cpu")):
        return Device.cpu()

    try:
        return Device("cpu")
    except TypeError:
        pass

    try:
        return Device()
    except TypeError as e:
        raise RuntimeError(
            "Could not construct a CPU Device. "
            "Please update _make_cpu_device() to match your Device API."
        ) from e


def _construct(cls, **candidates):
    """
    Construct cls by matching provided candidate kwargs to the __init__ signature.

    Enhancement:
    - If cls.__init__ accepts **kwargs, pass through all candidates.
    - Otherwise, filter strictly by named parameters (as before).
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kw:
        # __init__(..., **kwargs) -> forward everything
        return cls(**candidates)

    kwargs = {k: v for k, v in candidates.items() if k in params}
    return cls(**kwargs)


def _make_tensor():
    from src.keydnn.infrastructure._tensor import Tensor

    device = _make_cpu_device()

    # Try shape/device style first
    try:
        return _construct(Tensor, shape=(2, 3), device=device)
    except TypeError:
        pass

    # Try data/device style
    try:
        return _construct(
            Tensor,
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=device,
        )
    except TypeError as e:
        raise RuntimeError(
            "Could not construct Tensor for tests. "
            "Update _make_tensor() to match your Tensor __init__."
        ) from e


def _make_parameter():
    from src.keydnn.infrastructure._parameter import Parameter

    device = _make_cpu_device()

    for kwargs in (
        # Try shape/device first (most likely to work if Tensor is shape-based)
        {"shape": (2, 2), "device": device, "requires_grad": True},
        {"shape": (2, 2), "requires_grad": True},
        # Then try data-based construction
        {"data": [[1.0, 2.0], [3.0, 4.0]], "device": device, "requires_grad": True},
        {"data": [[1.0, 2.0], [3.0, 4.0]], "requires_grad": True},
        {"data": [[1.0, 2.0], [3.0, 4.0]]},
    ):
        try:
            return _construct(Parameter, **kwargs)
        except TypeError:
            continue

    raise RuntimeError(
        "Could not construct Parameter for tests. "
        "Update _make_parameter() to match your Tensor/Parameter __init__."
    )


class TestITensorProtocolCompatibility(unittest.TestCase):
    def test_tensor_is_compatible_with_itensor_protocol(self):
        from src.keydnn.domain._tensor import ITensor

        t = _make_tensor()

        # Runtime structural check (works only if ITensor is @runtime_checkable)
        self.assertIsInstance(t, ITensor)

        # Public interface checks (no private attrs)
        shape = t.shape
        self.assertIsInstance(shape, tuple)
        self.assertTrue(all(isinstance(d, int) for d in shape))

        device = t.device
        self.assertIsNotNone(device)


class TestIParameterProtocolCompatibility(unittest.TestCase):
    def test_parameter_is_compatible_with_iparameter_protocol(self):
        from src.keydnn.domain._parameter import IParameter

        p = _make_parameter()

        self.assertIsInstance(p, IParameter)

        # requires_grad should be public and mutable
        self.assertIsInstance(p.requires_grad, bool)
        p.requires_grad = False
        self.assertEqual(p.requires_grad, False)
        p.requires_grad = True
        self.assertEqual(p.requires_grad, True)

        # grad should exist and be None or tensor-like
        g = p.grad
        self.assertTrue(g is None or hasattr(g, "shape"))

        # zero_grad should clear grad
        p.zero_grad()
        self.assertIsNone(p.grad)

    def test_parameter_is_also_tensor_like_if_iparameter_extends_itensor(self):
        """
        If you defined `class IParameter(ITensor, Protocol)`,
        then Parameter should satisfy ITensor as well.
        """
        from src.keydnn.domain._tensor import ITensor
        from src.keydnn.domain._parameter import IParameter

        p = _make_parameter()

        self.assertIsInstance(p, IParameter)
        self.assertIsInstance(p, ITensor)


if __name__ == "__main__":
    unittest.main()
