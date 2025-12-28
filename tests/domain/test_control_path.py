import unittest


# ====== import the code under test ======
# If your code lives in some module, replace the below with:
# from your_module import create_path_builder
#
# For convenience, I assume create_path_builder is available in scope.
from typing import (
    runtime_checkable,
    Callable,
    Hashable,
    Optional,
    Protocol,
    Union,
    Dict,
    Type,
    Any,
)
from typing_extensions import ParamSpec, TypeVar
from collections import namedtuple
from functools import wraps
from abc import abstractmethod

P = ParamSpec("P")
R = TypeVar("R")


def create_path_builder() -> Callable[
    [
        Type,
        Callable[P, R],
        Hashable,
        Optional[Union[Exception, Callable[[Callable[P, R], Any], None]]],
    ],
    Callable[[Callable[P, R]], Callable[P, R]],
]:
    MethodKey = namedtuple(
        "MethodKey",
        [
            "ClassName",
            "MethodName",
            "StateVal",
        ],
    )

    methods_map: Dict[MethodKey, Callable] = {}

    @runtime_checkable
    class StatefulObject(Protocol):

        @property
        @abstractmethod
        def _state(self) -> Optional[Any]: ...

    STATE_PROPERTY_NAME = next(
        (
            name
            for name, value in StatefulObject.__dict__.items()
            if value is StatefulObject._state
        )
    )

    def templator(
        cls: Type,
        method: Callable[P, R],
        state: Hashable,
        trap_exception: Optional[
            Union[Exception, Callable[[Callable[P, R], Any], None]]
        ] = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        try:
            hash(state)
        except TypeError:
            STATE_NAME = next(
                (name for name, value in locals().items() if value is state)
            )
            raise TypeError(
                f"The argument for {repr(STATE_NAME)} must be hashable. Got {repr(state)}"
            )
        smk: MethodKey = MethodKey(cls.__name__, method.__name__, state)

        def _get_cur_smk(self: StatefulObject) -> MethodKey:
            return MethodKey(cls.__name__, method.__name__, self._state)

        def decorator(sub_method: Callable[P, R]) -> Callable[P, R]:
            methods_map[smk] = sub_method

            @wraps(method)
            def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Any:
                if not isinstance(self, StatefulObject):
                    raise NotImplementedError(
                        "{} is missing attribute {} (@property)".format(
                            type(self), repr(STATE_PROPERTY_NAME)
                        )
                    )
                if sm := methods_map.get(_get_cur_smk(self)):
                    return sm(*args, **kwargs)
                if not trap_exception:
                    raise NotImplementedError(
                        "Missing control path (state={}) for {}".format(
                            repr(self._state), repr(method)
                        )
                    )
                if callable(trap_exception):
                    trap_exception(method, self._state)
                raise trap_exception()

            setattr(cls, method.__name__, wrapper)
            return sub_method

        return decorator

    return templator


# ====== unit tests ======


class TestCreatePathBuilder(unittest.TestCase):
    def setUp(self) -> None:
        # Fresh builder per test to avoid map-sharing across tests.
        self.decorator = create_path_builder()

    def test_state_must_be_hashable(self) -> None:
        class C:
            @property
            def _state(self):
                return "A"

            def foo(self, x: int) -> int:
                return x

        with self.assertRaises(TypeError) as ctx:
            # list is unhashable
            self.decorator(C, C.foo, state=["not-hashable"])(lambda x: x)

        self.assertIn("must be hashable", str(ctx.exception))

    def test_dispatch_selects_registered_control_path(self) -> None:
        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                # base implementation never used once wrapper installed
                return -999

        @self.decorator(C, C.foo, state="A")
        def foo_A(x: int) -> int:
            return x + 10

        @self.decorator(C, C.foo, state="B")
        def foo_B(x: int) -> int:
            return x + 20

        a = C("A")
        b = C("B")
        self.assertEqual(a.foo(1), 11)
        self.assertEqual(b.foo(1), 21)

    def test_dispatch_supports_none_state(self) -> None:
        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                return -999

        @self.decorator(C, C.foo, state=None)
        def foo_none(x: int) -> int:
            return x * 2

        c = C(None)
        self.assertEqual(c.foo(3), 6)

    def test_missing_state_property_raises_not_implemented(self) -> None:
        class C:
            # No _state property on purpose
            def foo(self, x: int) -> int:
                return x

        # Registering a control path will install the wrapper on C.foo
        @self.decorator(C, C.foo, state="A")
        def foo_A(x: int) -> int:
            return x + 1

        obj = C()
        with self.assertRaises(NotImplementedError) as ctx:
            obj.foo(1)

        # Should mention missing attribute and the property name ('_state')
        self.assertIn("missing attribute", str(ctx.exception))
        self.assertIn("'_state'", str(ctx.exception))

    def test_missing_control_path_without_trap_exception_raises_not_implemented(
        self,
    ) -> None:
        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                return x

        @self.decorator(C, C.foo, state="A")
        def foo_A(x: int) -> int:
            return x + 1

        obj = C("B")  # no registered path
        with self.assertRaises(NotImplementedError) as ctx:
            obj.foo(1)

        self.assertIn("Missing control path", str(ctx.exception))
        self.assertIn("state='B'", str(ctx.exception))

    def test_trap_exception_as_exception_class_raises_that_exception(self) -> None:
        class MissingPathError(Exception):
            pass

        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                return x

        @self.decorator(C, C.foo, state="A", trap_exception=MissingPathError)
        def foo_A(x: int) -> int:
            return x + 1

        obj = C("B")
        with self.assertRaises(MissingPathError):
            obj.foo(1)

    def test_trap_exception_callable_is_called_and_then_raised(self) -> None:
        """
        In this implementation, if trap_exception is callable, it is invoked
        as trap_exception(method, self._state) for side-effects, and then
        the code does `raise trap_exception()`. Therefore the callable must also
        be instantiable / callable-with-no-args returning an exception instance.
        """

        class MyRaisedError(Exception):
            pass

        calls = {"count": 0, "method_name": None, "state": None}

        class TrapFactory:
            def __init__(self):
                self._last_method = None
                self._last_state = None

            def __call__(self, *args):
                if len(args) == 2:
                    # side-effect call: (method, state)
                    calls["count"] += 1
                    calls["method_name"] = getattr(args[0], "__name__", None)
                    calls["state"] = args[1]
                    self._last_method = args[0]
                    self._last_state = args[1]
                    return None
                # no-arg call: must return exception instance
                return MyRaisedError("boom")

        trap = TrapFactory()

        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                return x

        @self.decorator(C, C.foo, state="A", trap_exception=trap)
        def foo_A(x: int) -> int:
            return x + 1

        obj = C("B")
        with self.assertRaises(MyRaisedError) as ctx:
            obj.foo(123)

        self.assertEqual(calls["count"], 1)
        self.assertEqual(calls["method_name"], "foo")
        self.assertEqual(calls["state"], "B")
        self.assertIn("boom", str(ctx.exception))

    def test_wrapper_preserves_original_method_metadata(self) -> None:
        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                """Original foo docstring."""
                return x

        # Install wrapper
        @self.decorator(C, C.foo, state="A")
        def foo_A(x: int) -> int:
            return x + 1

        # wraps(method) should preserve __name__ and __doc__
        self.assertEqual(C.foo.__name__, "foo")
        self.assertEqual(C.foo.__doc__, "Original foo docstring.")

    def test_two_builders_do_not_share_control_paths(self) -> None:
        deco1 = create_path_builder()
        deco2 = create_path_builder()

        class C:
            def __init__(self, st):
                self.__st = st

            @property
            def _state(self):
                return self.__st

            def foo(self, x: int) -> int:
                return -999

        @deco1(C, C.foo, state="A")
        def foo_A_1(x: int) -> int:
            return 111

        # Now overwrite wrapper with deco2 installation
        @deco2(C, C.foo, state="B")
        def foo_B_2(x: int) -> int:
            return 222

        a = C("A")
        b = C("B")

        # After deco2 installation, class method wrapper consults deco2's map.
        # So state A path from deco1 should NOT be found.
        with self.assertRaises(NotImplementedError):
            a.foo(0)

        self.assertEqual(b.foo(0), 222)


if __name__ == "__main__":
    unittest.main()
