"""
State-based method dispatch (a.k.a. "control-path" templating) via decorators.

This module provides a small mechanism for dynamically routing a single method
call to one of several registered implementations based on an object's
runtime `_state` value.

Core idea
---------
- You define a *base* method on a class (its signature becomes the canonical one).
- You then register multiple "control paths" for that method, each keyed by:
    (ClassName, MethodName, StateVal)
- At runtime, the wrapper looks up `self._state` and dispatches to the
  registered implementation that matches the current state.

Intended use-cases
------------------
- Implementing state machines where behavior changes by state without large
  if/elif chains.
- Providing multiple fast-paths / backends selected by a runtime flag.
- Keeping per-state behaviors isolated as separate functions for readability.

Important notes
---------------
- This design mutates the class: the first time you decorate a control path,
  the original method name is replaced with a wrapper that performs dispatch.
- Registered implementations are stored in a closure-local mapping owned by
  `create_path_builder()`. Different builders do not share mappings.
- The wrapper currently calls the selected sub-method without passing `self`
  (i.e., `sm(*args, **kwargs)`), so sub-methods are expected to close over
  the instance or to not require `self`. If you intend methods to behave like
  normal instance methods, adjust the wrapper to call `sm(self, *args, **kwargs)`.
"""

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
    """
    Create and return a "path builder" function used to register stateful
    control paths for methods.

    The returned function (`templator`) is used like this:

        decorator = create_path_builder()

        class MyClass:
            def foo(self, x: int) -> int: ...

        @decorator(MyClass, MyClass.foo, state="A")
        def foo_A(x: int) -> int:
            ...

        @decorator(MyClass, MyClass.foo, state="B")
        def foo_B(x: int) -> int:
            ...

    When `MyClass.foo(...)` is called, it dispatches to `foo_A` or `foo_B`
    depending on `self._state`.

    Returns
    -------
    Callable
        A function with signature:

            (cls, method, state, trap_exception=None) -> decorator

        where `decorator(sub_method)` registers `sub_method` for that control path
        and replaces `cls.method` with a dispatcher wrapper.
    """

    MethodKey = namedtuple(
        "MethodKey",
        [
            "ClassName",
            "MethodName",
            "StateVal",
        ],
    )
    """
    Tuple-like key used to uniquely identify a control path.

    Fields
    ------
    ClassName : str
        The owning class name.
    MethodName : str
        The base method name being templated.
    StateVal : Hashable
        The state value that selects this implementation.
    """

    methods_map: Dict[MethodKey, Callable] = {}
    """Mapping from (class, method, state) keys to registered implementations."""

    @runtime_checkable
    class StatefulObject(Protocol):
        """
        Protocol describing an object that participates in state-based dispatch.

        Implementers must provide a `_state` property. The dispatcher uses this
        property at runtime to select the correct control path implementation.

        Notes
        -----
        - `_state` may be `None`; in that case, dispatch looks for a control path
          registered under state `None`.
        - Because this is runtime-checkable, the wrapper can validate that an
          object provides the required `_state` property before dispatching.
        """

        @property
        @abstractmethod
        def _state(self) -> Optional[Any]:
            """Current state value used for dispatch selection."""
            ...

    STATE_PROPERTY_NAME = next(
        (
            name
            for name, value in StatefulObject.__dict__.items()
            if value is StatefulObject._state
        )
    )
    """
    Name of the state property as discovered from the protocol definition.

    This is used only to produce a clearer error message when an object does
    not satisfy `StatefulObject` at runtime.
    """

    def templator(
        cls: Type,
        method: Callable[P, R],
        state: Hashable,
        trap_exception: Optional[
            Union[Exception, Callable[[Callable[P, R], Any], None]]
        ] = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Build a decorator that registers a control path implementation.

        Parameters
        ----------
        cls : Type
            The class whose method should be wrapped for state-based dispatch.
            The wrapper is installed on this class under `method.__name__`.
        method : Callable[P, R]
            The base method being templated. Its signature and metadata (name,
            docstring, annotations) are used for the installed wrapper via
            `functools.wraps(method)`.
        state : Hashable
            The state value that selects the decorated implementation.
            Must be hashable so it can be used as part of the dispatch key.
        trap_exception : Optional[Union[Exception, Callable[[Callable[P, R], Any], None]]]
            Controls what happens when a dispatch target is missing:

            - If `None`, the wrapper raises `NotImplementedError`.
            - If an exception instance (or exception class-like callable),
              the wrapper raises it (by calling it as `trap_exception()`).
            - If a callable, it is invoked as `trap_exception(method, self._state)`
              before raising `trap_exception()`.

            This allows custom error reporting, logging, or metrics before failing.

        Returns
        -------
        Callable[[Callable[P, R]], Callable[P, R]]
            A decorator that, when applied to `sub_method`, registers `sub_method`
            for `(cls, method, state)` and installs/updates the dispatcher wrapper
            on `cls.method.__name__`.

        Raises
        ------
        TypeError
            If `state` is not hashable.
        """
        try:
            hash(state)
        except TypeError:
            # Identify the local variable name for better error text (best effort).
            STATE_NAME = next(
                (name for name, value in locals().items() if value is state)
            )
            raise TypeError(
                f"The argument for {repr(STATE_NAME)} must be hashable. Got {repr(state)}"
            )

        smk: MethodKey = MethodKey(cls.__name__, method.__name__, state)
        """Static method key for the control path being registered by this call."""

        def _get_cur_smk(self: StatefulObject) -> MethodKey:
            """
            Compute the method key for the *current* runtime state of `self`.

            Parameters
            ----------
            self : StatefulObject
                Object providing `_state`.

            Returns
            -------
            MethodKey
                Key with the same class/method names but using `self._state` as
                the state value.
            """
            return MethodKey(cls.__name__, method.__name__, self._state)

        def decorator(sub_method: Callable[P, R]) -> Callable[P, R]:
            """
            Register `sub_method` as the implementation for the configured state.

            This function:
            1) stores `sub_method` in the internal dispatch map under the
               precomputed key `(cls.__name__, method.__name__, state)`,
            2) installs a wrapper on `cls` under `method.__name__` that:
                - validates `self` has `_state`,
                - finds the matching implementation by `self._state`,
                - calls it (or raises per `trap_exception`).

            Parameters
            ----------
            sub_method : Callable[P, R]
                The implementation to run when `self._state == state`.

            Returns
            -------
            Callable[P, R]
                The original `sub_method` (returned unchanged), enabling normal
                decorator stacking and introspection.
            """
            methods_map[smk] = sub_method

            @wraps(method)
            def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Any:
                """
                Dispatch to a registered implementation based on `self._state`.

                Behavior
                --------
                - If `self` does not satisfy `StatefulObject`, raises
                  `NotImplementedError` indicating the missing state property.
                - If a matching control path exists, calls it and returns its
                  result.
                - If no match exists:
                    * if `trap_exception` is falsy, raises `NotImplementedError`
                    * if `trap_exception` is callable, calls it with
                      `(method, self._state)` then raises `trap_exception()`
                    * otherwise raises `trap_exception()` directly

                Notes
                -----
                The current implementation calls the selected `sub_method` as
                `sub_method(*args, **kwargs)` (without passing `self`). This is
                intentional in this exact code, but differs from typical bound
                instance method semantics.
                """
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
