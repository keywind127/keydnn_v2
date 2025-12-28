"""
State-based method dispatch ("control-path" templating) via decorators.

This module implements a small, reusable mechanism for routing calls to a
single method name to one of multiple registered implementations, based on
a runtime "state" attribute on the receiver object.

Overview
--------
`create_path_builder()` returns a decorator factory ("templator") that lets you:

1) Choose a *base* method on a class. Its name/signature become the public API.
2) Register one or more implementations for specific state values.
3) At runtime, calls to the method are dispatched by looking up
   `getattr(self, STATE_PNAME)` and selecting the matching implementation.

Dispatch key
------------
Each registered control path is keyed by:

    (ClassName, MethodName, StateVal)

The mapping is stored in a closure owned by the specific builder instance,
so different builders do not share state.

Runtime contract
----------------
- The receiver object must provide an attribute (or property) named
  `STATE_PNAME` (default: "_state").
- The state value must be hashable so it can be used as part of the dispatch key.

Error handling
--------------
If no control path is found for the current state, behavior is controlled by
`trap_exception`:

- If `trap_exception` is falsy/None: raise NotImplementedError.
- If `trap_exception` is callable: call it as `trap_exception(method, state)`
  for side effects (logging/metrics), then raise `trap_exception()`.
- Otherwise: raise `trap_exception()`.

Important notes
---------------
- This mechanism mutates the class: the first registration for a given
  (cls, method) replaces `cls.<method_name>` with a dispatcher wrapper.
- The wrapper in *this* implementation invokes the selected function as
  `sm(self, *args, **kwargs)`, i.e., the registered implementation is expected
  to accept `self` as its first argument (normal instance-method style).
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


def create_path_builder(STATE_PNAME: str = "_state") -> Callable[
    [
        Type,
        Callable[P, R],
        Hashable,
        Optional[Union[Exception, Callable[[Callable[P, R], Any], None]]],
    ],
    Callable[[Callable[P, R]], Callable[P, R]],
]:
    """
    Create a state-based dispatch builder for registering control paths.

    Parameters
    ----------
    STATE_PNAME : str, optional
        Name of the attribute/property on the receiver object that stores the
        current state. Defaults to "_state".

        The wrapper will read the state via:

            getattr(self, STATE_PNAME)

    Returns
    -------
    Callable
        A "templator" function with signature:

            templator(cls, method, state, trap_exception=None) -> decorator

        The returned `decorator(sub_method)` registers `sub_method` for the
        given `(cls, method, state)` and installs/updates the dispatcher wrapper
        on `cls` under `method.__name__`.

    Notes
    -----
    - Each call to `create_path_builder()` creates an isolated registry.
    - The builder is intended for refactoring large methods into explicit,
      state-specific control paths without large if/elif blocks.
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
    Dispatch key type for registered control paths.

    The effective dispatch key is a triple:

        (ClassName, MethodName, StateVal)

    where:
    - ClassName is `cls.__name__`
    - MethodName is `method.__name__`
    - StateVal is the hashable state value selected at runtime
    """

    methods_map: Dict[MethodKey, Callable] = {}
    """
    Registry mapping dispatch keys to implementations.

    This registry is closure-local to a single builder instance, ensuring
    separate builders do not share control paths.
    """

    @runtime_checkable
    class StatefulObject(Protocol):
        """
        Runtime-checkable protocol for objects participating in dispatch.

        Objects are considered "stateful" if they expose an attribute named
        `STATE_PNAME` (default "_state"). The dispatcher reads the current state
        with `getattr(self, STATE_PNAME)`.

        This protocol is used only for runtime validation and clearer error
        messages when the state attribute is missing.
        """

    def _state_getter() -> Optional[Any]:
        """
        Abstract getter used to synthesize the required state property.

        This function exists solely to attach an abstract property to the
        `StatefulObject` protocol under the name `STATE_PNAME`.
        """
        ...

    # Attach a required (abstract) property to the protocol at runtime so that
    # `isinstance(obj, StatefulObject)` can validate the presence of STATE_PNAME.
    setattr(StatefulObject, STATE_PNAME, property(abstractmethod(_state_getter)))

    def templator(
        cls: Type,
        method: Callable[P, R],
        state: Hashable,
        trap_exception: Optional[
            Union[Exception, Callable[[Callable[P, R], Any], None]]
        ] = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Build a decorator that registers a state-specific implementation.

        Parameters
        ----------
        cls : Type
            The class whose method will be wrapped for state-based dispatch.
            The dispatcher wrapper is installed on this class under
            `method.__name__`.
        method : Callable[P, R]
            The base method being templated. Its name and signature represent
            the public API. Metadata is preserved on the wrapper via
            `functools.wraps(method)`.
        state : Hashable
            The state value that selects the decorated implementation.
            Must be hashable to serve as part of the dispatch key.
        trap_exception : Optional[Union[Exception, Callable[[Callable[P, R], Any], None]]]
            Determines behavior when a control path is missing for the current
            runtime state:

            - If falsy/None: raise NotImplementedError.
            - If callable: invoke as `trap_exception(method, state)` for side
              effects, then raise `trap_exception()`.
            - Otherwise: raise `trap_exception()`.

        Returns
        -------
        Callable[[Callable[P, R]], Callable[P, R]]
            A decorator that registers `sub_method` for the given state and
            installs/updates the dispatcher wrapper on the class.

        Raises
        ------
        TypeError
            If `state` is not hashable.
        """
        try:
            hash(state)
        except TypeError:
            # Best-effort: identify the local variable name for nicer messages.
            STATE_NAME = next(
                (name for name, value in locals().items() if value is state)
            )
            raise TypeError(
                f"The argument for {repr(STATE_NAME)} must be hashable. Got {repr(state)}"
            )

        smk: MethodKey = MethodKey(cls.__name__, method.__name__, state)
        """Static key for the control path being registered by this templator."""

        def _get_cur_smk(self: StatefulObject) -> MethodKey:
            """
            Compute the dispatch key for the receiver's current runtime state.

            Parameters
            ----------
            self : StatefulObject
                Receiver object providing the `STATE_PNAME` attribute/property.

            Returns
            -------
            MethodKey
                The dispatch key (ClassName, MethodName, current_state).
            """
            return MethodKey(cls.__name__, method.__name__, getattr(self, STATE_PNAME))

        def decorator(sub_method: Callable[P, R]) -> Callable[P, R]:
            """
            Register a control-path implementation for the configured state.

            This decorator:
            1) Stores `sub_method` in the internal registry keyed by
               (cls.__name__, method.__name__, state).
            2) Installs (or overwrites) `cls.<method_name>` with a dispatcher
               wrapper that selects an implementation based on the receiver's
               current state.

            Parameters
            ----------
            sub_method : Callable[P, R]
                Implementation to run when `getattr(self, STATE_PNAME) == state`.

                The selected implementation is invoked as:

                    sub_method(self, *args, **kwargs)

            Returns
            -------
            Callable[P, R]
                The original `sub_method`, enabling decorator stacking and
                straightforward introspection.
            """
            methods_map[smk] = sub_method

            @wraps(method)
            def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Any:
                """
                Dispatch the call to a registered state-specific implementation.

                Parameters
                ----------
                self : Any
                    Receiver object. Must expose `STATE_PNAME` to participate in
                    dispatch.
                *args, **kwargs
                    Arguments forwarded to the selected implementation.

                Returns
                -------
                Any
                    Return value from the selected control-path implementation.

                Raises
                ------
                NotImplementedError
                    If the receiver does not expose the required state attribute,
                    or if no control path exists and `trap_exception` is falsy.
                Exception
                    If `trap_exception` is provided, raised via `trap_exception()`
                    after optional side-effect callback invocation.
                """
                if not isinstance(self, StatefulObject):
                    raise NotImplementedError(
                        "{} is missing attribute {} (@property)".format(
                            type(self), repr(STATE_PNAME)
                        )
                    )

                cur_key = _get_cur_smk(self)
                if sm := methods_map.get(cur_key):
                    return sm(self, *args, **kwargs)

                if not trap_exception:
                    raise NotImplementedError(
                        "Missing control path (state={}) for {}".format(
                            repr(getattr(self, STATE_PNAME)), repr(method)
                        )
                    )

                if callable(trap_exception):
                    trap_exception(method, getattr(self, STATE_PNAME))
                raise trap_exception()

            setattr(cls, method.__name__, wrapper)
            return sub_method

        return decorator

    return templator
