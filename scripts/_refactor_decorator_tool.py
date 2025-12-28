import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from typing import (
    Callable,
    Union,
    Any,
)
from keydnn.domain.utils._control_path import create_path_builder


decorator = create_path_builder()


class Mixin:
    pass


class Mixin2:
    def add(self, a: int, b: int) -> Union[int, str]:
        if self._state == 0:
            print("1")
            return a + b
        elif self._state == 1:
            print("2")
            return str(a + b)
        else:
            raise Exception()

    def sub(self, a: int, b: int) -> Union[int, str]: ...


x = decorator(Mixin2, Mixin2.sub, 0)


@x
def y(a: int, b: int) -> int:
    print("1")
    return a - b


print(y(100, 2))


def raise_exc(method: Callable, state: Any) -> None:
    raise Exception(f"{method} {state}")


@decorator(Mixin2, Mixin2.sub, 0, raise_exc)
def _(a: int, b: int) -> str:
    print("2")
    return str(a - b)


class Tensor(Mixin, Mixin2):
    def __init__(self, state: int):
        self._state = state


tensor = Tensor(1)

print("Sum:", x := tensor.add(1, 2), type(x))
print("Dif:", x := tensor.sub(1, 2), type(x))

tensor = Tensor(0)
print("Dif:", x := tensor.sub(1, 2), type(x))

tensor = Tensor(2)
print("Dif:", x := tensor.sub(1, 2), type(x))
