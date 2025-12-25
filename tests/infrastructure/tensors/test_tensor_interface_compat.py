import unittest
from typing import Any, Dict, Set, Type

from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.domain._tensor import ITensor


def _iter_effective_member_names(cls: Type[Any]) -> Set[str]:
    """
    Return the set of member names that exist on a class/protocol, including:
    - methods (functions)
    - properties
    - staticmethods/classmethods
    - other descriptors

    We intentionally inspect the MRO's __dict__ to avoid missing decorated members.
    """
    names: Set[str] = set()

    for base in cls.__mro__:
        # Skip very-base objects to reduce noise (optional, but helps)
        if base is object:
            continue

        d: Dict[str, Any] = getattr(base, "__dict__", {})
        for name, obj in d.items():
            # Include everything here; filtering happens later
            names.add(name)

    return names


def _is_dunder(name: str) -> bool:
    return len(name) >= 4 and name.startswith("__") and name.endswith("__")


def _include_name(name: str) -> bool:
    """
    Include public names and dunders.
    Exclude single-underscore "private" helpers like `_foo`.
    """
    if _is_dunder(name):
        return True
    if name.startswith("_"):
        return False
    return True


class TestITensorMatchesTensorInterface(unittest.TestCase):
    def test_itensor_declares_every_tensor_member(self) -> None:
        tensor_names_all = _iter_effective_member_names(Tensor)
        itensor_names_all = _iter_effective_member_names(ITensor)

        tensor_names = {n for n in tensor_names_all if _include_name(n)}
        itensor_names = {n for n in itensor_names_all if _include_name(n)}

        # Remove common Protocol bookkeeping that isn't meaningful for interface parity.
        # (You can extend this ignore list if your environment adds more Protocol internals.)
        ignore = {
            "__annotations__",
            "__dict__",
            "__weakref__",
            "__doc__",
            "__module__",
            "__subclasshook__",
        }

        tensor_names -= ignore
        itensor_names -= ignore

        missing = sorted(tensor_names - itensor_names)

        self.assertFalse(
            missing,
            (
                "ITensor is missing public/dunder members that exist on Tensor.\n"
                "Note: single-underscore helpers are intentionally excluded.\n"
                f"Missing ({len(missing)}): {missing}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
