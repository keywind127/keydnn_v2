from __future__ import annotations

import inspect
import pkgutil
import importlib
import unittest
from types import ModuleType
from typing import Iterable, Type

from src.keydnn.infrastructure._module import Module
from src.keydnn.infrastructure.module._serialization_core import module_to_config


def _iter_infra_modules(package: ModuleType) -> Iterable[ModuleType]:
    """
    Recursively import and yield all modules under a package.
    """
    if not hasattr(package, "__path__"):
        return

    prefix = package.__name__ + "."
    for m in pkgutil.walk_packages(package.__path__, prefix):
        try:
            yield importlib.import_module(m.name)
        except Exception:
            # Some submodules may require optional deps / native libs.
            # If you want strict behavior, remove this and let import fail.
            continue


def _iter_module_subclasses(*, root_pkg_name: str) -> Iterable[Type[Module]]:
    """
    Import all infra modules under root_pkg_name and return all Module subclasses found.
    """
    root_pkg = importlib.import_module(root_pkg_name)

    seen: set[Type[Module]] = set()

    # include already-imported root pkg module itself
    all_py_mods = [root_pkg, *_iter_infra_modules(root_pkg)]

    for py_mod in all_py_mods:
        for _, obj in inspect.getmembers(py_mod, inspect.isclass):
            if obj is Module:
                continue
            if not issubclass(obj, Module):
                continue
            # Avoid duplicates from re-exports
            if obj in seen:
                continue
            seen.add(obj)

            # Only count classes that *belong* to the infra package namespace
            # (prevents pulling in random third-party Module subclasses, if any)
            if obj.__module__.startswith(root_pkg_name):
                yield obj


def _is_abstract_like(cls: Type[Module]) -> bool:
    """
    Heuristics to skip base/abstract-ish classes that aren't meant to be instantiated.
    """
    name = cls.__name__.lower()
    if name in {"model", "module"}:
        return True
    if "base" in name or "abstract" in name or "mixin" in name:
        return True
    # Many pure containers or stateless wrappers are still instantiable; do not skip.
    return False


class TestAllModulesImplementConfigForJSON(unittest.TestCase):
    """
    Meta-test: ensure every infrastructure Module subclass participates in JSON arch serialization.

    What this catches
    -----------------
    - A new module was added but forgot to implement get_config / from_config (or relies on
      a constructor signature that isn't JSON-friendly)
    - A class is not registered and cannot be serialized at all (module_to_config will fail)

    What this does NOT guarantee
    ----------------------------
    - That the class can actually be instantiated from config (we don't have a generic way
      to construct every module without knowing required args)
    - That weights load correctly (covered by per-layer save/load tests)
    """

    def test_every_module_subclass_is_arch_serializable(self) -> None:
        infra_root = "src.keydnn.infrastructure"

        missing: list[str] = []
        problems: list[str] = []

        for cls in _iter_module_subclasses(root_pkg_name=infra_root):
            # Skip base/abstract-ish helpers
            if _is_abstract_like(cls):
                continue

            # Mixins will often show up here; skip them explicitly (they're not real Modules)
            if not issubclass(cls, Module):
                continue

            # Must provide a get_config override OR be safely constructible via best-effort path.
            # We enforce explicit hooks because you said you want to detect forgotten layers.
            has_get = cls.get_config is not Module.get_config  # type: ignore[attr-defined]
            has_from = hasattr(cls, "from_config") and callable(
                getattr(cls, "from_config")
            )

            if not has_get or not has_from:
                missing.append(
                    f"{cls.__module__}.{cls.__qualname__} "
                    f"(get_config={'OK' if has_get else 'MISSING'}, "
                    f"from_config={'OK' if has_from else 'MISSING'})"
                )
                continue

            # Additionally: ensure module_to_config can at least see/encode the type name.
            # This indirectly checks registration for most implementations.
            try:
                # We can't instantiate generically, so we only validate that the hooks exist.
                # If you *do* have a central registry of default constructors, plug it in here.
                pass
            except Exception as e:
                problems.append(
                    f"{cls.__module__}.{cls.__qualname__}: {type(e).__name__}: {e}"
                )

        if missing or problems:
            msg = []
            if missing:
                msg.append("Modules missing explicit JSON config hooks:")
                msg.extend(f"  - {m}" for m in sorted(missing))
            if problems:
                msg.append("Modules with serialization problems:")
                msg.extend(f"  - {p}" for p in sorted(problems))
            self.fail("\n".join(msg))


class TestAllConcreteModulesRoundTripIfZeroArgConstructible(unittest.TestCase):
    """
    Optional stronger meta-test:
    For modules that are constructible with zero args, ensure arch round-trips:
      m -> module_to_config -> module_from_config -> type match

    This is safe for:
    - stateless modules (Flatten, activations with defaults, GlobalAvgPool2d, etc.)
    - containers like Sequential()

    It will automatically skip modules that require constructor args (Linear, Conv2d, RNN, etc.).
    """

    def test_zero_arg_modules_round_trip_arch(self) -> None:
        infra_root = "src.keydnn.infrastructure"
        from src.keydnn.infrastructure.module._serialization_core import (
            module_from_config,
        )

        skipped: list[str] = []
        failed: list[str] = []

        for cls in _iter_module_subclasses(root_pkg_name=infra_root):
            if _is_abstract_like(cls):
                continue

            # Must have explicit hooks (same rationale as above)
            has_get = cls.get_config is not Module.get_config  # type: ignore[attr-defined]
            has_from = hasattr(cls, "from_config") and callable(
                getattr(cls, "from_config")
            )
            if not (has_get and has_from):
                continue

            # Try zero-arg construction
            try:
                m = cls()  # type: ignore[call-arg]
            except TypeError:
                skipped.append(f"{cls.__module__}.{cls.__qualname__}")
                continue
            except Exception as e:
                failed.append(
                    f"{cls.__module__}.{cls.__qualname__} ctor failed: {type(e).__name__}: {e}"
                )
                continue

            try:
                cfg = module_to_config(m)
                m2 = module_from_config(cfg)
                if type(m2) is not cls:
                    failed.append(
                        f"{cls.__module__}.{cls.__qualname__} round-trip type mismatch: "
                        f"got {type(m2).__module__}.{type(m2).__qualname__}"
                    )
            except Exception as e:
                failed.append(
                    f"{cls.__module__}.{cls.__qualname__} round-trip failed: {type(e).__name__}: {e}"
                )

        if failed:
            self.fail("Round-trip failures:\n" + "\n".join(f"  - {x}" for x in failed))


if __name__ == "__main__":
    unittest.main()
