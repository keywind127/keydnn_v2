"""
Module configuration serialization/deserialization helpers.

This module provides a small registry-based mechanism to convert a "Module"
object graph into a JSON-serializable configuration tree and rebuild it later.

Key ideas
---------
- **Registry-driven reconstruction**: classes participating in deserialization
  must be registered via `@register_module(...)` so `module_from_config()` can
  map `"type"` strings back to concrete Python classes.
- **Best-effort construction**: during load, the system prefers a dedicated
  `from_config()` classmethod, then falls back to `cls(**config)`, and finally
  to `cls()` if the constructor signature does not match.
- **Children are rebuilt recursively**: if a module exposes a `_modules` dict,
  child modules are serialized under `"children"` and re-attached on load,
  preferring a public `register_module(name, child)` API when available.

Configuration node schema
-------------------------
Each node in the tree has the form:

{
  "type": "Linear",
  "config": {...},
  "children": { "0": <node>, "1": <node>, ... }
}

Expected module hooks (optional)
-------------------------------
- `get_config(self) -> dict`: returns JSON-serializable constructor/state config.
- `from_config(cls, cfg: dict) -> Module`: classmethod/constructor alternative.
- `_modules: dict[str, Module]`: child container (PyTorch-style).
- `register_module(self, name: str, child: Module) -> None`: public child attach.
- `_post_load(self) -> None`: invoked after children are attached.

Notes
-----
- This module does not enforce a specific base class; it relies on duck-typing.
- The global `_MODULE_REGISTRY` is process-local and should be populated
  at import time (e.g., in module definitions) to ensure deserialization works.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

_MODULE_REGISTRY: dict[str, Type[Any]] = {}


def register_module(name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]:
    """
    Register a module class for configuration-based deserialization.

    This decorator adds the decorated class to the global `_MODULE_REGISTRY`,
    enabling `module_from_config()` to rebuild instances from serialized nodes.

    Parameters
    ----------
    name:
        Optional explicit registry key. If omitted, the class's `__name__` is used.

    Returns
    -------
    Callable[[Type[Any]], Type[Any]]
        A decorator that registers the class and returns it unchanged.

    Examples
    --------
    >>> @register_module()
    ... class Linear:
    ...     ...
    >>> @register_module("MyFancyBlock")
    ... class Block:
    ...     ...
    """

    def deco(cls: Type[Any]) -> Type[Any]:
        key = name or cls.__name__
        _MODULE_REGISTRY[key] = cls
        return cls

    return deco


def module_to_config(m: Any) -> dict[str, Any]:
    """
    Convert a module instance into a JSON-serializable configuration tree.

    The returned dict is suitable for JSON encoding and can be fed back into
    `module_from_config()` to reconstruct the module graph.

    Serialization rules
    -------------------
    - `"type"` is set to `m.__class__.__name__`.
    - `"config"` is obtained from `m.get_config()` if present and callable;
      otherwise it defaults to `{}`.
    - `"children"` is built by recursively serializing entries from `m._modules`
      if `m._modules` exists and is a `dict`.

    Parameters
    ----------
    m:
        Module-like object to serialize.

    Returns
    -------
    dict[str, Any]
        A configuration node containing `"type"`, `"config"`, and `"children"`.

    Notes
    -----
    This function intentionally does not attempt to serialize arbitrary Python
    state. It only uses `get_config()` and the `_modules` structure.
    """
    type_name = m.__class__.__name__

    get_cfg = getattr(m, "get_config", None)
    cfg = get_cfg() if callable(get_cfg) else {}

    children: dict[str, Any] = {}
    submods = getattr(m, "_modules", None)
    if isinstance(submods, dict):
        for name, child in submods.items():
            children[name] = module_to_config(child)

    return {"type": type_name, "config": cfg, "children": children}


def module_from_config(node: dict[str, Any]) -> Any:
    """
    Rebuild a module instance from a configuration tree node.

    This function looks up `node["type"]` in `_MODULE_REGISTRY` and constructs an
    instance using the following precedence:

    1) If the class defines a callable `from_config(cfg)`, call it.
    2) Else, try `cls(**cfg)` as a best-effort constructor.
    3) If that fails (TypeError), fall back to `cls()`.

    After the module is created, children found under `node["children"]` are
    recursively deserialized and attached to the module. If the module exposes
    a callable `register_module(name, child)`, it is used to attach children;
    otherwise `setattr(m, name, child)` is used as a fallback.

    Finally, if the instance defines a callable `_post_load()`, it is invoked.

    Parameters
    ----------
    node:
        A configuration node with at least the `"type"` key. `"config"` and
        `"children"` are optional.

    Returns
    -------
    Any
        The reconstructed module instance.

    Raises
    ------
    ValueError
        If the type name is not registered, or if children are present but the
        module cannot accept them (missing or non-dict `_modules`).
    KeyError
        If `"type"` is missing from the node.
    """
    type_name = str(node["type"])
    if type_name not in _MODULE_REGISTRY:
        raise ValueError(
            f"Unknown module type '{type_name}'. " f"Register it via @register_module."
        )

    cls = _MODULE_REGISTRY[type_name]
    cfg = node.get("config", {}) or {}

    from_cfg = getattr(cls, "from_config", None)
    if callable(from_cfg):
        m = from_cfg(cfg)
    else:
        # Best-effort constructor
        try:
            m = cls(**cfg)
        except TypeError:
            m = cls()

    children = node.get("children", {}) or {}
    if children:
        if not hasattr(m, "_modules") or not isinstance(getattr(m, "_modules"), dict):
            raise ValueError(
                f"Module '{type_name}' cannot accept children (no _modules dict)."
            )

        # for name, child_node in children.items():
        #     m._modules[str(name)] = module_from_config(child_node)
        for name, child_node in children.items():
            child = module_from_config(child_node)

            # Prefer the public registration API so both attribute + _modules stay consistent
            reg = getattr(m, "register_module", None)
            if callable(reg):
                reg(str(name), child)
            else:
                # Fallback: ensure attribute is set (Module.__setattr__ should populate _modules)
                setattr(m, str(name), child)

    post = getattr(m, "_post_load", None)
    if callable(post):
        post()

    return m
