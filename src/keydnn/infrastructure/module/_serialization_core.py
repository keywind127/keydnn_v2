from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

_MODULE_REGISTRY: dict[str, Type[Any]] = {}


def register_module(name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a Module class for JSON deserialization.
    """

    def deco(cls: Type[Any]) -> Type[Any]:
        key = name or cls.__name__
        _MODULE_REGISTRY[key] = cls
        return cls

    return deco


def module_to_config(m: Any) -> dict[str, Any]:
    """
    Convert a Module into a JSON-serializable configuration tree.

    Node format
    -----------
    {
      "type": "Linear",
      "config": {...},
      "children": { "0": <node>, "1": <node>, ... }
    }
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
    Rebuild a Module from a configuration tree.
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

        for name, child_node in children.items():
            m._modules[str(name)] = module_from_config(child_node)

    post = getattr(m, "_post_load", None)
    if callable(post):
        post()

    return m
