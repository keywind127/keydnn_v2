#!/usr/bin/env python3
"""
Visualize a KeyDNN JSON model checkpoint.

This script inspects a JSON checkpoint created by `Model.save_json()` and
prints a human-readable summary of:

- Model format/version
- Module architecture tree
- Parameter names, shapes, dtypes, and encoded sizes

It is intended as a debugging / inspection utility and does not depend
on internal framework code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ----------------------------
# Architecture visualization
# ----------------------------
def _print_arch(node: dict[str, Any], indent: int = 0) -> None:
    pad = "  " * indent
    type_name = node.get("type", "<unknown>")
    print(f"{pad}- {type_name}")

    cfg = node.get("config", {})
    if cfg:
        for k, v in cfg.items():
            print(f"{pad}    {k}: {v}")

    children = node.get("children", {})
    for name, child in children.items():
        print(f"{pad}  [{name}]")
        _print_arch(child, indent + 2)


# ----------------------------
# State visualization
# ----------------------------
def _print_state(state: dict[str, Any], *, max_b64: int | None) -> None:
    print("\nParameters:")
    for name, payload in state.items():
        shape = payload.get("shape")
        dtype = payload.get("dtype")
        order = payload.get("order")
        b64 = payload.get("b64", "")

        size_chars = len(b64)
        shown = b64[:max_b64] + "..." if max_b64 and size_chars > max_b64 else b64

        print(f"  - {name}")
        print(f"      shape : {shape}")
        print(f"      dtype : {dtype}")
        print(f"      order : {order}")
        print(f"      b64   : ({size_chars} chars)")
        if max_b64:
            print(f"              {shown}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a KeyDNN JSON model checkpoint."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to JSON checkpoint file",
    )
    parser.add_argument(
        "--max-b64",
        type=int,
        default=0,
        help="Maximum base64 characters to display per parameter (0 = hide)",
    )

    args = parser.parse_args()
    path: Path = args.path

    payload = json.loads(path.read_text(encoding="utf-8"))

    print("=" * 80)
    print("KeyDNN JSON Checkpoint")
    print("=" * 80)

    print(f"File   : {path}")
    print(f"Format : {payload.get('format')}")

    print("\nArchitecture:")
    _print_arch(payload["arch"], indent=0)

    max_b64 = args.max_b64 if args.max_b64 > 0 else None
    _print_state(payload["state"], max_b64=max_b64)

    print("\nDone.")


if __name__ == "__main__":
    main()
