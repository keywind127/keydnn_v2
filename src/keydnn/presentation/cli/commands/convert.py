# src/keydnn/presentation/cli/commands/convert.py
"""
CLI command definitions for `keydnn convert`.

This command converts a Keras Sequential model into a KeyDNN model artifact.

Responsibilities
----------------
- Register the `convert` subcommand and flags.
- Load a Keras model from disk.
- Invoke the Keras -> KeyDNN importer.
- Serialize the resulting KeyDNN model to disk.

Non-responsibilities
--------------------
- Training or execution.
- Numerical validation.
- Backend/device execution.

Serialization format
--------------------
By default, this command writes a *checkpoint-style* JSON artifact containing:
- model architecture ("arch")
- model state / weights ("state")

This matches `Sequential.save_json(...)` / `Sequential.load_json(...)`.

If `--config-only` is passed, the command writes an *architecture-only* JSON
("keydnn.model.v1") without weights. This is not suitable for numerical parity
checks against a trained Keras model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import numpy as np

from ....presentation.interops.keras.importer import from_keras
from ....presentation.interops.keras.converters._base import KerasInteropError


def add_convert_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `convert` subcommand.
    """
    p = subparsers.add_parser(
        "convert",
        help="Convert a Keras Sequential model to a KeyDNN model artifact",
    )
    p.set_defaults(_handler=_handle_convert)

    p.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source Keras model path (.h5 or .keras).",
    )
    p.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination KeyDNN model file (.json).",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Target KeyDNN device (default: "cpu").',
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Target dtype (default: float32).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable strict conversion (default).",
    )
    p.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict conversion.",
    )
    p.add_argument(
        "--allow-non-linear-activation",
        action="store_true",
        default=False,
        help="Allow non-linear fused activations during conversion.",
    )

    p.add_argument(
        "--config-only",
        action="store_true",
        default=False,
        help=(
            "Write architecture-only JSON without weights (not suitable for "
            "numerical parity checks)."
        ),
    )


def _handle_convert(args: argparse.Namespace) -> int:
    """
    Dispatch handler for `keydnn convert`.

    Returns
    -------
    int
        Exit code (0=success, 1=conversion/serialization error, 2=usage error).
    """
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[keydnn convert] Source model not found: {src}")
        return 2

    if dst.suffix.lower() != ".json":
        print("[keydnn convert] Destination must be a .json file")
        return 2

    dtype = np.float32 if args.dtype == "float32" else None
    if dtype is None:
        print(f"[keydnn convert] Unsupported dtype: {args.dtype}")
        return 2

    try:
        kd_model = from_keras(
            str(src),
            device=args.device,
            dtype=dtype,
            strict=bool(args.strict),
            allow_non_linear_activation=bool(args.allow_non_linear_activation),
        )
    except (ImportError, KerasInteropError) as e:
        print(f"[keydnn convert] Conversion failed: {e}")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------
    try:
        if bool(getattr(args, "config_only", False)):
            # Phase 1: architecture-only JSON (no weights)
            cfg = _serialize_keydnn_model_config_only(kd_model)
            with dst.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        else:
            # Default: checkpoint-style JSON (arch + state / weights)
            seq = _ensure_sequential_for_ckpt(kd_model)
            seq.save_json(dst)
    except Exception as e:
        print(f"[keydnn convert] Serialization failed: {e}")
        return 1

    print(f"[keydnn convert] Converted model written to {dst}")
    return 0


def _ensure_sequential_for_ckpt(model: Any) -> Any:
    """
    Ensure the converted model is a `Sequential`-like object that supports
    `save_json(...)`.

    Notes
    -----
    - `from_keras(...)` normally returns a KeyDNN Sequential container.
    - In fallback mode, it may return a list of modules. In that case, wrap
      them into `Sequential` so we can save `arch` + `state`.
    """
    # Already supports checkpoint-style saving
    if hasattr(model, "save_json") and callable(getattr(model, "save_json")):
        return model

    # Wrap list output into Sequential
    if isinstance(model, list):
        try:
            from ....infrastructure.models._sequential import Sequential
        except Exception as e:
            raise ImportError(
                "Cannot import KeyDNN Sequential for ckpt export. "
                "Expected: src.keydnn.infrastructure.models._sequential.Sequential"
            ) from e
        return Sequential(*model)

    # Sequential-like container but without save_json
    if hasattr(model, "modules"):
        mods = list(getattr(model, "modules"))
        try:
            from ....infrastructure.models._sequential import Sequential
        except Exception as e:
            raise ImportError(
                "Cannot import KeyDNN Sequential for ckpt export. "
                "Expected: src.keydnn.infrastructure.models._sequential.Sequential"
            ) from e
        return Sequential(*mods)

    raise TypeError(
        f"Converted model type {type(model).__name__} cannot be exported as a ckpt."
    )


def _serialize_keydnn_model_config_only(model: Any) -> dict:
    """
    Serialize a KeyDNN model or module list into JSON-compatible config-only form.

    This format does NOT embed weights and is intended for:
    - debugging the converted architecture
    - lightweight inspection / diffs

    It is NOT intended for numerical parity checks.

    Returns
    -------
    dict
        JSON-serializable dict with:
        - format: "keydnn.model.v1"
        - modules: list of {"type": ..., "config": ...}
    """
    # Sequential-like container
    if hasattr(model, "modules"):
        mods = list(getattr(model, "modules"))
    elif isinstance(model, list):
        mods = list(model)
    else:
        mods = [model]

    out = {
        "format": "keydnn.model.v1",
        "modules": [],
    }

    for m in mods:
        if not hasattr(m, "get_config"):
            raise TypeError(f"Module {type(m).__name__} does not support get_config()")
        out["modules"].append(
            {
                "type": type(m).__name__,
                "config": m.get_config(),
            }
        )

    return out
