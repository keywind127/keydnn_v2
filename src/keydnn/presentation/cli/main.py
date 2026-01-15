# src/keydnn/presentation/cli/main.py
"""
Root CLI entrypoint for KeyDNN.

This module belongs to the Presentation layer (PIAD/DDD) and defines the
top-level argument parser and dispatch mechanism for `python -m keydnn`.

Responsibilities
----------------
- Construct the root argparse parser and register subcommands.
- Parse argv and dispatch execution to the selected subcommand handler.
- Return a process-style exit code (integer).

Non-responsibilities
--------------------
- Implementing use cases (Application).
- Performing dataset downloads, training, or device execution (Infrastructure).

Implementation notes
--------------------
- Subcommands are responsible for registering a callable handler by setting
  `args._handler` via `ArgumentParser.set_defaults`.
- The dispatcher in `main()` invokes this handler and returns its integer
  exit code.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .commands.test import add_test_subparser


def build_parser() -> argparse.ArgumentParser:
    """
    Build the root CLI parser for `python -m keydnn`.

    This function constructs an argparse `ArgumentParser`, registers all
    available subcommands, and returns the configured parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured root parser with subcommands attached.
    """
    parser = argparse.ArgumentParser(
        prog="keydnn", description="KeyDNN command line interface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_test_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint.

    Parses command-line arguments, locates the registered subcommand handler,
    and executes it. The handler is expected to return an integer exit code.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Optional argument vector. When None, argparse reads from `sys.argv`.
        This parameter exists primarily for programmatic invocation and
        integration testing.

    Returns
    -------
    int
        Process exit code:
        - 0 indicates success
        - 2 indicates invalid usage / missing handler
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Dispatch: each subcommand sets args._handler
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("No handler registered for command.")
        return 2

    return int(handler(args))
