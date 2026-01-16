# src/keydnn/presentation/cli/commands/test.py
"""
CLI command definitions for `keydnn test`.

This module belongs to the Presentation layer (PIAD/DDD) and defines the
argument parsing and dispatch logic for the `test` subcommand.

Responsibilities
----------------
- Register the `test` subcommand and its flags with the root argparse parser.
- Translate parsed CLI arguments into stable application-layer DTOs.
- Dispatch to application-layer use cases (e.g., training examples).

Non-responsibilities
--------------------
- Implementing training logic or model internals (Application/Infrastructure).
- Downloading datasets or performing device-specific execution (Infrastructure).

Notes
-----
- Each subcommand registers a handler via `set_defaults(_handler=...)`.
  The root CLI dispatcher calls this handler to obtain an integer exit code.
"""

from __future__ import annotations

import argparse

from ....application.dto.train_mnist_config import TrainMnistConfig
from ....application.examples.train_mnist_mlp import run_train_mnist_mlp

from ....application.dto.train_cifar10_config import TrainCifar10Config
from ....application.examples.train_cifar10_conv import run_train_cifar10_conv


def add_test_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `test` subcommand and its flags.

    This function adds the `test` subcommand under the provided `subparsers`
    collection and wires a dispatch handler using `set_defaults`.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object created by `ArgumentParser.add_subparsers()`.
        The `test` parser will be added to this collection.

    Returns
    -------
    None
        This function mutates `subparsers` in-place by registering the command.
    """
    p = subparsers.add_parser("test", help="Run KeyDNN test utilities and examples")
    p.set_defaults(_handler=_handle_test)

    # ------------------------------------------------------------------
    # Example selectors
    # ------------------------------------------------------------------
    p.add_argument(
        "--train_mnist_example",
        action="store_true",
        help="Run the MNIST MLP training example (smoke test / demo).",
    )
    p.add_argument(
        "--train_cifar_example",
        action="store_true",
        help="Run the CIFAR-10 CNN training example (smoke test / demo).",
    )

    # ------------------------------------------------------------------
    # Common knobs (used by selected example)
    # ------------------------------------------------------------------
    p.add_argument(
        "--root",
        type=str,
        default="~/.cache/keydnn",
        help="Dataset cache root directory.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device string, e.g. "cpu" or "cuda:0".',
    )
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for shuffling.")
    p.add_argument(
        "--no-shuffle", action="store_true", help="Disable shuffling each epoch."
    )
    p.add_argument(
        "--limit-train", type=int, default=0, help="Limit training samples (0=all)."
    )
    p.add_argument(
        "--limit-test", type=int, default=0, help="Limit test samples (0=all)."
    )
    p.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=quiet).")

    # ------------------------------------------------------------------
    # Example-specific knobs
    # ------------------------------------------------------------------
    # MNIST-only
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer width (MNIST MLP only).",
    )

    # CIFAR-only
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Apply CIFAR-10 mean/std normalization (CIFAR-10 only).",
    )


def _handle_test(args: argparse.Namespace) -> int:
    """
    Dispatch handler for `keydnn test`.

    This handler is registered onto the argparse namespace via
    `set_defaults(_handler=_handle_test)` in `add_test_subparser`.
    It translates presentation-layer CLI arguments into an application-layer
    configuration DTO and executes the selected use case.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments for the `test` subcommand.

    Returns
    -------
    int
        Process exit code:
        - 0 indicates success
        - 2 indicates invalid usage / no action selected
    """
    # Prefer explicit else-if to avoid ambiguity if both flags are set.
    if args.train_mnist_example and args.train_cifar_example:
        print(
            "Please select only one example at a time:\n"
            "  python -m keydnn test --train_mnist_example\n"
            "  python -m keydnn test --train_cifar_example"
        )
        return 2

    if args.train_mnist_example:
        cfg = TrainMnistConfig(
            root=args.root,
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden_dim=int(args.hidden_dim),
            seed=int(args.seed),
            shuffle=not bool(args.no_shuffle),
            limit_train=int(args.limit_train),
            limit_test=int(args.limit_test),
            verbose=int(args.verbose),
        )
        return run_train_mnist_mlp(cfg)

    if args.train_cifar_example:
        cfg = TrainCifar10Config(
            root=args.root,
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            shuffle=not bool(args.no_shuffle),
            limit_train=int(args.limit_train),
            limit_test=int(args.limit_test),
            normalize=bool(args.normalize),
            verbose=int(args.verbose),
        )
        return run_train_cifar10_conv(cfg)

    print(
        "No test action selected. Try one of:\n"
        "  python -m keydnn test --train_mnist_example\n"
        "  python -m keydnn test --train_cifar_example"
    )
    return 2
