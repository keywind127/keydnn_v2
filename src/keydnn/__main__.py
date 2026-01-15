# src/keydnn/__main__.py
"""
Package execution entrypoint for `python -m keydnn`.

Python executes `keydnn.__main__` when running the package as a module:

    python -m keydnn ...

This module is intentionally minimal and delegates to the Presentation-layer
CLI entrypoint (`keydnn.presentation.cli.main.main`). Keeping this file thin
ensures a clean layering boundary:

- __main__.py: launch glue only (no business logic)
- Presentation: CLI parsing and command routing
- Application: use case orchestration
- Infrastructure: datasets, tensors, kernels, and backends
"""

from __future__ import annotations

from .presentation.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
