#!/usr/bin/env python3
"""
Create a simple KeyDNN model and export it to a JSON checkpoint.

This script is intended to generate a reference JSON file that can be
visualized, diffed, or used for regression testing.
"""

import os
import sys

# Ensure repo_root/src is importable when running this file directly:
# repo_root/
#   src/keydnn/...
#   scripts/bench_pool2d_native_vs_numpy.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pathlib import Path
import numpy as np

from keydnn.infrastructure._tensor import Tensor
from keydnn.infrastructure._models import Sequential
from keydnn.infrastructure._linear import Linear


def main() -> None:
    np.random.seed(0)

    # ----------------------------
    # Build model
    # ----------------------------
    model = Sequential(
        Linear(in_features=3, out_features=2, bias=True),
    )

    lin: Linear = model[0]  # type: ignore[assignment]

    # Deterministic weights & bias
    W = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    b = np.array([0.5, -1.5], dtype=np.float32)

    lin.weight.copy_from_numpy(W)
    assert lin.bias is not None
    lin.bias.copy_from_numpy(b)

    # ----------------------------
    # Optional: sanity forward pass
    # ----------------------------
    x_np = np.array(
        [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]],
        dtype=np.float32,
    )
    x = Tensor(shape=x_np.shape, device=lin.device, requires_grad=False)
    x.copy_from_numpy(x_np)

    y = model.forward(x).to_numpy()
    print("Forward output:")
    print(y)

    # ----------------------------
    # Save JSON checkpoint
    # ----------------------------
    out_path = Path("linear_model.json")
    model.save_json(out_path)

    print(f"\nSaved JSON checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
