# scripts/train_mnist_mlp.py
from __future__ import annotations

import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from keydnn.application.dto.train_mnist_config import TrainMnistConfig
from keydnn.application.examples.train_mnist_mlp import run_train_mnist_mlp

if __name__ == "__main__":
    raise SystemExit(run_train_mnist_mlp(TrainMnistConfig()))
