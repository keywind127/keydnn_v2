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

from keydnn.infrastructure.native.python._native_loader import load_keydnn_native

if __name__ == "__main__":
    try:
        lib = load_keydnn_native()
        print("OK loaded:", lib)
    except Exception as e:
        print("FAILED:", repr(e))
        raise
