# Installation

KeyDNN is a lightweight deep learning framework implemented from scratch in Python with a practical CPU/CUDA execution stack.

> **Platform support (v2.1.0):** Windows 10/11 x64 is the supported platform (CPU + CUDA).  
> Linux/macOS may build for CPU-only use, but are not officially supported or CI-validated yet.

---

## Install from PyPI

```bash
pip install keydnn
```

---

## Install from source (development)

```bash
git clone https://github.com/keywind127/keydnn_v2.git
cd keydnn_v2
pip install -e .
```

---

## Verify installation

```bash
python - <<EOF
import keydnn
print("KeyDNN imported:", keydnn.__name__)
print("Tensor:", keydnn.Tensor)
print("CUDA available:", keydnn.cuda_available())
EOF
```

If this succeeds, your environment is ready for the Quickstart.

---

## Notes

- KeyDNN’s **public APIs** are re-exported at the top level of `keydnn`.
- Internal packages follow a layered PIAD architecture and are not part of the public contract.
- CUDA usage requires additional setup; see **Getting Started → CUDA**.
