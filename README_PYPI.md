# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python, with a strong focus on:

- **clean architecture** and explicit interfaces
- a **practical CPU / CUDA execution stack**
- correctness-first design validated by **CPU ↔ CUDA parity tests**

It is designed to be both:

- a **learning-friendly** implementation of modern DL abstractions (Tensor, autograd, modules), and
- a **performance-oriented sandbox** for building real backends (native CPU kernels, CUDA kernels, vendor libraries).

> ✅ **Status:** **v2.0.0 stable**. The public API is intended to be stable; changes will follow semantic versioning.

> **Documentation:** Module-level API reference is planned; current docs focus on examples, architecture, and tested usage patterns.

---

## Platform support

- **OS:** Windows 10 / 11 (**x64 only**)
- **Python:** ≥ 3.10
- **CUDA:** Optional (NVIDIA GPU required for acceleration)

CUDA acceleration requires a compatible CUDA runtime. Some backends use vendor libraries such as
**cuBLAS** / **cuDNN** when available.

If CUDA is unavailable, CPU execution remains supported.

### Support snapshot

- **Windows (CPU):** ✅ supported
- **Windows (CUDA):** ✅ supported (requires NVIDIA GPU + CUDA runtime; cuBLAS/cuDNN optional)
- **Linux/macOS:** ❌ not supported in v2.0.0 (v0 has CPU-focused Linux support)

---

## Highlights

- **CUDA device-pointer–backed Tensor backend**
- Explicit H2D / D2H / D2D memory boundaries (**no implicit host materialization**)
- Vendor-accelerated kernels:
  - **cuBLAS** GEMM for `matmul`
  - **cuDNN** acceleration for `conv2d` / `conv2d_transpose` (when enabled)
- CUDA implementations for core ops:
  - elementwise ops
  - reductions
  - pooling
  - in-place scalar ops (optimizer hot paths)
- Extensive **CPU ↔ CUDA parity tests**
- Standalone **microbenchmarks** under `scripts/`

---

## Installation

### From PyPI

```bash
pip install keydnn
```

### From source (development)

```bash
git clone https://github.com/keywind127/keydnn_v2.git
cd keydnn_v2
pip install -e .
```

---

## Quickstart

### Minimal Tensor + autograd (CPU)

```python
from keydnn.tensors import Tensor, Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

### CUDA example (device-resident ops)

```python
from keydnn.tensors import Tensor, Device
from keydnn.backend import cuda_available

device = Device("cuda:0") if cuda_available() else Device("cpu")

x = Tensor.rand((1024, 1024), device=device, requires_grad=True)
y = (x @ x.T).mean()
y.backward()

print("device:", device)
print("y:", y.item())
```

---

## CUDA setup (Windows)

KeyDNN’s Windows CUDA backend loads a native DLL and relies on the CUDA runtime
(and optionally cuDNN) being discoverable by the current process.

### Environment variables

- `CUDA_PATH` (recommended): points to your CUDA install root, e.g.
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2`
- `CUDNN_PATH` (optional): points to your cuDNN root that contains `bin/`, `lib/`, `include/`,
  e.g. `C:\cudnn`

If you copied cuDNN DLLs into the CUDA install (common manual setup), you typically do **not**
need `CUDNN_PATH` as long as `cudnn*.dll` exists in `<CUDA_PATH>\bin`.

### PowerShell examples

```powershell
# For the current terminal session only:
$env:CUDA_PATH  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
$env:CUDNN_PATH = "C:\cudnn"   # optional
```

> Note: If you change environment variables, restart the Python process (and sometimes the terminal)
> before retrying.

---

## CLI smoke test (MNIST / CIFAR-10)

```bash
# CPU
python -m keydnn test --train_mnist_example --device cpu --epochs 4 --limit-test 1000

# CUDA (if available)
python -m keydnn test --train_mnist_example --device cuda:0 --epochs 4 --limit-test 1000
```

---

## Versioning note

**KeyDNN v2 is a major rewrite** and is **not API-compatible** with KeyDNN v0.

---

## License

Licensed under the **Apache License, Version 2.0**.
