# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python, with a strong focus on:

- **clean architecture** and explicit interfaces
- a **practical CPU / CUDA execution stack**
- correctness-first design validated by **CPU ↔ CUDA parity tests**

It is designed to be both:

- a **learning-friendly** implementation of modern DL abstractions (Tensor, autograd, modules), and
- a **performance-oriented sandbox** for building real backends (native CPU kernels, CUDA kernels, vendor libraries).

> 🚧 **Status:** **v2.1.0 (alpha)**.  
> The v2 public API is largely stable and actively evolving toward a v2.1 release.
> Breaking changes are avoided when possible and documented when necessary.  
> 📚 **Documentation:** https://keywind127.github.io/keydnn_v2/  
> 💻 **Source:** https://github.com/keywind127/keydnn_v2

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
- **Linux/macOS:** ❌ not yet supported in v2.x (v0 has CPU-focused Linux support)

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

```bash
pip install keydnn
```

Development install:

```bash
git clone https://github.com/keywind127/keydnn_v2.git
cd keydnn_v2
pip install -e .
```

---

## Quickstart

```python
from keydnn.tensors import Tensor, Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

CUDA example:

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

CUDA requires additional setup on Windows (CUDA runtime discovery and optional cuDNN).
See the documentation for details:

- [https://keywind127.github.io/keydnn_v2/getting-started/cuda/](https://keywind127.github.io/keydnn_v2/getting-started/cuda/)

---

## Versioning note

**KeyDNN v2 is a major rewrite** and is **not API-compatible** with KeyDNN v0.

---

## License

Licensed under the **Apache License, Version 2.0**.
