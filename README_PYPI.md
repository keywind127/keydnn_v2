# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python, with a strong emphasis on:

- **clean architecture** and explicit interfaces
- a **practical CPU / CUDA execution stack**
- correctness-first design validated by extensive tests

It is designed to be both:

- a **learning-friendly** implementation of modern DL abstractions (Tensor, autograd, modules), and
- a **performance-oriented sandbox** for building real backends (native CPU kernels, CUDA kernels, vendor libraries).

> ⚠️ **Status:** Pre-stable (v2 alpha). APIs may change.

---

## Platform Support

- **OS:** Windows 10 / 11 (**x64 only**)
- **Python:** ≥ 3.10
- **CUDA:** Optional (NVIDIA GPU required for acceleration)

CUDA acceleration requires:

- an NVIDIA GPU
- a compatible CUDA runtime
- vendor libraries such as **cuBLAS** / **cuDNN** (when enabled)

If CUDA is unavailable, CPU execution remains fully supported.

> **Note on Linux support**
>
> Linux support is available in **KeyDNN v0** (CPU-focused implementation).
>
> **KeyDNN v2** is a major rewrite with a new CUDA backend and currently targets
> **Windows x64** during its early alpha phase. Linux support for v2 is planned
> and tracked separately.

---

## Highlights

- CUDA **device-pointer–backed Tensor backend**
- Explicit H2D / D2H / D2D memory boundaries (no implicit host materialization)
- Vendor-accelerated kernels:
  - **cuBLAS** GEMM for `matmul`
  - **cuDNN** acceleration for `conv2d`
- CUDA implementations for:
  - elementwise ops
  - reductions
  - pooling
  - in-place scalar ops (optimizer hot paths)
- Extensive **CPU ↔ CUDA parity tests**
- Standalone **microbenchmarks** under `scripts/`

---

## Installation

```bash
git clone https://github.com/keywind127/keydnn_v2.git
cd keydnn_v2
pip install -e .
```

---

## Quickstart

```python
from keydnn.infrastructure.tensor._tensor import Tensor
from keydnn.domain.device._device import Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

---

## Versioning Note

**KeyDNN v2 is a major rewrite** and is **not API-compatible** with KeyDNN v0.

---

## License

Licensed under the **Apache License, Version 2.0**.
