# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python,
with a strong emphasis on **clean architecture**, **explicit interfaces**, and a
**practical CPU / CUDA execution stack**.

It is designed to be both:

- a **learning-friendly** implementation of modern deep learning abstractions, and
- a **performance-oriented sandbox** for experimenting with real backends
  (native CPU kernels, CUDA kernels, and vendor libraries).

---

## Why KeyDNN?

KeyDNN prioritizes clarity and correctness over convenience shortcuts.

Key design goals include:

- **Explicit architecture boundaries**  
  Public APIs are clearly separated from internal infrastructure.

- **From-first-principles implementation**  
  Core components such as tensors, autograd, layers, and optimizers are implemented
  directly, without wrapping existing frameworks.

- **Practical performance paths**  
  CPU reference implementations are paired with optional native CPU acceleration
  and a CUDA backend with cuBLAS / cuDNN integration where available.

- **Tested behavior**  
  CPU ↔ CUDA parity tests and unit tests validate correctness across backends.

---

## Quick taste

### Minimal tensor + autograd

```python
from keydnn import Tensor, Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

---

### CUDA example (device-resident computation)

```python
from keydnn import Tensor, Device

x = Tensor.rand((1024, 1024), device=Device("cuda:0"), requires_grad=True)
y = (x @ x.T).mean()
y.backward()

print(y.item())
```

CUDA execution is enabled automatically when available and explicitly requested
via the device abstraction.

---

## What’s included

KeyDNN currently provides:

- Tensor abstraction with reverse-mode automatic differentiation
- CPU backend (NumPy) with optional native C++ acceleration
- CUDA backend with device-resident tensors
- Core layers:

  - Linear / Dense
  - Conv2D / Conv2DTranspose
  - Pooling (Max / Avg / GlobalAvg)
  - Normalization (BatchNorm1d/2d, LayerNorm)

- Optimizers (SGD, Adam)
- Loss functions (MSE, SSE, BCE, CCE)
- Keras-style training loop (`Model.fit`)
- Callbacks (EarlyStopping, ModelCheckpoint)
- JSON-based model checkpointing

---

## Documentation structure

If you are new to KeyDNN, start here:

1. **Getting Started → Installation**
   Platform support, CUDA setup, and environment configuration.

2. **Getting Started → Quickstart**
   End-to-end examples: tensors, models, training, and evaluation.

3. **Guides**
   Conceptual explanations for tensors, devices, determinism, and training flow.

4. **API Reference**
   Complete, auto-generated documentation for all public APIs.

---

## Public API contract

KeyDNN follows a layered architecture.

Only APIs documented in this site are considered **public and stable**.
Internal implementation details may change as long as public behavior is preserved.

Recommended import style:

```python
from keydnn import Tensor, Sequential, Adam
```

Avoid importing from internal modules unless you are developing KeyDNN itself.

---

## Status

- **Version:** v2.x (stable API)
- **Platform:** Windows 10/11 x64 (CPU + CUDA)
- **CUDA:** Supported when a compatible NVIDIA GPU and runtime are available

Linux and macOS CPU builds may work but are not yet CI-validated.

---

## Philosophy

KeyDNN is not intended to replace large production frameworks.

It exists to:

- make deep learning internals understandable,
- explore backend design trade-offs,
- and serve as a rigorous experimental platform for systems-level ML work.

If you value transparency, correctness, and architectural clarity, you’re in the
right place.
