# Determinism & Reproducibility

Reproducibility has two common sources of variability:

1. **Random number generation** (initialization, shuffling, data augmentation)
2. **Execution nondeterminism** (thread scheduling and floating-point accumulation order)

KeyDNN exposes two public utilities to address these.

---

## 1) Seed Python + NumPy RNGs

Call `seed()` once at the beginning of your program (before model initialization):

```python
from keydnn import seed

seed(42)
```

This seeds:

- Python `random`
- NumPy global RNG (`np.random`)

---

## 2) Configure CPU determinism

For CPU runs, BLAS/OpenMP may use multiple threads and cause small run-to-run differences
due to floating-point accumulation order.

To reduce this nondeterminism:

```python
from keydnn import set_deterministic

set_deterministic(True)  # defaults to cpu_threads=1
```

To explicitly control threads:

```python
set_deterministic(True, cpu_threads=1)
```

If you want KeyDNN to **not** modify thread-related environment variables:

```python
set_deterministic(True, cpu_threads=None)
```

> Note: Thread-related environment variables may need to be set **before importing NumPy**
> (or any BLAS-backed library) to take full effect in the current process.

---

## Recommended order

```python
from keydnn import seed, set_deterministic

seed(42)
set_deterministic(True)

# build / initialize model after reproducibility is configured
```

---

## CUDA determinism (current status)

CUDA determinism (cuDNN/cuBLAS) and device-side RNG seeding are handled separately.
KeyDNN will expose additional controls once the native backend configuration surface is public.

For now:

- Use CPU determinism controls for CPU runs.
- Treat CUDA runs as potentially nondeterministic unless explicitly documented otherwise.
