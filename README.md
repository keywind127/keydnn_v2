# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python with a strong focus on **clean architecture**, **explicit interfaces**, and **a practical CPU/CUDA execution stack**.

It is designed to be both:

- a **learning-friendly** implementation of modern DL abstractions (Tensor, autograd, modules), and
- a **performance-oriented sandbox** for building real backends (native CPU kernels, CUDA kernels, and vendor libraries).

## Highlights

- **CUDA Tensor backend** with device-pointer–backed storage and explicit memcpy boundaries (H2D / D2H / D2D)
- **Vendor-accelerated kernels**
  - **cuBLAS** GEMM for `matmul`
  - **cuDNN** acceleration for `conv2d` and `conv2d_transpose`
- CUDA implementations for core ops: **pooling**, **reductions**, **elementwise arithmetic**, and **in-place scalar ops** (optimizer-friendly)
- Extensive **unit tests** (CPU↔CUDA parity) and standalone **microbenchmarks** under `scripts/`
- Keras-style training loop (`Model.fit`) with callbacks (EarlyStopping, ModelCheckpoint) and JSON checkpointing

> **Status: v2.1.0b1** (beta). The v2 public API is largely stable and evolving toward v2.1.
> Breaking changes are avoided when possible and documented when necessary.

> **Docs:** https://keywind127.github.io/keydnn_v2/  
> **Source:** https://github.com/keywind127/keydnn_v2

> **Documentation:** API reference is being expanded incrementally (mkdocstrings),
> with an emphasis on examples, architecture, and tested usage patterns.

---

## Project Goals

- Implement core deep learning primitives (Tensor, Parameter, Modules) from first principles
- Clearly separate **domain contracts** from **infrastructure implementations**
- Explore design trade-offs around **autograd**, **device abstraction (CPU/CUDA)**, and **framework extensibility**
- Serve as a pedagogical and experimental alternative to large frameworks

---

## Installation

> **Platform support (v2.x):** Windows 10/11 x64 is the supported platform (CPU + CUDA).
> Linux/macOS may build for CPU-only use, but are not officially supported or CI-validated yet.

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

### CUDA notes

- CUDA execution requires a compatible NVIDIA GPU and a working CUDA runtime.
- Some backends may rely on vendor libraries (e.g., cuBLAS / cuDNN) depending on your build configuration.
- If CUDA native libraries are unavailable, CUDA tests are skipped and CUDA execution paths will raise or fall back where explicitly documented.
- Use `keydnn.backend.cuda_available()` to gate optional CUDA execution at runtime.

### CUDA setup (Windows)

KeyDNN’s Windows CUDA backend loads a native DLL and relies on the CUDA runtime
(and optionally cuDNN) being discoverable by the current process.

#### Environment variables

- `CUDA_PATH` (recommended): points to your CUDA install root, e.g.
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2`
- `CUDNN_PATH` (optional): points to your cuDNN root that contains `bin/`, `lib/`, `include/`,
  e.g. `C:\cudnn`

If you copied cuDNN DLLs into the CUDA install (common manual setup), you typically do **not**
need `CUDNN_PATH` as long as `cudnn*.dll` exists in `<CUDA_PATH>\bin`.

#### PowerShell examples

```powershell
# For the current terminal session only:
$env:CUDA_PATH  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
$env:CUDNN_PATH = "C:\cudnn"   # optional (only needed if cuDNN is not in CUDA\bin)
```

> Note: If you change environment variables, restart the Python process (and sometimes the terminal)
> before retrying.

## Quickstart

### Minimal Tensor + autograd

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
print(y.item())

```

### Reproducibility (random seed + determinism)

KeyDNN provides two separate knobs for reproducibility:

- **Random seeding** controls _random number generation_ used by Python/NumPy (e.g., weight initialization).
- **Determinism policy** controls _execution nondeterminism_ that can arise from CPU parallelism (e.g., OpenMP/BLAS thread scheduling).

#### (1) Seed Python + NumPy RNGs

Call `seed()` once at the beginning of your script (before model construction / initialization):

```python
from keydnn.utils import random

random.seed(42)

# alternative: from keydnn.presentation.apis.utils.random import seed
```

This seeds:

- Python `random`
- NumPy global RNG (`np.random`)

With no multiprocessing dataloader, this is typically sufficient for reproducible
CPU/NumPy behavior (initializers, shuffling in user code, etc.).

#### (2) Configure CPU determinism (threading)

For CPU-only runs, numerical libraries may use multiple threads (OpenMP/MKL/OpenBLAS),
which can introduce small run-to-run differences due to floating-point accumulation
order. To reduce this nondeterminism:

```python
from keydnn.utils import determinism

determinism.set_deterministic(True)  # defaults to cpu_threads=1
# or explicitly:
determinism.set_deterministic(True, cpu_threads=1)

# alternative: from keydnn.presentation.apis.utils.determinism import set_deterministic
```

If you want KeyDNN to **not** modify thread-related environment variables:

```python
determinism.set_deterministic(True, cpu_threads=None)
```

> Note: Thread-related environment variables may need to be set **before importing NumPy**
> (or any BLAS-backed library) to take full effect in the current process.

#### (3) Recommended order

```python
from keydnn.utils import determinism, random

random.seed(42)
determinism.set_deterministic(True)

# build / initialize model after reproducibility is configured
model = ...
```

#### CUDA determinism

CUDA determinism (cuDNN/cuBLAS) and device-side RNG seeding are handled separately
and will be exposed once the native backend configuration surface is available.

### CLI demo (MNIST MLP & CIFAR10 CNN smoke test)

KeyDNN includes a small runnable training example wired through the package CLI:

```bash
# CPU (always available)
python -m keydnn test --train_mnist_example --device cpu --epochs 4 --limit-train 50000 --limit-test 1000

# CUDA (if CUDA backend + native libraries are available)
python -m keydnn test --train_mnist_example --device cuda:0 --epochs 4 --limit-train 50000 --limit-test 1000

# Device: cuda:0
# Train samples: 50000 | Test samples: 1000
# MLP: 784 -> 256 -> 10 | lr=0.1 | batch=128 | epochs=4
# Loss: MSE(one-hot) | Metric: acc(argmax logits)
# Epoch 01/4 | train_loss=0.0569 train_acc=0.7513 | test_acc=0.8560 | 1.99s
# Epoch 02/4 | train_loss=0.0364 train_acc=0.8800 | test_acc=0.8880 | 2.03s
# Epoch 03/4 | train_loss=0.0307 train_acc=0.9045 | test_acc=0.8990 | 2.22s
# Epoch 04/4 | train_loss=0.0274 train_acc=0.9174 | test_acc=0.9070 | 2.11s
```

```bash
# CUDA (if CUDA backend + native libraries are available)
python -m keydnn test --train_cifar_example --device cuda:0 --epochs 4 --limit-train 50000 --limit-test 1000

# Device: cuda:0
# Train samples: 50000 | Test samples: 1000
# CNN CIFAR-10 | lr=0.1 | batch=128 | epochs=4 | normalize=False
# Loss: MSE(one-hot) | Metric: acc(argmax logits)
# Epoch 01/4 | train_loss=0.1236 train_acc=0.2213 | test_acc=0.3010 | 10.87s
# Epoch 02/4 | train_loss=0.0813 train_acc=0.3351 | test_acc=0.3550 | 10.18s
# Epoch 03/4 | train_loss=0.0788 train_acc=0.3770 | test_acc=0.3900 | 10.97s
# Epoch 04/4 | train_loss=0.0771 train_acc=0.4059 | test_acc=0.4090 | 10.45s
```

> Tip: For CUDA runs, keep `--limit-test` modest or evaluate in batches (future releases aim to reduce remaining large-batch limitations).

#### Notes:

- The first run will download and cache MNIST under `~/.cache/keydnn/mnist/raw` (configurable via `--root`).
- CUDA execution requires a compatible NVIDIA GPU and a working CUDA runtime (and may be skipped/raise if unavailable).

### Inference batching on CUDA (important)

When running on CUDA, KeyDNN currently expects inference/evaluation to be performed in **mini-batches** (e.g., `batch_size=128`)
rather than passing an entire dataset tensor (e.g., `N=10000`) through the model at once.

This is both a standard deep-learning practice (memory/performance) and avoids current CUDA kernel launch limits in some ops
(e.g., padding used by Pool2D). If you hit a runtime error during evaluation with a large `N`, rerun inference in batches.

### Training example (`Model.fit` + callbacks)

```python
import numpy as np

from keydnn.callbacks import EarlyStopping, ModelCheckpoint
from keydnn.backend import cuda_available
from keydnn.tensors import Tensor, Device
from keydnn.utils import numpy_to_tensor
from keydnn.activations import Sigmoid
from keydnn.models import Sequential
from keydnn.layers import Linear


def _xor_data_numpy():
    x_np = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)
    return x_np, y_np


def _accuracy_from_pred_np(y_true_np: np.ndarray, pred_np: np.ndarray) -> float:
    y_hat = (pred_np >= 0.5).astype(np.float32)
    return float((y_hat == y_true_np).mean())


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = Device("cuda:0") if cuda_available() else Device("cpu")

    # ------------------------------------------------------------------
    # XOR dataset
    # ------------------------------------------------------------------
    x_base, y_base = _xor_data_numpy()
    repeats = 256
    x_np = np.repeat(x_base, repeats=repeats, axis=0)
    y_np = np.repeat(y_base, repeats=repeats, axis=0)

    x = numpy_to_tensor(x_np, device=device)
    y = numpy_to_tensor(y_np, device=device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    hidden_dim = 8

    model = Sequential(
        Linear(2, hidden_dim),
        Sigmoid(),
        Linear(hidden_dim, 1),
        Sigmoid(),
    )

    model.to_(device)

    # (N, 2)
    model.build((1, 2), device=device)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        EarlyStopping(
            monitor="acc",
            mode="max",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath="xor_epoch{epoch:03d}_loss{loss:.6f}.json",
            monitor="acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history = model.fit(
        x,
        y,
        loss="mse",
        optimizer="sgd",
        optimizer_kwargs={"lr": 1.0},
        metrics=["acc"],
        batch_size=32,
        epochs=2000,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    x_eval = numpy_to_tensor(x_base, device=device)
    pred: Tensor = model(x_eval)
    pred_np = np.asarray(pred.to_numpy(), dtype=np.float32)

    print("device:", str(device))
    print("pred:", pred_np.reshape(-1).round(3).tolist())
    print("acc:", _accuracy_from_pred_np(y_base, pred_np))
```

### Keras → KeyDNN model conversion (Sequential)

KeyDNN can convert a supported **Keras Sequential** model into an equivalent KeyDNN module graph,
preserving weights deterministically.

```python
import numpy as np

import keydnn as kd
from keydnn import from_keras

# Optional dependency: TensorFlow/Keras
from tensorflow.keras.layers import Dense, Input, Softmax, ReLU
from tensorflow.keras.models import Sequential


def build_keras_model() -> Sequential:
    # Tip: explicit activation layers are recommended for the cleanest conversion.
    return Sequential(
        [
            Input(shape=(2,)),
            Dense(32, activation="linear", name="hidden"),
            ReLU(),
            Dense(2, activation="linear", name="output"),
            Softmax(axis=-1),
        ]
    )


if __name__ == "__main__":
    keras_model = build_keras_model()

    # Convert Keras -> KeyDNN (strict by default)
    kd_model = from_keras(keras_model)

    # Run a quick forward pass and compare outputs
    x_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    y_keras = keras_model.predict(x_np)

    x_kd = kd.numpy_to_tensor(x_np, device=kd.Device("cpu"))
    y_kd = kd_model(x_kd).to_numpy()

    print("keras:", y_keras)
    print("keydnn:", y_kd)

    # (Optional) Save the converted KeyDNN model as a JSON checkpoint
    kd_model.save_json("converted_model.json")

```

> Notes:
>
> - Conversion currently targets **Keras Sequential** models and a supported layer subset.
> - Keras fused activations (e.g., `Dense(..., activation="relu")`) may be normalized into explicit KeyDNN activation modules.
> - For CLI conversion, see: `python -m keydnn convert --src model.h5 --dst model.json`

---

## Feature / Backend Support Matrix

| Area                                          | CPU | CUDA | Notes                                                     |
| --------------------------------------------- | --: | ---: | --------------------------------------------------------- |
| Tensor core + autograd                        |  ✅ |   ✅ | reverse-mode AD with dynamic computation graphs           |
| Elementwise ops (add/sub/mul/div/neg/compare) |  ✅ |   ✅ | float32/float64 paths where supported                     |
| In-place ops (optimizer hot paths)            |  ✅ |   ✅ | avoids temporary tensor materialization on CUDA           |
| Reductions (sum/mean/max)                     |  ✅ |   ✅ | axis support varies by op; tested with parity checks      |
| Pooling (max/avg/global avg)                  |  ✅ |   ✅ | CUDA kernels + correctness tests                          |
| Matmul / transpose                            |  ✅ |   ✅ | CUDA uses **cuBLAS** GEMM where available                 |
| Conv2D                                        |  ✅ |   ✅ | CUDA backend with **cuDNN** acceleration where available  |
| Conv2D Transpose                              |  ✅ |   ✅ | CUDA backend with **cuDNN** acceleration where available  |
| Indexing / slicing (`__getitem__`)            |  ✅ |   ⚠️ | CUDA path may use a correctness-first CPU fallback        |
| RNN modules (RNN/LSTM/GRU)                    |  ✅ |   ⚠️ | implemented via Tensor ops; no fused CUDA RNN kernels yet |
| Normalization (BatchNorm1d/2d, LayerNorm)     |  ✅ |   ⚠️ | LayerNorm is CPU; CUDA coverage varies by module          |
| Concatenation (`Tensor.concat`)               |  ✅ |   ⚠️ | CUDA path may use a correctness-first CPU fallback        |
| Training loop (`Model.fit`, `train_on_batch`) |  ✅ |   ✅ | Keras-like training APIs; works with CPU/CUDA tensors     |
| Callbacks (EarlyStopping, ModelCheckpoint)    |  ✅ |   ✅ | hook wiring via CallbackList; JSON checkpoint integration |

---

## Architecture Overview

KeyDNN exposes its intended public surface under `keydnn.presentation.apis`.
These modules provide stable import paths and aliases, while internal infrastructure
packages may evolve as long as they satisfy domain interfaces.

### Domain

Defines framework **contracts** using structural typing (`Protocol`):

- `ITensor` — tensor interface
- `IParameter` — trainable tensor interface
- `IModule` — neural network module/layer interface

Domain code is backend-agnostic and contains no NumPy or CUDA logic.

---

### Infrastructure

> Note: The lists below are intentionally comprehensive. For a quick overview of backend support and maturity, see the **Feature / Backend Support Matrix** above.

Provides concrete implementations of domain contracts:

- `Tensor` — CPU (NumPy-backed) + CUDA (device-pointer–backed) tensor
  - CUDA tensors are backed by device allocations and explicit memcpy helpers (H2D / D2H / D2D)
  - Backend dispatch is explicit and validated (shape/device/dtype checks; no silent mixed-device execution)
- `Context` — backward propagation context for dynamic computation graphs
- `Parameter` — trainable tensor with gradient semantics
  - supports `initializer=...` to apply `WeightInitializer` in-place at construction time
- `WeightInitializer` — registry-based initializer dispatcher (Xavier/Kaiming/zeros/ones, activation-aware variants)
- `Module` — base class for neural network layers
- `Model` — top-level network abstraction with inference utilities
- `Sequential` — ordered container for composing multi-layer models
- `Linear` — fully connected (dense) layer implementation
- `BatchNorm1d` — feature-wise batch normalization for dense / sequence models
- `BatchNorm2d` — channel-wise batch normalization for convolutional models
- `Flatten` — reshape layer bridging spatial outputs to dense layers
- `Conv2d` — 2D convolution layer (NCHW)
  - Naive NumPy reference implementation (forward + backward)
  - Optional native CPU acceleration (float32 / float64) via C++ + ctypes
    - Single-threaded native C++ kernels
    - OpenMP-parallelized native kernels for compute-heavy workloads
  - Automatic dtype-aware dispatch with safe fallback
- `Conv2dTranspose` — 2D transposed convolution layer (NCHW)
  - CPU reference implementation + native CPU backend (where available)
  - CUDA backend with **cuDNN** acceleration where available
  - Tested via forward/backward parity checks and module/Fn unit tests
- Recurrent modules:
  - Cells: `RNNCell`, `LSTMCell`, `GRUCell`
  - Sequence modules: `RNN`, `LSTM`, `GRU`
  - `Bidirectional` wrapper (Keras-style)
  - Time-major execution with BPTT and `return_sequences` / `return_state`
- Keras-style bidirectional execution via `Bidirectional`
  - Generic wrapper supporting `RNN`, `LSTM`, and `GRU`
  - Forward/backward layer cloning
  - Correct backward-time alignment
- Pooling layers:
  - `MaxPool2d`
  - `AvgPool2d`
  - `GlobalAvgPool2d`
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (SSE, MSE, Binary Cross Entropy, Categorical Cross Entropy)
- Optimizers (SGD, Adam)
- Autograd execution engine via dynamic computation graphs (`Context`, `Tensor.backward`)
- Tensor indexing and slicing (`Tensor.__getitem__`)
  - Supports basic slicing, integer indexing, fancy indexing, and boolean masks
  - Scatter-based gradient propagation for correct backward behavior
- Tensor stacking (`Tensor.stack`)
  - Enables sequence-level outputs with gradient distribution to individual tensors
  - Critical for recurrent models and BPTT
- Convolution and pooling primitives:
  - Conv2D forward/backward kernels
    - Naive NumPy reference implementations
    - Optional native CPU acceleration via C++ + ctypes (float32 / float64)
    - Automatic dispatch with fallback for unsupported dtypes
  - Pool2D forward/backward kernels
    - AvgPool2D and MaxPool2D native CPU kernels via ctypes
    - Float32 / Float64 dtype-aware dispatch
    - Safe fallback to NumPy reference implementations
  - Autograd integration via `Conv2dFn` and pooling `Function`s
- **Weight initialization framework** via `WeightInitializer` registry (Xavier/Kaiming + activation-aware variants, zeros/ones)
  - `Parameter(initializer=...)` initializes tensors in-place at construction time (device/dtype preserved)

Infrastructure code is free to evolve independently as long as it satisfies domain interfaces.

---

### Native CPU Acceleration (Optional)

KeyDNN supports optional native CPU acceleration for selected operations
via C++ kernels exposed through `ctypes`, including:

- Conv2D (forward and backward)
- AvgPool2D
- MaxPool2D
- Native kernels are used automatically for supported dtypes (`float32`, `float64`)
- If the native shared library is unavailable, KeyDNN emits a runtime warning
  and safely falls back to NumPy reference implementations
- Build scripts are provided for Windows (MinGW), Linux, and macOS
- Native acceleration is an optimization layer only; correctness is always
  validated against reference implementations via unit tests

This design allows incremental performance optimization without sacrificing
portability or debuggability.

## CUDA Backend (Implemented)

KeyDNN includes a CUDA execution backend with device-resident tensor storage and tested CUDA kernels.

> Note: The CUDA backend is currently supported on **Windows**. Linux CUDA support will be added once cross-platform CUDA builds are finalized.

### Design principles

- CUDA tensors are backed by **device pointers** (no implicit host materialization)
- Python↔CUDA boundaries are explicit and centralized (ctypes wrappers + ops-layer helpers)
- Correctness is locked in via **CPU↔CUDA parity tests**, with graceful skips when CUDA natives are unavailable

### Implemented CUDA capabilities

- Tensor ops: elementwise arithmetic, comparisons, unary ops (e.g., `exp`)
- Reductions: `sum`, `mean`, `max` (including backward fill/scatter where applicable)
- Pooling: MaxPool2D / AvgPool2D / GlobalAvgPool2D forward + backward kernels
- Linear algebra:
  - `matmul` accelerated via **cuBLAS GEMM** where available
  - `transpose` and memcpy utilities (H2D/D2H/D2D)
- Conv2D:
  - CUDA forward/backward backend with **cuDNN acceleration** where available
  - ops/Function/module layers preserve autograd semantics while dispatching by device
- Performance-focused kernels:
  - CUDA **in-place elementwise ops** (e.g., `__iadd__`, `__imul__`) and scalar kernels
  - reduces temporary tensor allocation overhead in optimizer updates

### Known correctness-first fallbacks

Some CUDA features are implemented with correctness-first fallbacks (documented in code and tests), e.g. CUDA `__getitem__` may stage through CPU to match NumPy semantics until native CUDA gather/scatter kernels are added.

---

## Benchmarks

Standalone microbenchmarks live under `scripts/` and are designed to measure kernel/runtime performance while excluding unrelated overhead where possible (e.g., excluding H2D/D2H transfers when isolating device kernel speed).

Typical scripts include:

```bash
python scripts/bench_linear_backward_cpu_vs_cuda.py --dtype float32 --sanity
python scripts/bench_pool2d_cpu_vs_cuda.py --dtype float32 --sanity
python scripts/bench_conv2d_cpu_vs_cuda.py --dtype float32 --sanity
python scripts/bench_tensor_ops_cpu_vs_cuda.py --dtype float32 --sanity
```

### Linear Layer Backward (CPU vs CUDA)

Benchmark configuration:

- dtype: float32
- warmup: 10 iterations
- repeats: 50
- sanity check: enabled

| Batch |   In |  Out | Bias | CPU Backward | CUDA Backward |   Speedup |
| ----: | ---: | ---: | :--: | -----------: | ------------: | --------: |
|  1024 | 1024 | 1024 | Yes  |     26.93 ms |       3.79 ms | **7.10×** |

This benchmark isolates backward propagation cost for a single `Linear` layer.
CUDA acceleration significantly reduces backward time for realistically sized dense layers.

### Pool2D Forward (CPU vs CUDA)

Benchmark configuration:

- shape: N=8, C=32, H=W=64
- kernel: 2×2, stride=2, padding=0
- dtype: float32
- warmup: 10 iterations
- repeats: 50
- sanity check: enabled

|           Operation | CPU Time | CUDA Time |    Speedup |
| ------------------: | -------: | --------: | ---------: |
| MaxPool2D (forward) |  6.49 ms | 278.15 µs | **23.32×** |
| AvgPool2D (forward) |  2.69 ms | 181.55 µs | **14.84×** |

These results show that eliminating Python loop overhead and dispatching to native CUDA
kernels yields large speedups for spatial pooling on realistic CNN feature maps.

### Tensor Elementwise Operations (CPU vs CUDA)

Benchmark configuration:

- shape: (512, 512)
- dtype: float32
- warmup: 50 iterations
- repeats: 200
- CUDA synchronization: disabled (`--sync_each_iter=False`)
- sanity check: enabled

|  Operation | CPU Median | CUDA Median |   Speedup |
| ---------: | ---------: | ----------: | --------: |
|        add |   723.8 µs |    126.4 µs | **5.73×** |
|        sub |   725.8 µs |    121.2 µs | **5.99×** |
|        mul |   759.9 µs |    138.4 µs | **5.49×** |
|        div |   729.6 µs |    128.1 µs | **5.70×** |
|         gt |   847.6 µs |    129.6 µs | **6.54×** |
|        neg |   679.2 µs |    114.4 µs | **5.93×** |
|        exp |   902.0 µs |    160.4 µs | **5.62×** |
| add_scalar |   1.036 ms |    121.8 µs | **8.51×** |
| mul_scalar |   1.048 ms |    133.2 µs | **7.87×** |

> Note: CUDA timings may be optimistic without explicit synchronization, as kernel
> launches are asynchronous. Relative speedups remain representative for compute-heavy paths.

These results highlight the benefit of CUDA in-place and scalar kernels, especially for
optimizer hot paths that previously required temporary tensor materialization.

### Conv2D Forward (CPU vs CUDA, cuDNN)

Benchmark configuration:

- N=8, Cin=64, Cout=128
- spatial size: 112×112
- kernel: 3×3, stride=1, padding=1
- bias: enabled
- dtype: float32
- warmup: 10 iterations
- repeats: 50
- sanity check: enabled

|            Configuration | CPU Time | CUDA Time |    Speedup |
| -----------------------: | -------: | --------: | ---------: |
| 8×64×112×112 → 128 (3×3) |  1.024 s |  64.51 ms | **15.88×** |

This benchmark uses the CUDA Conv2D backend with **cuDNN acceleration** and excludes
host↔device transfer overhead, isolating kernel execution performance.

### Summary

Across dense layers, pooling, elementwise tensor ops, and Conv2D workloads,
KeyDNN’s CUDA backend consistently delivers **5×–25× speedups** over CPU execution
for realistically sized tensors, with Conv2D benefiting further from vendor-
optimized libraries such as **cuDNN**.

These benchmarks validate both correctness (sanity checks enabled) and the
practical performance gains of the current CUDA execution stack.

### Notes on benchmarking methodology

- Benchmarks typically use warmup iterations + repeated timings with median reporting
- Some benchmarks optionally synchronize CUDA to produce accurate timing
- Conv2D benchmarks may pre-pad inputs and pre-allocate outputs outside the timed region to isolate the core kernel path

#### Benchmark Environment

All benchmarks were collected on a single-machine development setup:

- CPU: Intel i5-12450H
- GPU: NVIDIA RTX 3050
- RAM: 32 GB
- OS: Windows 11
- CUDA: CUDA 12.4
- Libraries: cuBLAS, cuDNN (vendor-provided)

Benchmarks focus on **relative CPU vs CUDA speedups** and typically exclude
host↔device transfer overhead unless explicitly stated.

---

#### OpenMP Parallelization (CPU)

This section is primarily relevant for CPU-only execution and kernel development.

For compute-intensive CPU kernels (Conv2D, Pool2D), KeyDNN optionally enables
**OpenMP-based intra-kernel parallelism** in the native C++ backend.

Design rationale:

- Python multiprocessing introduces high overhead from process creation,
  inter-process communication, and tensor marshaling.
- Native kernels invoked via `ctypes` already execute outside the Python GIL.
- OpenMP enables fine-grained parallelism directly inside hot loops
  without crossing the Python/C boundary.

Behavioral characteristics:

- OpenMP provides additional speedups (up to ~3×) for sufficiently large
  Conv2D workloads.
- For small tensor shapes, thread scheduling overhead may dominate and
  reduce gains.
- OpenMP acceleration is workload-dependent and conservatively enabled
  only at the native kernel level.

On Windows (MinGW-w64), OpenMP runtime DLLs are staged next to the native
shared library to ensure reliable dynamic loading via `ctypes`.

---

#### Performance Evaluation (Pool2D)

Standalone benchmark scripts were used to evaluate the impact of native
C++ Pool2D kernels compared to Python-based implementations.

Three implementations were compared:

1. Pure Python reference (explicit nested loops)
2. NumPy-based reference implementation
3. Native C++ kernels compiled as shared libraries and loaded via `ctypes`

Representative results on CPU (float32):

| Shape (N,C,H,W) | Operation     | Speedup vs Python | Speedup vs NumPy |
| --------------- | ------------- | ----------------- | ---------------- |
| 1×8×28×28       | MaxPool2D fwd | ~150×             | ~240×            |
| 8×16×32×32      | AvgPool2D fwd | ~400×             | ~800×            |
| 8×32×64×64      | MaxPool2D fwd | ~250×             | ~210×            |
| 16×64×56×56     | AvgPool2D fwd | ~430×             | ~890×            |

These results demonstrate that eliminating Python loop overhead—rather than
micro-optimizing NumPy expressions—is the dominant factor in accelerating
Pool2D operations.

Benchmark scripts are provided under `scripts/` for reproducibility and are
not part of the unit test suite.

#### Performance Evaluation (Conv2D)

Standalone benchmark scripts were used to evaluate the impact of native
C++ Conv2D kernels compared to Python-based implementations.

Three implementations were compared:

1. Pure Python reference (explicit nested loops)
2. NumPy-based reference implementation (Python loops with NumPy reductions)
3. Native C++ kernels compiled as shared libraries and loaded via `ctypes`

Benchmarks measure **forward Conv2D only**, isolating kernel performance by:

- Pre-padding inputs outside the timed region
- Preallocating outputs outside the timed region
- Loading the native shared library once per run

Representative results on CPU (float32, bias enabled):

| Case       | Shape (N,C_in,H,W → C_out) | Speedup vs Python | Speedup vs NumPy |
| ---------- | -------------------------- | ----------------: | ---------------: |
| mnist-ish  | 1×8×28×28 → 8              |             ~370× |             ~42× |
| tiny       | 1×8×16×16 → 8              |             ~460× |             ~59× |
| small      | 1×16×32×32 → 16            |             ~340× |             ~21× |
| downsample | 1×8×28×28 → 8 (stride=2)   |             ~480× |             ~56× |

These results highlight three distinct performance regimes:

- **Pure Python Conv2D** is dominated by interpreter overhead from deeply
  nested loops and is several hundred times slower than native code.
- **NumPy-based reference Conv2D** significantly improves performance by
  using vectorized reductions, but still incurs Python control-flow and
  slicing overhead.
- **Native C++ Conv2D kernels** eliminate Python overhead entirely in the
  hot path, yielding consistent **20×–60× speedups over NumPy** and
  **300×–480× speedups over pure Python**.

Conservatively summarized, native Conv2D forward execution in KeyDNN
achieves an average **~50× speedup over NumPy-based implementations** on
CPU while preserving correctness and portability.

Additional benchmarking comparing **native C++ with and without OpenMP**
shows that OpenMP provides incremental acceleration only for sufficiently
large workloads. Representative results indicate:

- Up to ~3× speedup for larger spatial resolutions and channel counts
- Minimal or negative gains for small tensors due to threading overhead

This behavior matches expected CPU parallelization trade-offs and validates
OpenMP as a scalable optimization path for realistic CNN workloads.

Benchmark scripts and full timing reports are provided under `scripts/`
and are not part of the unit test suite.

#### Native Build & Reproducibility

Native CPU kernels are built from source and are **not required** to use KeyDNN.

- Platform-specific build scripts are provided under `scripts/`
- Windows builds use MinGW-w64 and support both OpenMP and non-OpenMP variants
- Required OpenMP runtime DLLs are staged automatically for reliable loading
- Environment variables (e.g., compiler paths) are centralized in `.env`
  to ensure reproducible builds across systems

---

## Model Serialization (JSON Checkpoints)

KeyDNN supports saving and loading complete models (architecture + weights)
to a single JSON file.

### Format

A checkpoint JSON has the form:

```json
{
  "format": "keydnn.json.ckpt.v1",
  "arch": { "...": "module config tree" },
  "state": {
    "layer_name.weight": { "b64": "...", "dtype": "<f4", "shape": [..], "order": "C" }
  }
}
```

- `arch` stores a recursive module configuration tree (`type`, `config`, `children`)
- `state` stores trainable tensors serialized as base64-encoded raw bytes
- Stateless modules (e.g., activations/pooling/flatten) serialize via config only

### Usage

```python
from keydnn.layers import Conv2d, Linear, MaxPool2d, Flatten
from keydnn.activations import ReLU, Softmax
from keydnn.models import Sequential

model = Sequential(
    Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Flatten(),
    Linear(in_features=8 * 14 * 14, out_features=10),
    Softmax(axis=-1),
)

model.save_json("checkpoint.json")
loaded = Sequential.load_json("checkpoint.json")
```

#### Supported layers for JSON save/load

- Containers: `Sequential`
- Trainable: `Linear`, `Conv2d`, `RNNCell`, `RNN`
- Stateless: `Flatten`, pooling modules, activation modules

---

### Tests

The test suite is split into two categories:

- **Interface compatibility tests**
  - Ensure infrastructure classes conform to domain `Protocol`s
- **Behavioral unit tests**
  - Validate tensor, parameter, module, and layer behavior
  - Avoid private attribute access
- Integration tests validating end-to-end training on nonlinear tasks (e.g., XOR)
- Unit tests validating Conv2D and Pool2D ops, autograd functions, and module behavior
- Edge-case tests for pooling semantics (tie-breaking, padding traps)
- Shape matrix tests for stride/padding correctness
- Chain tests validating Conv2D → Pooling → Activation compatibility
- JSON checkpointing tests (save/load):
  - forward output equivalence before/after load
  - parameter round-trip correctness for trainable layers
  - config/hyperparameter preservation for stateless layers
  - failure modes for missing keys / malformed configs
  - registry coverage tests to prevent missing serialization hooks
- End-to-end CNN chain tests (Conv2D → ReLU → MaxPool2D → Flatten → Linear → Softmax)
- End-to-end CNN composition validated via chain tests (Conv2D → Pooling → Flatten → Dense)
- Finite-difference gradient checks for Conv2D forward and backward
- Native vs reference path consistency tests via mocked dispatch
- Recurrent Neural Network (RNN) tests:
  - RNNCell forward correctness vs NumPy reference
  - RNNCell backward gradient propagation (inputs, hidden state, parameters)
  - End-to-end BPTT gradient flow across time steps
  - Training sanity tests (sequence fitting with loss decrease)
  - Hidden-state (h0) gradient correctness and edge cases

---

## Implemented Features

- Tensor abstraction with CPU (NumPy) backend and arithmetic operator overloading
- Device abstraction (`cpu`, `cuda:<index>`)
- Trainable `Parameter` class with gradient management
- Module system with parameter registration
- Weight initialization via `WeightInitializer` registry (Xavier/Kaiming/zeros/ones and activation-aware variants)
- `Parameter(initializer=...)` for in-place initialization at construction time
- 2D transposed convolution layer (`Conv2dTranspose`) with CPU/CUDA backends (cuDNN where available)
- Fully connected `Linear` layer
- `Flatten` layer for reshaping (N, C, H, W) → (N, C*H*W)
- 2D convolution layer (`Conv2d`) with configurable kernel size, stride, padding, and optional bias
- 2D pooling layers (`MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`)
- Batch normalization layers:
  - BatchNorm1d (dense / sequence features)
  - BatchNorm2d (convolutional channels)
- CPU Conv2D and Pool2D forward/backward kernels with reference and native implementations
- Autograd-compatible pooling functions with correct gradient routing
- Model checkpointing: JSON save/load (architecture config + base64-encoded weights)
- Regression and classification loss functions (SSE, MSE, BCE, CCE)
- Softmax activation module with numerically stable forward and efficient backward
- Tensor reduction operations (`numel`, `sum`, `mean`)
- Tensor concatenation (`Tensor.concat`) with backward gradient splitting
- Dynamic computation graph metadata via `Context`
- Domain-level interfaces using `Protocol` (duck typing)
- Comprehensive unit tests for contracts and behavior
- Reverse-mode automatic differentiation (autograd) with dynamic computation graphs
- End-to-end model training via backpropagation and optimizers
- Sequential model composition with parameter discovery
- Recurrent neural networks:

  - `RNNCell`, `LSTMCell`, `GRUCell`
  - `RNN`, `LSTM`, `GRU` sequence modules
  - Time-major execution with Backpropagation Through Time (BPTT)
  - Keras-compatible `return_sequences` / `return_state` semantics
  - Generic `Bidirectional` wrapper supporting RNN, LSTM, and GRU
  - Gradient propagation across time and both directions

  ```python
  from keydnn.layers import Bidirectional, LSTM

  model = Bidirectional(
      LSTM(input_size=16, hidden_size=32, return_sequences=True), return_sequences=True
  )
  ```

---

### Supported Layers

- Linear (fully connected)
- BatchNorm1d
- BatchNorm2d
- Flatten
- Conv2d (2D convolution, NCHW)
- Conv2dTranspose (2D transposed convolution, NCHW)
- Recurrent:
  - RNNCell, LSTMCell, GRUCell
  - RNN, LSTM, GRU
  - Bidirectional (generic wrapper for recurrent modules)
- Pooling layers:
  - MaxPool2d
  - AvgPool2d
  - GlobalAvgPool2d
- Activation layers:
  - ReLU
  - Sigmoid
  - Softmax
  - Tanh
  - LeakyReLU
- JSON checkpoint support via get_config / from_config hooks (for supported modules)

---

## Roadmap

### Next (near-term)

- Packaging polish: wheels where feasible, improve install story, add CI
- Add CI: CPU-only by default, optional CUDA jobs for environments with GPUs
- Reduce remaining correctness-first CPU fallbacks on CUDA (e.g., indexing/concat general-axis paths)
- Improve developer docs: architecture diagram, backend boundaries, and contributor guide

### Later

- Cross-platform build support improvements (formalize Linux/macOS support with CI and packaging; CUDA support beyond Windows as available)
- Additional CUDA kernel coverage and optimizations (fusion opportunities, reduced allocations, async-friendly paths)
- Import utilities (scoped): partial model/weight conversion from PyTorch/Keras (explicit supported subset)
- Fused/faster RNN kernels (optional): device-optimized sequence ops beyond Tensor-op composition
- Checkpoint format evolution: versioning, migrations, partial loading, optional binary layouts

---

## Disclaimer

KeyDNN is a research/learning-oriented framework and is not production-hardened.
The v2 public API is intended to be stable; internal design and backend coverage will continue to evolve.

The project prioritizes:

- correctness (verified by tests),
- clear architecture boundaries,
- and incremental backend performance improvements (native CPU / CUDA).

---

## License

Licensed under the **Apache License, Version 2.0**.

You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
