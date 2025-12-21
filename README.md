# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python, designed to explore and demonstrate the core abstractions behind modern neural network libraries.

The project emphasizes **clean architecture**, **explicit interfaces**, and **separation of concerns**, making it suitable for learning, experimentation, and incremental extension.

---

## Project Goals

- Implement core deep learning primitives (Tensor, Parameter, Modules) from first principles
- Clearly separate **domain contracts** from **infrastructure implementations**
- Explore design trade-offs around **autograd**, **device abstraction (CPU/CUDA)**, and **framework extensibility**
- Serve as a pedagogical and experimental alternative to large frameworks

---

## Architecture Overview

KeyDNN follows a layered, interface-driven design:

### Domain

Defines framework **contracts** using structural typing (`Protocol`):

- `ITensor` — tensor interface
- `IParameter` — trainable tensor interface
- `IModule` — neural network module/layer interface

Domain code is backend-agnostic and contains no NumPy or CUDA logic.

---

### Infrastructure

Provides concrete implementations of domain contracts:

- `Tensor` — NumPy-backed CPU tensor (with CUDA placeholder)
- `Context` — backward propagation context for dynamic computation graphs
- `Parameter` — trainable tensor with gradient semantics
- `Module` — base class for neural network layers
- `Model` — top-level network abstraction with inference utilities
- `Sequential` — ordered container for composing multi-layer models
- `Linear` — fully connected (dense) layer implementation
- `Flatten` — reshape layer bridging spatial outputs to dense layers
- `Conv2d` — 2D convolution layer (NCHW)
  - Naive NumPy reference implementation (forward + backward)
  - Optional native CPU acceleration (float32 / float64) via C++ + ctypes
  - Automatic dtype-aware dispatch with safe fallback
- Pooling layers:
  - `MaxPool2d`
  - `AvgPool2d`
  - `GlobalAvgPool2d`
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (SSE, MSE, Binary Cross Entropy, Categorical Cross Entropy)
- Optimizers (SGD, Adam)
- Autograd execution engine via dynamic computation graphs (`Context`, `Tensor.backward`)
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

| Case        | Shape (N,C_in,H,W → C_out) | Speedup vs Python | Speedup vs NumPy |
|-------------|-----------------------------|------------------:|-----------------:|
| mnist-ish  | 1×8×28×28 → 8               | ~370×             | ~42×             |
| tiny       | 1×8×16×16 → 8               | ~460×             | ~59×             |
| small      | 1×16×32×32 → 16             | ~340×             | ~21×             |
| downsample | 1×8×28×28 → 8 (stride=2)    | ~480×             | ~56×             |

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

Benchmark scripts and full timing reports are provided under `scripts/`
and are not part of the unit test suite.

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
- End-to-end CNN chain tests (Conv2D → ReLU → MaxPool2D → Flatten → Linear → Softmax)
- End-to-end CNN composition validated via chain tests (Conv2D → Pooling → Flatten → Dense)
- Finite-difference gradient checks for Conv2D forward and backward
- Native vs reference path consistency tests via mocked dispatch

---

## Implemented Features

- Tensor abstraction with CPU (NumPy) backend and arithmetic operator overloading
- Device abstraction (`cpu`, `cuda:<index>`)
- Trainable `Parameter` class with gradient management
- Module system with parameter registration
- Fully connected `Linear` layer
- `Flatten` layer for reshaping (N, C, H, W) → (N, C*H*W)
- 2D convolution layer (`Conv2d`) with configurable kernel size, stride, padding, and optional bias
- 2D pooling layers (`MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`)
- CPU Conv2D and Pool2D forward/backward kernels with reference and native implementations
- Autograd-compatible pooling functions with correct gradient routing
- Regression and classification loss functions (SSE, MSE, BCE, CCE)
- Softmax activation module with numerically stable forward and efficient backward
- Tensor reduction operations (`numel`, `sum`, `mean`)
- Dynamic computation graph metadata via `Context`
- Domain-level interfaces using `Protocol` (duck typing)
- Comprehensive unit tests for contracts and behavior
- Reverse-mode automatic differentiation (autograd) with dynamic computation graphs
- End-to-end model training via backpropagation and optimizers
- Sequential model composition with parameter discovery

---

### Supported Layers

- Linear (fully connected)
- Flatten
- Conv2d (2D convolution, NCHW)
- Pooling layers:
  - MaxPool2d
  - AvgPool2d
  - GlobalAvgPool2d
- Activation layers: ReLU, Sigmoid, Softmax

---

## Roadmap (Planned)

- Additional layers and activation functions (beyond core Conv2D and Pooling)
- CUDA-backed tensor operations
- CUDA acceleration for convolution layers
- Performance optimizations and kernel fusion
- Model serialization and checkpointing

---

## Disclaimer

KeyDNN is a **work in progress** and not intended for production use.  
APIs and internal design may change as the framework evolves.

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
