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
    - Single-threaded native C++ kernels
    - OpenMP-parallelized native kernels for compute-heavy workloads
  - Automatic dtype-aware dispatch with safe fallback
- `RNNCell` — vanilla recurrent neural network cell (tanh activation)
  - NumPy-based forward and backward
  - Autograd-compatible via dynamic computation graphs
- `RNN` — sequence-level RNN module
  - Executes `RNNCell` over time-major sequences
  - Supports Backpropagation Through Time (BPTT) via chained autograd contexts
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

#### OpenMP Parallelization (CPU)

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

```python=
from keydnn.infrastructure._models import Sequential
from keydnn.infrastructure._conv2d_module import Conv2d
from keydnn.infrastructure._linear import Linear
from keydnn.infrastructure.pooling._pooling_module import MaxPool2d
from keydnn.infrastructure.nn._flatten_module import Flatten
from keydnn.infrastructure._activations_module import ReLU, Softmax

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
- Fully connected `Linear` layer
- `Flatten` layer for reshaping (N, C, H, W) → (N, C*H*W)
- 2D convolution layer (`Conv2d`) with configurable kernel size, stride, padding, and optional bias
- 2D pooling layers (`MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`)
- CPU Conv2D and Pool2D forward/backward kernels with reference and native implementations
- Autograd-compatible pooling functions with correct gradient routing
- Model checkpointing: JSON save/load (architecture config + base64-encoded weights)
- Regression and classification loss functions (SSE, MSE, BCE, CCE)
- Softmax activation module with numerically stable forward and efficient backward
- Tensor reduction operations (`numel`, `sum`, `mean`)
- Dynamic computation graph metadata via `Context`
- Domain-level interfaces using `Protocol` (duck typing)
- Comprehensive unit tests for contracts and behavior
- Reverse-mode automatic differentiation (autograd) with dynamic computation graphs
- End-to-end model training via backpropagation and optimizers
- Sequential model composition with parameter discovery
- Vanilla recurrent neural networks (RNN)
  - `RNNCell` with tanh nonlinearity
  - `RNN` sequence module with Backpropagation Through Time (BPTT)
  - Gradient propagation through time via dynamic autograd graphs

---

### Supported Layers

- Linear (fully connected)
- Flatten
- Conv2d (2D convolution, NCHW)
- Recurrent:
  - RNNCell (vanilla tanh)
  - RNN (sequence module)
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

## Roadmap (Planned)

- Additional layers and activation functions (beyond core Conv2D and Pooling)
- CUDA-backed tensor operations (following validated CPU OpenMP parallelism)
- CUDA acceleration for convolution layers
- Performance optimizations and kernel fusion
- Expand checkpoint compatibility (versioning, migration utilities, partial loading)
- Add additional formats (optional): compressed JSON, msgpack, or safetensors-like layout
- Advanced recurrent architectures:
  - LSTM
  - GRU
- Fused recurrent kernels for improved performance
- Sequence masking and variable-length sequence support
- Bidirectional and multi-layer RNNs
- Plan: “compression / chunking” or “binary format” (optional)

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
