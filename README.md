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
- `Conv2d` — 2D convolution layer (NCHW, CPU reference implementation)
- Pooling layers:
  - `MaxPool2d`
  - `AvgPool2d`
  - `GlobalAvgPool2d`
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (SSE, MSE, Binary Cross Entropy, Categorical Cross Entropy)
- Optimizers (SGD, Adam)
- Autograd execution engine via dynamic computation graphs (`Context`, `Tensor.backward`)
- Convolution and pooling primitives:
  - Naive CPU Conv2D forward/backward reference kernels
  - Pool2D forward/backward kernels with optional native C++ acceleration
    - AvgPool2D and MaxPool2D native CPU kernels via ctypes
    - Float32 / Float64 dtype-aware dispatch
    - Safe fallback to NumPy reference implementations when native kernels are unavailable
  - Autograd integration via `Conv2dFn` and pooling `Function`s

Infrastructure code is free to evolve independently as long as it satisfies domain interfaces.

---

### Native CPU Acceleration (Optional)

KeyDNN supports optional native CPU acceleration for selected operations
(currently AvgPool2D and MaxPool2D) via C++ kernels exposed through `ctypes`.

- Native kernels are used automatically for supported dtypes (`float32`, `float64`)
- If the native shared library is unavailable, KeyDNN emits a runtime warning
  and safely falls back to NumPy reference implementations
- Build scripts are provided for Windows (MinGW), Linux, and macOS
- Native acceleration is an optimization layer only; correctness is always
  validated against reference implementations via unit tests

This design allows incremental performance optimization without sacrificing
portability or debuggability.

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
