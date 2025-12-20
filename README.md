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
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (SSE, MSE, Binary Cross Entropy, Categorical Cross Entropy)
- Optimizers (SGD, Adam)
- Autograd execution engine via dynamic computation graphs (`Context`, `Tensor.backward`)

Infrastructure code is free to evolve independently as long as it satisfies domain interfaces.

---

### Tests

The test suite is split into two categories:

- **Interface compatibility tests**
  - Ensure infrastructure classes conform to domain `Protocol`s
- **Behavioral unit tests**
  - Validate tensor, parameter, module, and layer behavior
  - Avoid private attribute access
- Integration tests validating end-to-end training on nonlinear tasks (e.g., XOR)

---

## Implemented Features

- Tensor abstraction with CPU (NumPy) backend and arithmetic operator overloading
- Device abstraction (`cpu`, `cuda:<index>`)
- Trainable `Parameter` class with gradient management
- Module system with parameter registration
- Fully connected `Linear` layer
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

## Roadmap (Planned)

- Additional layers and activation functions (beyond core ReLU/Sigmoid)
- CUDA-backed tensor operations
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
