# KeyDNN

**KeyDNN** is a lightweight deep learning framework built from scratch in Python, designed to explore and demonstrate the core abstractions behind modern neural network libraries.

The project emphasizes **clean architecture**, **explicit interfaces**, and **separation of concerns**, making it suitable for learning, experimentation, and future extension.

---

## Project Goals

- Implement core deep learning primitives (Tensor, Parameter, layers) from first principles
- Clearly separate **domain concepts** from **infrastructure implementations**
- Explore design trade-offs around **autograd**, **device abstraction (CPU/CUDA)**, and **framework extensibility**
- Serve as a pedagogical and experimental alternative to large frameworks

---

## Current Architecture

KeyDNN follows a layered design:

- **Domain**
  - Defines framework contracts using `Protocol` interfaces
  - Examples: `ITensor`, `IParameter`
- **Infrastructure**
  - Concrete implementations backed by NumPy (CPU) and CUDA placeholders
  - Examples: `Tensor`, `Parameter`
- **Tests**
  - Interface compatibility tests
  - Behavioral unit tests for infrastructure classes

This structure allows implementations to evolve independently of the core domain definitions.

---

## Implemented Features

- Tensor abstraction with CPU (NumPy) backend
- Device abstraction (`cpu`, `cuda:<index>`)
- Trainable `Parameter` class with gradient management
- Domain-level interfaces using structural typing (`Protocol`)
- Comprehensive unit tests for contracts and behavior

---

## Roadmap (Planned)

- Automatic differentiation (autograd)
- Neural network modules (Linear, activation layers)
- Optimizers (SGD, Adam)
- CUDA-backed tensor operations
- Model composition utilities

---

## Disclaimer

KeyDNN is a **work in progress** and not intended for production use.  
APIs may change as the framework evolves.

---

## License

MIT License
