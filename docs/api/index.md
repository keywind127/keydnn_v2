# API Overview

This section documents the **public API surface of KeyDNN**.

All APIs listed here are part of KeyDNN’s **presentation layer** and are considered
stable for user-facing development. Internal implementation details are intentionally
hidden and may change without notice.

---

## Public API Scope

KeyDNN follows a layered architecture (PIAD: Presentation → Infrastructure → Application → Domain).

- **Presentation layer**  
  Public, documented, and stable. Safe for direct use.

- **Domain / Infrastructure layers**  
  Internal. Subject to change. Not documented here.

If an object or function is documented in this section, it is safe to depend on.
If it is not documented, it should be considered internal.

---

## API Categories

The public API is organized into the following categories:

- **Tensors**  
  Core data structures for numerical computation and automatic differentiation.

- **Layers**  
  Neural network building blocks such as dense layers, convolutions,
  normalization, pooling, and regularization layers.

- **Activations**  
  Common activation functions implemented as callable layers.

- **Losses**  
  Standard loss functions used for training neural networks.

- **Optimizers**  
  Algorithms for updating model parameters based on gradients.

- **Models**  
  High-level abstractions for composing layers and managing training workflows.

- **Callbacks**  
  Hook-based utilities for extending training behavior
  (e.g., early stopping, checkpointing).

- **Datasets**  
  Convenience loaders for commonly used benchmark datasets.

- **Backend**  
  Utilities for querying system capabilities such as CUDA availability.

- **Utilities**  
  Helper functions for determinism, randomness control, and preprocessing.

Each category has its own dedicated reference page.

---

## Import Conventions

All public APIs are re-exported at the top level of the `keydnn` package.

Recommended import style:

```python
from keydnn import Tensor, Sequential, Adam
```
