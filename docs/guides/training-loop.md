# Training Loop

KeyDNN provides a Keras-style training interface via `Sequential.fit(...)` and integrates with callbacks.
This guide explains the typical workflow and recommended patterns.

---

## Typical workflow

1. Select device (`cpu` or `cuda:0`)
2. Prepare data (NumPy → Tensor)
3. Build model (layers + activations)
4. Move model to device
5. Call `fit(...)`
6. Evaluate / predict (prefer batching on CUDA)

---

## Minimal example (XOR)

```python
import numpy as np

from keydnn import (
    Device,
    cuda_available,
    numpy_to_tensor,
    Sequential,
    Linear,
    Sigmoid,
    EarlyStopping,
    ModelCheckpoint,
)

def xor_numpy():
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)
    return x, y

device = Device("cuda:0") if cuda_available() else Device("cpu")

x_np, y_np = xor_numpy()
x = numpy_to_tensor(np.repeat(x_np, 256, axis=0), device=device)
y = numpy_to_tensor(np.repeat(y_np, 256, axis=0), device=device)

model = Sequential(
    Linear(2, 8),
    Sigmoid(),
    Linear(8, 1),
    Sigmoid(),
)

model.to_(device)
model.build((1, 2), device=device)

callbacks = [
    EarlyStopping(monitor="acc", mode="max", patience=5, min_delta=1e-4, restore_best_weights=True),
    ModelCheckpoint(filepath="xor_epoch{epoch:03d}_loss{loss:.6f}.json", monitor="acc", mode="max",
                    save_best_only=True, verbose=1),
]

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
```

---

## Choosing losses and target formats

KeyDNN exposes common losses as functions (and may also accept string aliases in `fit`).

Be consistent about target format:

- **MSE/SSE** often pair naturally with one-hot labels for classification-style experiments.
- **BCE** expects targets in `[0,1]` and predictions in `[0,1]` (or logits, depending on implementation).
- **CCE** typically expects class indices or one-hot targets (document in your `cce_loss` docstring).

When in doubt, check the **API Reference → Losses** page.

---

## Optimizers and gradient clearing

Optimizers update model parameters based on accumulated gradients.

Typical semantics (confirm against your API):

- `loss.backward()` accumulates gradients
- `optimizer.step()` updates weights
- `optimizer.zero_grad()` clears gradients

In `fit(...)`, these steps are orchestrated internally.

---

## Callbacks lifecycle

Callbacks allow you to extend training behavior without modifying the training loop.

Common use cases:

- early stopping (`EarlyStopping`)
- saving checkpoints (`ModelCheckpoint`)
- custom logging

If you implement custom callbacks, inherit from `Callback` and override relevant hooks.

---

## Evaluation tips (CUDA)

For CUDA evaluation:

- prefer **mini-batches**
- avoid sending very large `N` tensors through ops that may have kernel launch limits

If you hit errors evaluating `N=10000` at once, evaluate in batches and aggregate.
