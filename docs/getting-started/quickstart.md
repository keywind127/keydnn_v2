# Quickstart

This page provides minimal, tested examples to get started with KeyDNN.

KeyDNN’s public APIs are available from the top-level `keydnn` package.

---

## Minimal Tensor + autograd

```python
from keydnn import Tensor, Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

---

## CUDA example (device-resident ops)

This example runs on CUDA if the backend is available.

```python
from keydnn import Tensor, Device, cuda_available

device = Device("cuda:0") if cuda_available() else Device("cpu")

x = Tensor.rand((1024, 1024), device=device, requires_grad=True)
y = (x @ x.T).mean()
y.backward()

print("device:", str(device))
print("y:", y.item())
```

---

## Reproducibility (seed + determinism)

KeyDNN provides two separate knobs for reproducibility:

- **Random seeding** controls random number generation (Python/NumPy).
- **Determinism policy** controls nondeterminism from CPU threading (BLAS/OpenMP scheduling).

### Recommended order

```python
from keydnn import seed, set_deterministic

seed(42)
set_deterministic(True)  # defaults to cpu_threads=1

# build / initialize model after reproducibility is configured
```

> Note: Thread-related environment variables may need to be set **before importing NumPy**
> (or any BLAS-backed library) to take full effect.

---

## CLI demo (MNIST & CIFAR smoke tests)

KeyDNN includes a small runnable training example wired through the package CLI:

```bash
# CPU (always available)
python -m keydnn test --train_mnist_example --device cpu --epochs 4 --limit-train 50000 --limit-test 1000

# CUDA (if CUDA backend + native libraries are available)
python -m keydnn test --train_mnist_example --device cuda:0 --epochs 4 --limit-train 50000 --limit-test 1000
```

CIFAR-10 CNN smoke test:

```bash
python -m keydnn test --train_cifar_example --device cuda:0 --epochs 4 --limit-train 50000 --limit-test 1000
```

---

## Training example (Model.fit + callbacks)

This example trains a small XOR network using `Sequential`, an optimizer, and callbacks.

```python
import numpy as np

from keydnn import (
    EarlyStopping,
    ModelCheckpoint,
    cuda_available,
    Tensor,
    Device,
    numpy_to_tensor,
    Sigmoid,
    Sequential,
    Linear,
)

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

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------
    device = Device("cuda:0") if cuda_available() else Device("cpu")

    # --------------------------------------------------------------
    # XOR dataset (repeat to form a small batchable dataset)
    # --------------------------------------------------------------
    x_base, y_base = _xor_data_numpy()
    repeats = 256
    x_np = np.repeat(x_base, repeats=repeats, axis=0)
    y_np = np.repeat(y_base, repeats=repeats, axis=0)

    x = numpy_to_tensor(x_np, device=device)
    y = numpy_to_tensor(y_np, device=device)

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    hidden_dim = 8

    model = Sequential(
        Linear(2, hidden_dim),
        Sigmoid(),
        Linear(hidden_dim, 1),
        Sigmoid(),
    )

    model.to_(device)
    model.build((1, 2), device=device)

    # --------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    x_eval = numpy_to_tensor(x_base, device=device)
    pred: Tensor = model(x_eval)
    pred_np = np.asarray(pred.to_numpy(), dtype=np.float32)

    print("device:", str(device))
    print("pred:", pred_np.reshape(-1).round(3).tolist())
    print("acc:", _accuracy_from_pred_np(y_base, pred_np))
```

---

## Notes on CUDA inference batching

When running on CUDA, KeyDNN currently expects evaluation to be performed in **mini-batches**
(e.g., `batch_size=128`) rather than passing an entire dataset tensor through the model at once.

If you hit a runtime error during evaluation with a large `N`, rerun inference in batches.
