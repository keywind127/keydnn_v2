# Tensors & Devices

This guide explains KeyDNN’s core data structure (`Tensor`) and how device placement works.

KeyDNN supports both:

- **CPU tensors** (NumPy-backed)
- **CUDA tensors** (device-pointer–backed storage, explicit memcpy boundaries)

Public APIs are re-exported from `keydnn`.

```python
from keydnn import Tensor, Device, cuda_available
```

---

## Creating tensors

### Constructing an allocated tensor

```python
from keydnn import Tensor, Device

x = Tensor(shape=(2, 3), device=Device("cpu"), requires_grad=True)
```

### Factory functions

```python
from keydnn import Tensor, Device

x = Tensor.zeros((2, 3), device=Device("cpu"))
y = Tensor.ones((2, 3), device=Device("cpu"))
z = Tensor.rand((2, 3), device=Device("cpu"))
```

> Note: The exact set of factory methods is documented on the API page for `Tensor`.

---

## Device placement

### Choosing a device

```python
from keydnn import Device, cuda_available

device = Device("cuda:0") if cuda_available() else Device("cpu")
```

KeyDNN uses explicit device strings:

- `Device("cpu")`
- `Device("cuda:0")`, `Device("cuda:1")`, ...

### Moving data between devices

If your `Tensor` API supports `.to_(device)` (in-place) or `.to(device)` (out-of-place),
use the form documented in the `Tensor` API reference.

Typical patterns:

```python
# Out-of-place move (returns a new tensor)
x2 = x.to(Device("cuda:0"))

# In-place move (mutates)
x.to_(Device("cuda:0"))
```

If a method is not available, use `numpy_to_tensor()` to explicitly bridge from NumPy.

---

## NumPy interop

### Converting NumPy → Tensor

```python
import numpy as np
from keydnn import numpy_to_tensor, Device

a = np.random.randn(4, 5).astype(np.float32)
x = numpy_to_tensor(a, device=Device("cpu"))
```

### Converting Tensor → NumPy

```python
x_np = x.to_numpy()
```

> On CUDA, `to_numpy()` implies a device → host transfer.

---

## Autograd basics

Gradients are accumulated on leaf tensors/parameters that have `requires_grad=True`.

```python
from keydnn import Tensor, Device

x = Tensor.rand((2, 3), device=Device("cpu"), requires_grad=True)
y = (x * 2.0).sum()
y.backward()

print(x.grad.to_numpy())
```

Common gotchas:

- Gradients may accumulate across backward passes unless cleared.
- Some operations may create non-leaf tensors; check your API if you need `.retain_grad()`-style behavior.

---

## Shape conventions (recommended)

For convolution and pooling layers, KeyDNN follows the standard **NCHW** layout:

- `N`: batch
- `C`: channels
- `H`: height
- `W`: width

Example: a batch of images is shaped `(N, C, H, W)`.

---

## Contiguity and performance

For performance (especially on CUDA), inputs are expected to be **contiguous** in memory.

- Non-contiguous inputs may be internally made contiguous (extra copy), or may trigger slower fallback paths.
- When bridging from NumPy, prefer contiguous arrays:

  - `np.ascontiguousarray(...)` if you are unsure.

If you encounter surprising slowdowns, inspect:

- tensor shapes
- device placement
- whether you are repeatedly copying between CPU and CUDA

---

## CUDA evaluation batching (important)

When running on CUDA, KeyDNN currently expects inference/evaluation to be performed in **mini-batches**
(e.g., `batch_size=128`) rather than passing an entire dataset tensor through the model at once.

If evaluation fails for large `N` (e.g., `N=10000`) due to kernel limits in some ops, evaluate in batches:

```python
def predict_in_batches(model, x, batch_size=128):
    outs = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        outs.append(model(x[i : i + batch_size]))
    return Tensor.concat(outs, axis=0)  # if supported; otherwise stage to NumPy
```

> The exact batching utility depends on your exposed `Tensor.concat` / slicing behavior.
> Use this as a conceptual pattern.
