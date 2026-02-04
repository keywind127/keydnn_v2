# Activations

Activation functions are provided as **layers** (callable components) in the public API.

They operate on `Tensor` inputs and are compatible with automatic differentiation.

---

::: keydnn.ReLU
    options:
      show_root_heading: true

::: keydnn.Sigmoid
    options:
      show_root_heading: true

::: keydnn.Tanh
    options:
      show_root_heading: true

::: keydnn.Softmax
    options:
      show_root_heading: true

---

## Notes

- Unless otherwise documented, activations preserve input shape.
- `Softmax` typically takes a `dim`/`axis` argument to specify which dimension to normalize over.
- For numerical stability, prefer using losses that accept logits directly (if your API supports that).
