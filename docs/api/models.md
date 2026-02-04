# Models

Models provide high-level abstractions for composing layers, running training loops,
and tracking training history.

All models documented here are part of KeyDNN’s **public presentation API**.

---

::: keydnn.Sequential
    options:
      show_root_heading: true

---

::: keydnn.History
    options:
      show_root_heading: true

---

## Notes

- `Sequential` is intended for linear stacks of layers.
- Parameters are registered automatically when layers are added.
- Training utilities such as callbacks and optimizers integrate with the model API.
- For custom architectures, users may subclass lower-level building blocks
  (documented in the Guides section).
