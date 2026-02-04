# Utilities

Utility functions provide common helpers for reproducibility, preprocessing,
and random number control.

All utilities documented here are part of KeyDNN’s **public presentation API**.

---

## Determinism

### set_deterministic

::: keydnn.set_deterministic
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Randomness

### seed

::: keydnn.seed
    options:
      show_root_heading: false
      show_root_toc_entry: false

### get_seed

::: keydnn.get_seed
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Preprocessing

### numpy_to_tensor

::: keydnn.numpy_to_tensor
    options:
      show_root_heading: false
      show_root_toc_entry: false

### one_hot

::: keydnn.one_hot
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Notes

- For fully reproducible experiments, call `set_deterministic()` and `seed()`
  before creating models, tensors, or datasets.
- `numpy_to_tensor()` is useful for bridging external NumPy-based pipelines
  with KeyDNN’s tensor system.
- `one_hot()` is commonly used with classification losses such as
  categorical cross-entropy.
