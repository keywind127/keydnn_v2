# Datasets

Dataset utilities provide convenient access to commonly used datasets for
experimentation and benchmarking.

All dataset loaders documented here are part of KeyDNN’s **public presentation API**.

---

## load_mnist

::: keydnn.load_mnist
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## load_cifar10

::: keydnn.load_cifar10
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## load_cifar100

::: keydnn.load_cifar100
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Notes

- Dataset loaders may download data automatically if not already present.
- Returned data is typically provided as NumPy arrays or `Tensor` objects,
  depending on configuration.
- For reproducibility, consider setting seeds and determinism
  **before** loading datasets.
