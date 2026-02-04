# Layers

This section documents the **neural network layers** provided by KeyDNN’s public API.
All layers are part of the **presentation layer** and are safe to depend on.

Unless otherwise noted, layers:

- operate on `Tensor` inputs
- support automatic differentiation
- respect the device (`CPU` / `CUDA`) of their parameters
- follow PyTorch-style shape conventions where applicable

---

## Core Layers

::: keydnn.Dense
    options:
      show_root_heading: true

::: keydnn.Linear
    options:
      show_root_heading: true

---

## Convolution Layers

::: keydnn.Conv2D
    options:
      show_root_heading: true

::: keydnn.Conv2DTranspose
    options:
      show_root_heading: true

---

## Normalization Layers

> **Note**  
> KeyDNN provides both `BatchNorm1D` / `BatchNorm2D` and
> `BatchNorm1d` / `BatchNorm2d`.  
> These are equivalent and exist for naming compatibility.

::: keydnn.BatchNorm1d
    options:
      show_root_heading: true

::: keydnn.BatchNorm2d
    options:
      show_root_heading: true

::: keydnn.BatchNorm1D
    options:
      show_root_heading: true

::: keydnn.BatchNorm2D
    options:
      show_root_heading: true

::: keydnn.LayerNorm
    options:
      show_root_heading: true

---

## Regularization Layers

::: keydnn.Dropout
    options:
      show_root_heading: true

---

## Pooling Layers

::: keydnn.MaxPool2D
    options:
      show_root_heading: true

::: keydnn.AvgPool2D
    options:
      show_root_heading: true

::: keydnn.GlobalAvgPool2D
    options:
      show_root_heading: true

---

## Notes on Shapes and Devices

- Convolution and pooling layers expect **NCHW** layout by default.
- Parameters are created on the same device as the layer unless explicitly moved.
- Inputs must be contiguous for optimal CUDA performance.
- Shape mismatches are reported at runtime with descriptive errors.

For more details, see:

- **Guides → Tensors & Devices**
- **Guides → Training Loop**
