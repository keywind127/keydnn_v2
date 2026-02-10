# Keras Model Conversion

This guide documents how KeyDNN converts Keras models into KeyDNN model
artifacts using the public `keydnn.from_keras` API and the
`keydnn convert` CLI command.

The Keras importer is designed to provide **deterministic, explicit,
and debuggable** model conversion, rather than performing implicit or
lossy transformations.

---

## Overview

KeyDNN supports conversion of **Keras Sequential models** into a
KeyDNN module graph suitable for inference and further processing.

Conversion consists of two logical phases:

1. **Graph normalization**
   - Validate layer ordering and semantics.
   - Normalize Keras-specific constructs into explicit KeyDNN modules.
2. **Serialization**
   - Emit a KeyDNN model artifact (JSON checkpoint format).
   - Preserve weights and parameters exactly.

Conversion can be performed via:

- Python API:
    ```python
    from keydnn import from_keras
    model = from_keras("model.h5")
    ```

* CLI:

  ```bash
  python -m keydnn convert --src model.h5 --dst model.json
  ```

---

## Supported Keras Layers

The following Keras layers are currently supported by the importer.

### Convolution & Pooling

* `Conv2D`
* `Conv2DTranspose`
* `MaxPooling2D`
* `AveragePooling2D`
* `GlobalAveragePooling2D`

**Requirements**

* `data_format="channels_first"` is expected for convolution and pooling layers.
* Kernel size, stride, padding, and bias must be explicitly defined or inferable.

---

### Dense & Shape Layers

* `Dense`
* `Flatten`
* `Dropout` (inference-time identity behavior)

---

### Normalization

* `BatchNormalization`
* `LayerNormalization`

**Notes**

* BatchNorm running statistics and affine parameters are preserved.
* LayerNorm currently requires trailing-axis normalization.

---

### Activations

Supported activation layers:

* `ReLU`
* `LeakyReLU`
* `Sigmoid`
* `Tanh`
* `Softmax`

Activations may appear either as:

* standalone Keras layers (recommended), or
* fused into other layers (see below).

---

## Handling Fused Activations

In Keras, it is common to define activations directly inside layers:

```python
Dense(128, activation="relu")
Conv2D(32, 3, activation="relu")
```

However, KeyDNN uses **explicit activation modules** by design.

### Automatic Unfusing

During conversion, KeyDNN will automatically normalize fused activations by:

```text
Dense(..., activation="relu")
↓
Dense(..., activation="linear")
ReLU()
```

This ensures:

* consistent execution semantics,
* clearer module graphs,
* easier debugging and inspection.

This behavior is enabled by default.

---

### Strict Mode vs Non-Strict Mode

By default, conversion runs in **strict mode**.

* Unsupported layers or configurations will raise an error.
* Fused non-linear activations are only allowed if explicitly enabled.

You can control this behavior via:

#### Python API

```python
from keydnn import from_keras

model = from_keras(
    "model.h5",
    strict=True,
    allow_non_linear_activation=False,
)
```

#### CLI

```bash
python -m keydnn convert \
  --src model.h5 \
  --dst model.json \
  --allow-non-linear-activation
```

---

## Common Errors and Diagnostics

### Unsupported Layer

```text
KerasInteropError: Unsupported Keras layer: XYZ
```

This indicates the layer is not yet mapped to a KeyDNN equivalent.

---

### Fused Activation Rejected (Strict Mode)

```text
KerasInteropError: Non-linear activation fused into Dense is not allowed
```

**Resolution**

* Use explicit activation layers in Keras, or
* enable `--allow-non-linear-activation`.

---

### Missing or Ambiguous Weights

```text
ValueError: No weights found for module Conv2d
```

This usually indicates:

* a mismatch between serialization format and loader expectations, or
* a partially converted model artifact.

KeyDNN will fail fast rather than silently running with uninitialized weights.

---

## Design Rationale

KeyDNN intentionally avoids opaque or implicit conversions.

Key principles:

* **Explicit is better than implicit**
* **Fail early, fail clearly**
* **Preserve numerical fidelity**

This makes the conversion pipeline suitable for:

* research validation,
* backend development,
* and production debugging workflows.

---

## See Also

* `keydnn.from_keras` API reference
* CLI: `keydnn convert`
* Guides → Model Serialization
* Guides → Activation Modules
