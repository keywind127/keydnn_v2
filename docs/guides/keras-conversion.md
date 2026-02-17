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

- Python API: ```from keydnn import from_keras; model = from_keras("model.h5")```

* CLI: ```python -m keydnn convert --src model.h5 --dst model.json```

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

## ⚠️ Important: Convolution Layers Must Use `channels_first`

Keras defaults to:

```
data_format="channels_last"  # (N, H, W, C)
```

However, KeyDNN internally uses **NCHW layout**:

```
(N, C, H, W)
```

Therefore, all convolution and pooling layers **must explicitly specify**:

```python
data_format="channels_first"
```

### Example (Correct)

```python
Conv2D(
    32,
    kernel_size=3,
    padding="same",
    data_format="channels_first",
)
```

### Incorrect (Default Keras Behavior)

```python
Conv2D(32, 3)  # ❌ defaults to channels_last
```

This will raise:

```
KerasInteropError: Keras Conv2D with data_format='channels_last' is not supported in Phase 1.
```

---

### Why Is This Required?

KeyDNN:

- assumes NCHW layout at the tensor level,
- maps convolution weights directly without layout permutation,
- does not perform implicit tensor transpositions during import.

Allowing `channels_last` would require automatic graph rewriting and tensor reordering, which violates KeyDNN's design principles:

- **Explicit over implicit**
- **Deterministic conversion**
- **No hidden layout transformations**

---

### Recommended Pattern

For CNN models intended for conversion:

```python
layers.Input(shape=(1, 28, 28)),  # NCHW

Conv2D(..., data_format="channels_first"),
BatchNormalization(axis=1),
ReLU(),
MaxPool2D(..., data_format="channels_first"),
```

Always ensure:

- Input tensors are `(C, H, W)`
- `BatchNormalization(axis=1)` for channel dimension
- All pooling layers also use `channels_first`

---

### ⚠️ Important: `GlobalAveragePooling2D` Requires Explicit `Flatten`

When converting models that use:

```
GlobalAveragePooling2D
```

it is strongly recommended to **explicitly insert a `Flatten` layer immediately after**:

```python
GlobalAveragePooling2D(data_format="channels_first"),
Flatten(),
Dense(...)
```

### Why?

Although Keras may return a tensor shaped `(N, C)` after global pooling,
the KeyDNN importer operates under explicit shape semantics and expects
a clear dimensional transition before `Dense`.

Without an explicit `Flatten`:

- Shape inference may become ambiguous.
- Conversion may succeed but produce unintended tensor reshaping behavior.
- Future strict-mode checks may reject the model.

KeyDNN prioritizes **explicit shape transitions** over implicit assumptions.

### Recommended Pattern

```python
Conv2D(..., data_format="channels_first"),
BatchNormalization(axis=1),
ReLU(),
MaxPool2D(..., data_format="channels_first"),

GlobalAveragePooling2D(data_format="channels_first"),
Flatten(),  # <-- required for safe conversion
Dense(64),
```

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
