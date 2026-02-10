# Keras Interop

This section documents KeyDNN’s Keras interoperability utilities exposed by the public API.

Keras interop allows you to:
- Convert supported `tf.keras.Sequential` models into KeyDNN models for inference and experimentation.
- Validate conversion behavior using the CLI (`python -m keydnn convert`).

---

## from_keras

::: keydnn.from_keras
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Usage

### Python API

```python
import numpy as np
from keydnn import from_keras

kd_model = from_keras(
    "./model.h5",
    device="cpu",
    dtype=np.float32,
    strict=True,
    allow_non_linear_activation=False,
)
```

### CLI

```bash
python -m keydnn convert --src "./model.h5" --dst "./model.json"
```

---

## Supported model types

Phase 1 supports **Sequential-only** conversion:

* ✅ `tf.keras.Sequential`
* ❌ Functional API (`tf.keras.Model` graphs)
* ❌ Subclassed models

If a model is not `tf.keras.Sequential`, conversion raises `KerasInteropError` in strict mode.

---

## Supported layers

Phase 1 targets a small, explicit, test-backed subset. Supported layers currently include:

### Convolution / Pooling

* `tf.keras.layers.Conv2D` (**constraints apply**, see below)
* `tf.keras.layers.MaxPooling2D`
* `tf.keras.layers.GlobalAveragePooling2D`

### Shape / routing

* `tf.keras.layers.Flatten`

### Fully-connected

* `tf.keras.layers.Dense`

### Activations

* `tf.keras.layers.ReLU`
* `tf.keras.layers.Softmax`

> Note: We strongly recommend using **dedicated activation layers** (e.g., `ReLU()`, `Softmax()`)
> rather than fused activations inside `Conv2D(..., activation=...)` or `Dense(..., activation=...)`,
> unless you explicitly enable fused activations via `allow_non_linear_activation=True`.

---

## Layer constraints and expectations

### Data format (Conv/Pool stacks)

Current conversion assumes **channels_first** for convolution/pooling stacks in many phase-1 pipelines.

* Prefer: `data_format="channels_first"` for `Conv2D`, `MaxPooling2D`, `GlobalAveragePooling2D`
* If you trained with `channels_last` for CPU convenience, you can build an equivalent
  channels_first model and copy weights before saving (see the end-to-end script).

### Conv2D

Supported configuration (typical):

* `padding="same"` or `"valid"` (depending on converter support)
* `strides` supported
* `dilation_rate` may not be supported (converter-dependent)
* grouped/depthwise conv not supported unless explicitly implemented

### Dense input shape

KeyDNN `Dense` expects 2D input `(batch, in_features)`. If your model uses global pooling that yields
a `(N, C, 1, 1)` tensor, ensure you include `Flatten()` before `Dense`.

---

## Arguments

| Name                          | Type                    |      Default | Description                                                                                                                                                                                                  |
| ----------------------------- | ----------------------- | -----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model_or_path`               | `tf.keras.Model \| str` |            — | Either a loaded Keras `Sequential` model or a filesystem path to a saved model (`.keras` or `.h5`).                                                                                                          |
| `device`                      | `Any`                   |      `"cpu"` | Target KeyDNN device (e.g. `"cpu"`, `"cuda:0"`, or a KeyDNN `Device` object).                                                                                                                                |
| `dtype`                       | `Any`                   | `np.float32` | Target floating dtype for created parameters (typically `np.float32`).                                                                                                                                       |
| `strict`                      | `bool`                  |       `True` | If `True`, unsupported layers/configurations raise `KerasInteropError`. If `False`, behavior depends on the converter registry (unsupported layers may be skipped or mapped to placeholders if implemented). |
| `allow_non_linear_activation` | `bool`                  |      `False` | If `True`, converters may accept fused non-linear activations inside certain Keras layers. Recommended to keep `False` and use dedicated activation layers for clarity and stability.                        |

---

## Returns

| Type                         | Description                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Sequential \| list[Module]` | A KeyDNN model (typically a `Sequential` container). If a `Sequential` container is unavailable, returns a list of KeyDNN modules in execution order. |

---

## Errors and diagnostics

### Unsupported layers / models

If the input model contains unsupported layer types, unsupported configurations, or is not
`tf.keras.Sequential`, conversion will raise:

* `ImportError` if TensorFlow is not installed
* `KerasInteropError` for unsupported model types/layers/configs (when `strict=True`)

When authoring models for conversion, prefer a simple sequential stack and keep activations explicit.

---

## Recommended conversion pattern (Phase 1)

A reliable “interop-friendly” pattern:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 28, 28)),
    tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="linear", data_format="channels_first"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first"),
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="linear", data_format="channels_first"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="linear"),
    tf.keras.layers.Softmax(axis=-1),
])
```

---

## End-to-end verification script

KeyDNN includes an end-to-end script that validates the CLI conversion path:

* Train a small Keras MNIST CNN.
* Save as `.h5`.
* Run `python -m keydnn convert --src ... --dst ...`.
* Load the KeyDNN artifact and verify inference matches Keras.

This script is intended as a **CLI feature test** for `keydnn convert`.

It verifies that:
- the CLI invocation works end-to-end,
- a Keras model can be converted into a valid KeyDNN artifact,
- all trained weights are correctly serialized and reloaded, and
- KeyDNN inference numerically matches Keras within tolerance.

It is **not** a training benchmark, performance test, or comprehensive
layer-coverage suite. Its purpose is to prevent silent conversion failures
(e.g., missing weights, schema mismatches, or default re-initialization)
when using the CLI.
