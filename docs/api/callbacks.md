# Callbacks

Callbacks allow users to inject custom logic into the training process,
such as early stopping, checkpointing, and logging.

Callbacks are executed at predefined hook points during training.

---

::: keydnn.Callback
    options:
      show_root_heading: true

---

::: keydnn.CallbackList
    options:
      show_root_heading: true

---

::: keydnn.EarlyStopping
    options:
      show_root_heading: true

---

::: keydnn.ModelCheckpoint
    options:
      show_root_heading: true

---

## Notes

- Custom callbacks should subclass `Callback` and override the relevant hook methods.
- Callbacks are typically passed as a list or managed via `CallbackList`.
- The exact hook order and lifecycle are documented in the base `Callback` class.
