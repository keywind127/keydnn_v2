# Backend

This section documents backend-related utilities exposed by KeyDNN’s public API.

Backend APIs allow users to query system capabilities without directly depending
on internal infrastructure or native bindings.

---

## cuda_available

::: keydnn.cuda_available
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Notes

- `cuda_available()` can be used to conditionally enable CUDA-specific logic.
- This function reflects whether CUDA support is available at runtime,
  not whether a particular tensor or model is currently on a CUDA device.
- Backend initialization is handled internally; users should not manually
  load or manage native libraries.
