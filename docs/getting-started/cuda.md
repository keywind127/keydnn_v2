# CUDA (Windows)

KeyDNN’s Windows CUDA backend loads a native DLL and relies on the CUDA runtime
(and optionally cuDNN) being discoverable by the current process.

If CUDA native libraries are unavailable, CUDA tests are skipped and CUDA execution
paths will raise or fall back where explicitly documented.

---

## Requirements

- A compatible NVIDIA GPU
- A working CUDA runtime installation
- (Optional) cuDNN, depending on which operations you use and how your native backend is configured

You can check availability from Python:

```python
from keydnn import cuda_available
print(cuda_available())
```

---

## Environment variables

KeyDNN uses environment variables to help Windows locate CUDA runtime dependencies:

- `CUDA_PATH` (recommended): points to your CUDA install root, e.g.
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2`

- `CUDNN_PATH` (optional): points to your cuDNN root that contains `bin/`, `lib/`, `include/`, e.g.
  `C:\cudnn`

If you copied cuDNN DLLs into the CUDA install (common manual setup), you typically do **not**
need `CUDNN_PATH` as long as `cudnn*.dll` exists in `<CUDA_PATH>\bin`.

---

## PowerShell examples

```powershell
# For the current terminal session only:
$env:CUDA_PATH  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
$env:CUDNN_PATH = "C:\cudnn"   # optional (only needed if cuDNN is not in CUDA\bin)
```

> Note: If you change environment variables, restart the Python process (and sometimes the terminal)
> before retrying.

---

## Common issues

### `cuda_available()` is False

- Verify your NVIDIA driver + CUDA toolkit installation.
- Confirm `CUDA_PATH` points to your CUDA install root.
- Ensure `<CUDA_PATH>\bin` contains CUDA runtime DLLs (e.g., `cudart64_*.dll`).

### cuDNN not found

- If using cuDNN-accelerated ops, ensure `cudnn*.dll` is discoverable:

  - either present in `<CUDA_PATH>\bin`, or
  - reachable via `CUDNN_PATH` → `<CUDNN_PATH>\bin`.

### Import-time CUDA failures

If importing `keydnn` triggers CUDA loading failures on machines without CUDA, prefer:

- lazy CUDA initialization in backend code paths (only load CUDA DLLs when CUDA is requested), and
- using `cuda_available()` before selecting CUDA devices.

---

## Recommended usage pattern

```python
from keydnn import Device, cuda_available

device = Device("cuda:0") if cuda_available() else Device("cpu")
```

Use this pattern to keep scripts runnable on both CPU-only and CUDA machines.

---

## Notes on performance and correctness

- CUDA kernels are asynchronous by default; timing measurements may require explicit synchronization.
- Some CUDA features may use correctness-first fallbacks in specific paths until full kernel coverage is implemented.
- For evaluation/inference, prefer mini-batching to avoid large-tensor limits in certain kernels.
