# 1. The "Ghost of Qiskit Past" (Version Conflict)

## The Problem

uv or pip often tries to resolve dependencies by pulling in qiskit 2.3.1. This is a legacy metapackage from 2021. Modern Qiskit (Post-2024) starts at v1.0. Because 2.3.1 > 1.0.0 numerically, package managers "upgraded" you into a broken, ancient version.

## The Fix Strategy

Nuclear Purge: If you see qiskit 2.x in your list, delete the metadata manually:
rm -rf .venv/lib/python3.12/site-packages/qiskit*

Explicit Versioning: Always force the version in your install command:
uv pip install "qiskit>=1.1.0" "qiskit-aer-gpu>=0.15.1"

# 2. The "Arch Linux Blindness" (Library Linkage)

## The Problem

On Arch, NVIDIA libraries are in /usr/lib, but qiskit-aer-gpu wheels (built for Ubuntu/Debian) expect them in specific CUDA-named paths. Even if the files exist, the C++ backend of Aer cannot "see" the symbols during the hardware handshake.

## The Fix Strategy

The Global Symbol Force (RTLD_GLOBAL): This was your winning move. By loading the driver libraries into the Global Symbol Table before importing Qiskit, you essentially "handed" the GPU drivers to the Python process.

```Python
import ctypes
ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
```

The Path Injection: Arch requires an explicit LD_LIBRARY_PATH that points to both the system drivers and the .venv's internal cuquantum libs.

# 3. The "Auto-Detection vs. Forced Initialization"

## The Problem

AerSimulator().available_devices() might only return ('CPU',). This is often just a metadata failure in Aer’s discovery script—it doesn't mean the GPU won't work.

## The Fix Strategy

Don't Ask, Just Tell: Bypass the check. If you know the drivers are loaded via ctypes, force the backend:
sim = AerSimulator(method='matrix_product_state', device='GPU')

Status Check: If the code doesn't crash upon initialization, the link is successful.

# Library Stuff

## 1. The Manual Symlink (The "Version Cheat")

By creating symlinks inside your .venv, you performed a Library Redirection.

What you did: You told the Python environment, "Whenever a program asks for the old CUDA 12 files, give them the new CUDA 13 files from the system instead."

Why it worked: Aer's C++ controller is dynamically linked. It doesn't check the internal version of the library immediately; it just checks if the filename exists in the search path. Since CUDA is generally backward compatible, the version 13 library was able to fulfill the requests meant for version 12.

## 2. The Forced Initialization (The "Bypass")

This is a critical distinction in software engineering: Discovery vs. Execution.

Discovery (available_devices): This is a high-level Python check. It scans metadata and environment flags. On Arch, this often returns CPU only because the discovery script doesn't know where Arch hides NVIDIA libraries.

Execution (AerSimulator(device='GPU')): This ignores the high-level "list" and goes straight to the low-level C++ engine. It attempts to open a communication channel with the GPU.

The Result: Because you had already "cheated" the library names with symlinks and forced the symbols into memory with ctypes, the communication channel opened successfully, even though the "Discovery" script was still confused.

Strategy Summary for Future Reports
If you ever move this project to a new Arch machine, follow this 3-Step Recovery Protocol:

Step 1: The "Ghost" Purge
Ensure you are on the modern stack.

```Bash
rm -rf .venv/lib/python3.12/site-packages/qiskit*
uv pip install "qiskit>=1.1" qiskit-aer-gpu
Step 2: The Binary Shim
Link the system's current CUDA versions to the names the wheel expects.
```

```Bash
cd .venv/lib/python3.12/site-packages/cuquantum/lib
ln -sf /usr/lib/libcudart.so libcudart.so.12
ln -sf /usr/lib/libcusolver.so libcusolver.so.11
Step 3: The Runtime Force
In your Python code, use the "Global Scope" trick and ignore the device list.
```

```Python
import ctypes
ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)

## Don't check available_devices(), just boot it
sim = AerSimulator(method='matrix_product_state', device='GPU')
```
