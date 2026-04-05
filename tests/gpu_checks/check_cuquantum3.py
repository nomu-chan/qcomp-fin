import os
import subprocess
import sys
import ctypes
import platform
import logging
logging.getLogger('qiskit_aer').setLevel(logging.DEBUG)
import ctypes
try:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    # Initializing the CUDA Driver API
    cuda.cuInit(0)
    # Getting the device count
    cuda.cuDeviceGetCount(ctypes.byref(count))
    print(f"Driver API Device Count: {count.value}")
except Exception as e:
    print(f"Driver API Probe Failed: {e}")

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except Exception as e:
        return f"Error: {e}"

print("=== QISKIT AER GPU FORENSICS ===")
print(f"OS: {platform.platform()}")
print(f"Python: {sys.version}")
print(f"Venv: {os.environ.get('VIRTUAL_ENV', 'None')}")

print("\n--- 1. Linker Path Status ---")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")

print("\n--- 2. Aer Binary Inspection ---")
# Find the actual shared object for Aer
aer_so = run_cmd("find .venv -name 'controller_wrapper*.so' | head -n 1")
if "Error" not in aer_so and aer_so:
    print(f"Found Aer Binary: {aer_so}")
    # Check for specific missing CUDA symbols
    ldd_out = run_cmd(f"ldd {aer_so}")
    print("Missing symbols (ldd):")
    for line in ldd_out.split('\n'):
        if "not found" in line:
            print(f"  ❌ {line.strip()}")
    if "not found" not in ldd_out:
        print("  ✅ All shared libraries resolved by ldd.")
else:
    print("❌ Could not find Aer controller binary.")

print("\n--- 3. NVIDIA Driver/Toolbox Check ---")
print(f"NVIDIA-SMI: {run_cmd('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")
print(f"NVCC Version: {run_cmd('nvcc --version | grep release')}")

print("\n--- 4. Venv cuQuantum Inventory ---")
cuq_path = run_cmd("find .venv -name 'libcustatevec.so*' | xargs -I {} dirname {} | uniq")
print(f"cuQuantum Lib Dir: {cuq_path}")
if cuq_path:
    print("Files in cuq_path:")
    print(run_cmd(f"ls -F {cuq_path}"))

print("\n--- 5. Runtime Symbol Probe ---")
try:
    cuda = ctypes.CDLL("libcuda.so.1")
    print("✅ libcuda.so.1: LOADABLE")
except Exception as e:
    print(f"❌ libcuda.so.1: FAILED ({e})")

try:
    # This is the specific one Aer needs for GPU simulation
    custate = ctypes.CDLL("libcustatevec.so.12")
    print("✅ libcustatevec.so.12: LOADABLE")
except Exception as e:
    print(f"❌ libcustatevec.so.12: FAILED ({e})")

print("\n--- 6. Qiskit Aer Internal View ---")
try:
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    print(f"Available Devices: {sim.available_devices()}")
    print(f"Available Methods: {sim.available_methods()}")
    
    # Check the actual hardware config detected by Aer
    config = sim.configuration().to_dict()
    print(f"Aer Backend Name: {config.get('backend_name')}")
    print(f"Custom Options: {config.get('custom_backend_options', 'None')}")
except Exception as e:
    print(f"❌ Aer Probe Failed: {e}")

print("\n=== End of Report ===")