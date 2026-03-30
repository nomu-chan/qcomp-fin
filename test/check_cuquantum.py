import os
import subprocess
import ctypes
import sys

# Force the Nvidia Management Library into the Global Symbol Table
try:
    # On Arch, we need to load these with RTLD_GLOBAL so the Aer C++ 
    # extensions can "see" them without needing their own explicit link.
    ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    print("✅ Successfully forced NVML/CUDA into Global Scope")
except Exception as e:
    print(f"⚠️ Global Link failed: {e}")

# Before importing qiskit_aer, we must manually load the entry point for CUDA
try:
    # This forces the symbols into the global scope for the Aer binary to find
    ctypes.CDLL("libnvidia-ml.so.1") 
    print("✅ Successfully pre-loaded NVML")
except Exception as e:
    print(f"⚠️ Pre-load failed: {e}")


# Arch typically puts these in /usr/lib
system_libs = "/usr/lib"
if system_libs not in os.environ.get('LD_LIBRARY_PATH', ''):
    print("LD_LIBRARY_PATH not found, adding..")
    os.environ['LD_LIBRARY_PATH'] = f"{system_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"Is true? {system_libs not in os.environ.get('LD_LIBRARY_PATH', '')}")
from qiskit_aer import AerSimulator

sim = AerSimulator()
print(f"Devices found: {sim.available_devices()}")


try:
    # We bypass the 'available_devices' check and try to boot the GPU directly
    sim_gpu = AerSimulator(method='matrix_product_state', device='GPU')
    print("🚀 SUCCESS: GPU Simulator initialized despite auto-detection failure!")
except Exception as e:
    print(f"❌ Forced GPU Init Failed: {e}")

def check_gpu():
    print("🔍 Checking for NVIDIA libraries in venv...")
    # Find where the .so files actually are
    try:
        venv_base = os.environ.get('VIRTUAL_ENV', '.venv')
        # Search for the specific library Aer needs
        search_cmd = f"find {venv_base} -name 'libcustatevec.so*'"
        result = subprocess.check_output(search_cmd, shell=True).decode().strip().split('\n')
        
        if result and result[0]:
            lib_dir = os.path.dirname(result[0])
            print(f"✅ Found libcustatevec at: {lib_dir}")
            
            # Inject into environment for the current process
            os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            
            # Now check Aer
            sim = AerSimulator()
            if "GPU" in sim.available_devices(): #type: ignore
                print("🚀 SUCCESS: Aer recognizes the GPU via local venv libraries.")
            else:
                print("⚠️  Aer still doesn't see GPU. May need 'nvidia-libs' via pacman.")
        else:
            print("❌ Could not find libcustatevec.so in .venv.")
            
    except Exception as e:
        print(f"❌ Error during check: {e}")

if __name__ == "__main__":
    check_gpu()