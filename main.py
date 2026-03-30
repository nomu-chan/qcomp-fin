# --- ABSOLUTELY MUST BE LINE 1 ---
import ctypes
import os
try:
    ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
# ---------------------------------

from src.controller import run

if __name__ == "__main__":
    print("Running...")
    run()
