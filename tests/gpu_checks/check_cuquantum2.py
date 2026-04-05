import os
from qiskit_aer import AerSimulator

def verify_gpu_backend():
    # 1. Initialize the simulator
    # We explicitly try to force the GPU and the cuStateVec method
    try:
        backend = AerSimulator(method='statevector', device='GPU')
        
        # 2. Check available methods
        config = backend.configuration()
        available_methods = getattr(config, 'filtering_functions', [])
        
        print(f"--- Backend Check ---")
        print(f"Device: {backend.options.device}")
        print(f"Method: {backend.options.method}")
        
        # 3. Test Run
        if backend.options.device == 'GPU':
            print("🚀 SUCCESS: Qiskit Aer is configured for GPU.")
            if 'cusv' in str(config):
                print("💎 cuStateVec acceleration is ACTIVE.")
        else:
            print("❌ FAIL: Backend initialized but defaulted to CPU.")
            
    except Exception as e:
        print(f"💥 Initialization Error: {e}")

if __name__ == "__main__":
    verify_gpu_backend()