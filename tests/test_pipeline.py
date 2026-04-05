import numpy as np
import jijmodeling as jm
from src.quantum.middleware.instantiator import InstantiatorCommand
from src.quantum.middleware.bridge import ModelBridgeCommand
from src.quantum.middleware.minimizer import QuantumEngineCommand
from src.utils.logging_mod import get_logging
import time
import pytest

logger = get_logging(__name__)
class TestPipeline:
    def test_pipeline(self, num_assets=20, p_max=4):
        logger.info(f"🚀 Starting Stress Test: {num_assets} Qubits up to p={p_max}")
        
        # 1. Setup Problem using your @problem.update pattern
        problem = jm.Problem("StressTest")
        
        @problem.update
        def _(p: jm.DecoratedProblem):
            # Define Variables and Placeholders through 'p'
            # We'll use a dense quadratic interaction for the stress test
            x = p.BinaryVar("x", shape=(num_assets,))
            
            # Objective: Sum of all x[i] * x[j] (Dense QUBO)
            # This creates a 'worst-case' complexity for the QAOA circuit
            interaction = jm.sum(x[i] * x[j] for i in range(num_assets) for j in range(num_assets))
            
            p += interaction

        instance_data = {} # num_assets is hardcoded in the decorator for this test
        
        # 2. Initialize Components
        instantiator = InstantiatorCommand()
        
        # FIX: Pylance identified 'gpu_venv_path' wasn't recognized. 
        # Ensure this matches your QuantumEngineCommand.__init__ argument name.
        # Based on your previous message, it might be 'gpu_venv_python'.        
        bridge = ModelBridgeCommand(layers_p=p_max, gpu_venv_python="./.venv-gpu/bin/python")
        
        # 3. Generate QUBO Product
        product = instantiator.get_qubo_prod(problem, instance_data)
        logger.info(f"QUBO generated. Hash: {product.hash}")

        # 4. Execute Warming Loop
        start_time = time.time()
        try:
            # This triggers the subprocess bridge to the .venv-gpu
            result = bridge.minimize_with_warming(product, is_using_cache=False)
            duration = time.time() - start_time
            
            print("\n" + "="*50)
            print("STRESS TEST SUCCESSFUL")
            print("="*50)
            print(f"Total Time: {duration:.2f}s")
            print(f"Final Energy (p={p_max}): {result.fun:.6f}")
            print(f"Parameters optimized: {len(result.x)}")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Pipeline crashed: {e}")
            raise

# if __name__ == "__main__":
#     # Start with 15 assets to be safe, then scale to 25
#     run_stress_test(num_assets=15, p_max=3)