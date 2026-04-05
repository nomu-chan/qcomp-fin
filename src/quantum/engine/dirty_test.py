import time
import numpy as np
import torch
from src.quantum.engine.analog_engine import AnalogQuantumEngine, AnnealingStrategy

def generate_rugged_portfolio_qubo(size=50, ruggedness_factor=1.0):
    """
    Generates a QUBO with high-frequency local minima.
    ruggedness_factor: Multiplier for the variance of the off-diagonal terms.
    """
    qubo = {}
    # Linear terms: Random returns
    for i in range(size):
        qubo[(i, i)] = np.random.uniform(-1, 1)
        
    # Quadratic terms: Densely connected interactions with high variance
    # High variance = steeper 'cliffs' and deeper 'wells' in the landscape
    for i in range(size):
        for j in range(i + 1, size):
            # As ruggedness_factor increases, the 'energy barriers' become harder to jump
            qubo[(i, j)] = np.random.uniform(-ruggedness_factor, ruggedness_factor)
            
    return qubo

def run_rugged_benchmark():
    size = 100
    # Increase agents to show GPU parallelism advantage
    num_reads = 5000 
    
    # Levels of Ruggedness to test
    rugged_levels = [1.0, 5.0, 20.0, 500, 1024] 
    
    engine = AnalogQuantumEngine()
    engine.set_num_reads(num_reads)

    print(f"{'Level':<10} | {'Strategy':<25} | {'Time (s)':<10} | {'Energy':<12}")
    print("-" * 65)

    for level in rugged_levels:
        qubo = generate_rugged_portfolio_qubo(size=size, ruggedness_factor=level)
        
        for name, strategy in [("CPU-SA", AnnealingStrategy.CLASSICAL), 
                               ("GPU-SB", AnnealingStrategy.BIFURCATION)]:
            print(f"name: {name}, {strategy}")
            engine.set_annealing_strategy(strategy)
            
            start = time.perf_counter()
            bitstring = engine.execute(qubo)
            duration = time.perf_counter() - start
            
            # Simple Energy check
            matrix = engine._qubo_to_matrix(qubo)
            x = np.array(bitstring)
            energy = x.T @ matrix @ x
            
            print(f"Lvl {level:<5} | {name:<25} | {duration:<10.4f} | {energy:<12.4f}")

if __name__ == "__main__":
    # Ensure CUDA is available for the Bifurcation engine
    if torch.cuda.is_available():
        run_rugged_benchmark()
    else:
        print("Skipping test: CUDA not available for GPU comparison.")