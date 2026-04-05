import torch
import numpy as np
import openjij as oj
from enum import Enum, auto
from simulated_bifurcation import minimize
from src.quantum.engine.logging_mod import get_logging  

class AnnealingStrategy(Enum):
    QUANTUM = auto()
    BIFURCATION = auto() # GPU Native
    CLASSICAL = auto()
    
logger = get_logging(__name__)    

class AnalogQuantumEngine:
    def __init__(self, annealing_strategy=AnnealingStrategy.BIFURCATION, use_gpu=True):
        self.annealing_strategy = annealing_strategy
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self._num_reads = 1024
        
        if self.use_gpu:
            logger.info(f"🚀 GPU BIFURCATION ENABLED (Device: {torch.cuda.get_device_name(0)})")
        else:
            logger.info("💻 Using CPU-based Optimization.")
        
    def set_num_reads(self, num_reads: int):
        self._num_reads = num_reads
        
    def set_annealing_strategy(self, annealing_strategy: AnnealingStrategy):
        self.annealing_strategy = annealing_strategy

    def execute(self, qubo_dict):
        """Main entry point that dispatches to the correct strategy."""
        if self.annealing_strategy == AnnealingStrategy.BIFURCATION:
            matrix = self._qubo_to_matrix(qubo_dict)
            return self._execute_simulated_bifurcation(matrix)
        
        # Dispatch to OpenJij strategies
        if self.annealing_strategy == AnnealingStrategy.QUANTUM:
            return self._execute_quantum_annealing(qubo_dict)
        return self._execute_classical_annealing(qubo_dict)

    def _execute_quantum_annealing(self, qubo_dict):
        """CPU-based Simulated Quantum Annealing via OpenJij."""
        sampler = oj.SQASampler()
        response = sampler.sample_qubo(qubo_dict, num_reads=self._num_reads)
        best_sample = response.first.sample
        return [best_sample[i] for i in sorted(best_sample.keys())]

    def _execute_classical_annealing(self, qubo_dict):
        """CPU-based Simulated Annealing via OpenJij."""
        sampler = oj.SASampler()
        response = sampler.sample_qubo(qubo_dict, num_reads=self._num_reads)
        best_sample = response.first.sample
        return [best_sample[i] for i in sorted(best_sample.keys())]
    
    def _execute_simulated_bifurcation(self, matrix):
        """GPU-accelerated Bifurcation via PyTorch."""
        Q = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        
        # SB API Fix: 
        # 1. 'domain="binary"' defines it as a QUBO {0, 1}
        # 2. 'model="ballistic"' replaces the ballistic=True flag
        best_vector, _ = minimize(
            Q, 
            domain="binary",
            agents=self._num_reads, 
            max_steps=self._num_reads, 
            mode="ballistic",
            verbose=False
        )
        
        return best_vector.cpu().numpy().tolist()
    
    def _qubo_to_matrix(self, qubo_dict):
        """Converts QUBO dict to symmetric matrix for SB."""
        nodes = set()
        for u, v in qubo_dict.keys():
            nodes.add(u); nodes.add(v)
        
        size = max(nodes) + 1
        matrix = np.zeros((size, size))
        for (u, v), val in qubo_dict.items():
            if u == v:
                matrix[u][u] += val
            else:
                # Divide by 2 because SB expects a symmetric matrix (Q + Q.T)
                matrix[u][v] += val / 2
                matrix[v][u] += val / 2
        return matrix