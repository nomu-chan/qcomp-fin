import openjij as oj
import numpy as np
from enum import Enum, auto

class AnnealingStrategy(Enum):
    QUANTUM = auto()
    CLASSICAL = auto()

class AnalogQuantumEngine:
    def __init__(self, annealing_strategy: AnnealingStrategy = AnnealingStrategy.QUANTUM, num_reads = 1000):
        self.annealing_strategy = annealing_strategy
        self.quantum_annealing = oj.SQASampler()
        self.classical_annealing = oj.SASampler()
        self._num_reads = num_reads
        
    def set_num_reads(self, num_reads: int):
        self._num_reads = num_reads
        
    def set_annealing_strategy(self, annealing_strategy: AnnealingStrategy):
        self.annealing_strategy = annealing_strategy

    def execute(self, qubo_dict):
        if self.annealing_strategy == AnnealingStrategy.QUANTUM:
            return self._execute_quantum_annealing(qubo_dict)
        return self._execute_classical_annealing(qubo_dict)

    def _execute_quantum_annealing(self, qubo_dict):
        """
        Implementation of Simulated Quantum Annealing (SQA).
        """
        # Initialize the SQA Sampler
        # 'gpu=True' triggers the CUDA kernels for the Trotter slices
        response = self.quantum_annealing.sample_qubo(qubo_dict, num_reads=self._num_reads)
        
        # Extract the best bitstring (lowest energy)
        best_sample = response.first.sample # e.g., {0: 1, 1: 0, 2: 1}
        return [best_sample[i] for i in sorted(best_sample.keys())]

    def _execute_classical_annealing(self, qubo_dict):
        """Standard Simulated Annealing (Thermal)"""
        response = self.classical_annealing.sample_qubo(qubo_dict, num_reads=self._num_reads)
        return [response.first.sample[i] for i in sorted(response.first.sample.keys())]