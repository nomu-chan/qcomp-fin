
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

import os, sys
from enum import Enum, auto
import ctypes
from src.quantum.engine.logging_mod import get_logging
logger = get_logging(__name__)


class BackendStrategy(Enum):
    CPU = auto()
    GPU = auto()

class GateBasedQuantumEngine:
    def __init__(self, strategy: BackendStrategy = BackendStrategy.GPU, seed: int = 67):
        self.seed = seed
        self.strategy = strategy
        self.backend = self._initialize_backend(strategy)
        self._residual_parameters_format = "Circuit has unbound parameters: {circuit.parameters}.\nbind_parameters() should have been called first."

    def _initialize_backend(self, strategy: BackendStrategy) -> AerSimulator:
        """Initializes the backend to be a GPU or CPU

        Args:
            strategy (BackendStrategy): BackendStrategy Enum to choose

        Returns:
            AerSimulator: Backend to use
        """
        try:
            if strategy == BackendStrategy.GPU:
                logger.info(f"Attempting GPU initialization (matrix_product_state)...")
                sim = AerSimulator(
                    method='statevector',#statevector or matrix_product_state
                    device='GPU',
                    seed_simulator=self.seed,
                    batched_shots_gpu=True,
                    cuStateVec_enable=True
                )
                return sim

        except Exception as e:
            logger.warning(f"GPU Strategy {strategy.name} failed: {e}. Falling back to CPU.")

        # Final Fallback
        return AerSimulator(method='statevector', device='CPU', seed_simulator=self.seed)
    
    def execute(self, circuit: QuantumCircuit, shots: int) -> list:
        logger.debug(f"DEBUG: Executing circuit with {circuit.num_qubits} qubits...\n")
        if circuit.parameters: 
            raise ValueError(f"Circuit still has unbound parameters: {circuit.parameters}")

        result = self.backend.run(circuit, shots=shots, seed=self.seed).result()
    
        return result.get_counts() # type_ignore
