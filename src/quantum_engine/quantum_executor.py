from qiskit_aer import AerSimulator
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qiskit import QuantumCircuit
from src.utils.logging_mod import get_logging
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
import os
from enum import Enum, auto
import ctypes

logger = get_logging(__name__)


class BackendStrategy(Enum):
    CPU = auto()
    GPU = auto()
    MPS_GPU = auto() # Matrix Product State - Recommended for 4GB VRAM



class LocalExecutor:
    def __init__(self, strategy: BackendStrategy = BackendStrategy.CPU, seed: int = 67):
        self.seed = seed
        self.strategy = strategy
        self.backend = self._initialize_backend(strategy)

    def _initialize_backend(self, strategy: BackendStrategy) -> AerSimulator:
        """Facade logic to resolve the best available backend with robust fallbacks."""
        
        if strategy in [BackendStrategy.GPU, BackendStrategy.MPS_GPU]:
            try:
                logger.info(f"Forcing GPU initialization: {strategy.name}")
                # Core config only to avoid 'Invalid Option' errors
                sim = AerSimulator(
                    method='matrix_product_state',
                    device='GPU',
                    seed_simulator=self.seed,
                    precision='single'
                )
                
                # Attempt to enable cuQuantum silently
                try:
                    sim.set_options(cuquantum_enable=True)
                except Exception:
                    pass 
                return sim
            except Exception as e:
                logger.warning(f"GPU failed: {e}. Falling back to CPU.")

        return AerSimulator(method='statevector', device='CPU', seed_simulator=self.seed)
        
    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        indexed_bindings: dict[str, float],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        """Bind QAOA angle parameters to the circuit before execution."""
        param_map = {
            param: indexed_bindings[param.name]
            for param in circuit.parameters
            if param.name in indexed_bindings
        }
        return circuit.assign_parameters(param_map)

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict:
        if circuit.parameters:
            raise ValueError(
                f"Circuit has unbound parameters: {circuit.parameters}. "
                "bind_parameters() should have been called first."
            )
        job = self.backend.run(circuit, shots=shots, seed=self.seed)
        return job.result().get_counts()
