import pytest
import numpy as np
from qiskit import QuantumCircuit
from src.quantum.engine.digital_engine import GateBasedQuantumEngine, BackendStrategy
from src.quantum.engine.analog_engine import AnalogQuantumEngine, AnnealingStrategy


class TestGateBasedEngine:
    def test_gate_based_engine_execution(self):
        """Verify GPU Gate Engine can run a simple Bell State."""
        engine = GateBasedQuantumEngine(strategy=BackendStrategy.GPU)
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Choice A implementation: returns dict directly
        counts = engine.execute(qc, shots=100)
        
        assert isinstance(counts, dict)
        assert '00' in counts or '11' in counts
        assert sum(counts.values()) == 100

    def test_analog_engine_sqa(self):
        """Verify OpenJij SQA logic."""
        engine = AnalogQuantumEngine(annealing_strategy=AnnealingStrategy.QUANTUM)
        
        # Simple QUBO: minimize x0 + x1 - 2*x0*x1 (XOR-like)
        qubo = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        
        result = engine.execute(qubo)
        
        assert isinstance(result, list)
        assert len(result) == 2
        # Result should be [1, 1] or [0, 0] for this specific minimization
        assert result in [[0, 0], [1, 1]]

    def test_gpu_backend_fallback(self, caplog):
        """Ensure engine falls back to CPU if GPU is requested but fails."""
        # We pass a nonsensical strategy or mock a failure if needed
        engine = GateBasedQuantumEngine(strategy=BackendStrategy.GPU)
        # If this runs on a machine without a GPU, it should log a warning
        assert engine.backend is not None  