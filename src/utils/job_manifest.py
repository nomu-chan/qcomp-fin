from dataclasses import dataclass

@dataclass
class QuantumJobManifest:
    job_id: str           # The hash from your CacheProxy
    p_level: int          # QAOA depth
    qasm_string: str      # The circuit instructions
    shots: int            # Hardware sampling count
    qubo_payload: dict    # For energy calculation or Annealing mode
