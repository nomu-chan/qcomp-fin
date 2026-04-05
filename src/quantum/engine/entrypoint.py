# src/quantum/engine/entrypoint.py
import sys
import json
import numpy as np
from qiskit import QuantumCircuit
from src.quantum.engine.digital_engine import GateBasedQuantumEngine
from src.quantum.engine.analog_engine import AnalogQuantumEngine, AnnealingStrategy
from qiskit import qasm3

def main():
    try:
        raw_input = sys.stdin.read()
        manifest = json.loads(raw_input)
    except Exception as e:
        print(json.dumps({"error": f"Invalid Manifest: {str(e)}"}))
        sys.exit(1)

    job_type = manifest.get("type")
    
    # --- BRANCH 1: GATE BASED (QAOA) ---
    if job_type == "GATE_BASED":
        engine = GateBasedQuantumEngine() # Initializes GPU Aer
        try:
            qc = qasm3.loads(manifest["payload"])
        except Exception:
            # Fallback for older Qiskit 1.0 environments if qasm3.loads fails
            from qiskit.circuit import QuantumCircuit
            qc = QuantumCircuit.from_qasm_str(manifest["payload"])
        
        bindings = manifest.get("bindings", {})
        # Bind parameters if provided
        if bindings:
            full_binding_dict = {}
            if "betas" in bindings:
                for i, val in enumerate(bindings["betas"]):
                    full_binding_dict[f"_betas_{i}_"] = val
            if "gammas" in bindings:
                for i, val in enumerate(bindings["gammas"]):
                    full_binding_dict[f"_gammas_{i}_"] = val
            
            # Bind the parameters to the circuit
            qc = qc.assign_parameters(full_binding_dict)
        
        counts = engine.execute(qc, shots=manifest["shots"]) 
        print(json.dumps(counts))

    # --- BRANCH 2: ANALOG (OpenJij SQA) ---
    elif job_type == "ANALOG":
        # OpenJij QUBO keys must be integers, but JSON turns them into strings
        # We must cast them back: {"(0, 1)": 0.5} -> {(0, 1): 0.5}
        raw_qubo = manifest["payload"]
        qubo = {tuple(map(int, k.strip("()").split(","))): v for k, v in raw_qubo.items()}
        
        engine = AnalogQuantumEngine(num_reads=manifest["num_reads"])
        result = engine.execute(qubo)
        print(json.dumps(result))

    sys.exit(0)

if __name__ == "__main__":
    main()