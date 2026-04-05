from fastapi import FastAPI
import uvicorn
from qiskit import qasm3
from src.quantum.engine.digital_engine import GateBasedQuantumEngine, BackendStrategy
from src.quantum.engine.analog_engine import AnalogQuantumEngine, AnnealingStrategy
from src.quantum.engine.logging_mod import get_logging

# Setup logging to stderr so it doesn't pollute stdout if you ever run manually
logger = get_logging("qcompfin-gpu-server")

app = FastAPI()

# --- INITIALIZE YOUR ENGINE ONCE ---
# This triggers your _initialize_backend(GPU) logic immediately on startup
digital_engine = GateBasedQuantumEngine(strategy=BackendStrategy.GPU)
analog_engine = AnalogQuantumEngine(annealing_strategy=AnnealingStrategy.BIFURCATION)

def map_to_qasm_names(binding: dict) -> dict:
    mapped = {}
    for key in ["betas", "gammas"]:
        if key in binding:
            for i, val in enumerate(binding[key]):
                mapped[f"_{key}_{i}_"] = val  # <--- It expects "_betas_0_", "_gammas_0_"
    logger.info(f"map to qasp names:{mapped}")
    return mapped

@app.get("/health")
async def health():
    return {"status" : "success", "results": "Hello"}

@app.post("/execute")
async def execute(manifest: dict):
    run_type = manifest.get("type", "GATE_BASED")
    try:
        if run_type == "GATE_BASED":
            # 1. Load the QASM template
            qc = qasm3.loads(manifest["payload"])
            shots = manifest.get("shots", 512)
            
            # 2. Normalize bindings to a list
            raw_bindings = manifest.get("bindings", [])
            
            if isinstance(raw_bindings, list):
                logger.info(f"📤 Dispatching Batch Request: {len(raw_bindings)} circuits")
            elif isinstance(raw_bindings, dict):
                logger.info(f"📤 Dispatching Single Request: p={len(raw_bindings.get('betas', []))}")
            
            if isinstance(raw_bindings, dict):
                raw_bindings = [raw_bindings]

            # 3. Batch Bind (Vectorized Parameter Assignment)
            # This is where the magic happens: creating N circuits for the GPU
            bound_circuits = []
            for b in raw_bindings:
                mapped_params = map_to_qasm_names(b)
                bound_circuits.append(qc.assign_parameters(mapped_params))

            logger.info(f"🚀 GPU Batch Execution: {len(bound_circuits)} circuits")
            
            # 4. Run the entire batch in ONE GPU call
            # Aer handles the internal parallelization across CUDA cores
            job = digital_engine.backend.run(bound_circuits, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Aer returns a single dict if len == 1, or a list of dicts if len > 1
            if not isinstance(counts, list):
                counts = [counts]
                
                

            return {"results": counts, "status": "success"}

        # --- PATH B: ANNEALING (Analog) ---
        elif run_type == "ANALOG":
            raw_payload = manifest["payload"] 
            
            # Convert JSON keys "0,1" back to tuples (0, 1) for the solver
            qubo_dict = {}
            for k, v in raw_payload.items():
                tuple_key = tuple(map(int, k.split(',')))
                qubo_dict[tuple_key] = v
            
            # Execute on GPU via SQA (Simulated Quantum Annealing)
            best_bitstring = analog_engine.execute(qubo_dict)

            logger.info(f"Analog Annealing Complete. Raw: {type(best_bitstring)}")
            
            best_bitstring = analog_engine.execute(qubo_dict)
            # OpenJij often returns np.int8; convert to standard int
            serializable_bits = [int(b) for b in best_bitstring] 
            return {"results": serializable_bits, "status": "success"}
        else:
            logger.error(f"No Run Type")
            return {"results": [], "status": "error", "message": "No Run Type"}

    except Exception as e:
        logger.error(f"Engine Crash: {e}")
        return {"results": [], "status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the server on localhost
    uvicorn.run(app, host="127.0.0.1", port=8000)