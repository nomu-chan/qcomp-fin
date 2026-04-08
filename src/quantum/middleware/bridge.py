from src.quantum.middleware.instantiator import QUBOProduct, InstantiatorCommand
from src.quantum.middleware.minimizer import QuantumEngineCommand, MinimizerCommand
from src.utils.logging_mod import get_logging
import numpy as np
from etc.config import DATAPATH
import pickle
from scipy.optimize import OptimizeResult
from typing import Optional, cast
from qiskit import qasm3
import json

logger = get_logging(__name__)



class ModelBridgeCommand:
    def __init__(self, layers_p: int, shots=512, total_energy_gain_scalar=1.0, seed=67, maxiter=512, gpu_venv_python=".venv-gpu/bin/python"):
        self.layers_p = layers_p
        self.engine_worker = QuantumEngineCommand(gpu_venv_python)
        self.instantiator = InstantiatorCommand()
        self.minimizer_cmd = MinimizerCommand(
            qaoa_layers=layers_p,
            shots=shots,
            total_energy_gain_scalar=total_energy_gain_scalar,
            seed=seed,
            maxiter=maxiter
        )
        self.shots = shots

    def _interpolate_angles(self, old_angles: np.ndarray) -> np.ndarray:
        p = len(old_angles) // 2
        gammas, betas = old_angles[:p], old_angles[p:]
        
        # --- ADD LOGGING HERE ---
        if np.isnan(old_angles).any():
            logger.error(f"❌ Input to interpolation for p={p+1} contains NaNs!")
            
        new_angles = np.concatenate([np.append(gammas, gammas[-1]), np.append(betas, betas[-1])])
        
        logger.info(f"Warming p={p}->{p+1}: Shape {old_angles.shape} -> {new_angles.shape}")
        return new_angles

    def minimize_with_warming(self, product: QUBOProduct, is_using_cache: bool = True) -> OptimizeResult:
        current_angles: Optional[np.ndarray] = None
        last_min_result: Optional[OptimizeResult] = None

        for p_level in range(1, self.layers_p + 1):
            cache_file = DATAPATH / f"warm_{product.hash}_p{p_level}.pkl"
            
            if is_using_cache and cache_file.exists():
                with open(cache_file, 'rb') as f:
                    last_min_result = pickle.load(f)
                if last_min_result:
                    current_angles = last_min_result.x
                    continue

            # Generate QASM 3.0 string for the bridge
            exec_prog = self.instantiator.get_qiskit_program(product, p_level)
            qasm_str = qasm3.dumps(exec_prog.quantum_circuit) 

            logger.info(f"Layer p={p_level}: Starting Optimization...")

            if p_level == 1:
                last_min_result = self.minimizer_cmd.minimize_cost_global_batched(
                    qasm_str, product.qubo_dict, product.constant, p=1
                )
            else:
                if current_angles is None:
                    raise RuntimeError("p_level > 1 reached without p_level = 1 optimization.")
                
                x0 = self._interpolate_angles(current_angles)
                
                logger.info(f"🌡️ Warming Layer p={p_level}: Interpolated x0 from p={p_level-1}")
                logger.debug(f"x0 Vector: {x0.tolist()}")   
                last_min_result = self.minimizer_cmd.minimize_cost_local(
                    qasm_str, product.qubo_dict, product.constant, x0=x0, p_level=p_level
                )
            logger.info(f"✅ Layer p={p_level} Complete. Energy: {last_min_result.fun:.6f}")
            current_angles = last_min_result.x
            with open(cache_file, 'wb') as f:
                pickle.dump(last_min_result, f)

        if last_min_result is None:
            raise RuntimeError("Optimization failed to produce a result.")
            
        return cast(OptimizeResult, last_min_result)

    def sample_best_configuration(self, qprod: QUBOProduct, min_result: OptimizeResult):
        """
        Performs a high-shot sampling at the discovered global minimum.
        """
        logger.info("🔭 Sampling optimal quantum state for portfolio selection...")
        exec_prog = self.instantiator.get_qiskit_program(qprod, self.layers_p)
        qasm_str = qasm3.dumps(exec_prog.quantum_circuit) 
        # We use the minimizer's executor to run a final high-fidelity pass
        # min_result.x contains the [betas, gammas]
        final_counts = self.minimizer_cmd.executor.run_server(
            run_type="GATE_BASED",
            payload=qasm_str,
            bindings={
                "betas": min_result.x[:self.layers_p].tolist(),
                "gammas": min_result.x[self.layers_p:].tolist()
            },
            shots=self.shots
        )

        return final_counts
    
    def minimize_analog(self, product: QUBOProduct, is_using_cache=True):
        """
        Bridge to the Analog (Annealing) path on the GPU server.
        """
        # 1. Check Cache First
        cache_file = DATAPATH / "analog_saves" / f"analog_{product.hash}.pkl"
        
        if is_using_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                logger.info(f"♻️  Analog Cache Hit: {product.hash}")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load analog cache: {e}. Proceeding.")

        logger.info(f"🌀 Dispatching to Analog Annealer (No Cache)")

        # 2. Prepare JSON-compatible payload
        sanitized_payload = {
            f"{k[0]},{k[1]}" if isinstance(k, tuple) else f"{k},{k}": float(v) 
            for k, v in product.qubo_dict.items()
        }

        # 3. Call the server
        response_obj = self.minimizer_cmd.executor.run_server(
            run_type="ANALOG", 
            payload=sanitized_payload
        )

        # 4. Extract Data (Handling the 'int' return type)
        if isinstance(response_obj, dict):
            data = response_obj
        elif hasattr(response_obj, 'json'):
            data = response_obj.json()
        else:
            # If your executor returns '1' or '200', we need to check if it stored 
            # the result in an internal attribute, or if you need to fix the executor.
            raise RuntimeError(
                f"Executor returned {type(response_obj)} ({response_obj}). "
                "Please update src/quantum/middleware/minimizer.py to return the JSON dict."
            )

        # 5. Extract and Cache Results
        if data.get("status") == "success":
            # The server might return a dict: {"[1, 0, ...]": 1}
            # OR a list of dicts: [{"[1, 0, ...]": 1}]
            raw_results = data.get("results", {})

            # Handle case where results is a list instead of a dict
            if isinstance(raw_results, list):
                if len(raw_results) > 0 and isinstance(raw_results[0], dict):
                    raw_results = raw_results[0]
                else:
                    # If it's just a raw list [1, 0, 0...], wrap it into a dict
                    raw_results = {str(raw_results): 1}
            
            if not raw_results:
                raise RuntimeError("Analog server returned success but empty results.")

            # TRANSFORM: Clean stringified lists into flat bitstrings
            final_results = {
                "".join(c for c in str(k) if c in "01"): v 
                for k, v in raw_results.items()
            }

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(final_results, f)
                logger.info(f"💾 Analog dict cached: {cache_file.name}")
            except Exception as e:
                logger.error(f"Failed to cache: {e}")
                
            return final_results
        else:
            error_msg = data.get('message', 'Unknown Server Error')
            raise RuntimeError(f"Analog Execution Failed: {error_msg}")