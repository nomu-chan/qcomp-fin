import numpy as np
from scipy.optimize import shgo, minimize, OptimizeResult
import json
import subprocess
from typing import Dict, Any, Union
import requests
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize

from utils.logging_mod import get_logging
logger = get_logging(__name__)

class QuantumEngineCommand:
    def __init__(self, gpu_venv_python: str = ".venv-gpu/bin/python", url="http://127.0.0.1:8000/execute"):
        self.url = url
        # Points to the python executable in your dedicated GPU venv
        self.gpu_python = gpu_venv_python
        self.entrypoint = "src/quantum/engine/entrypoint.py"
        
    def verify_landscape_with_annealing(self, qubo_dict: dict):
        """
        Sends the raw QUBO to the server for Simulated Quantum Annealing.
        Use this to find the 'True' global minimum to compare against QAOA.
        """
        manifest = {
            "type": "ANALOG",
            "payload": qubo_dict, # Send the raw dictionary
            "num_reads": 1000
        }
        
        # This calls the /execute endpoint on your FastAPI server
        
        # This replaces subprocess.run()
        response = requests.post(self.url, json=manifest)
        data = response.json()
        
        if data["status"] == "error":
            raise RuntimeError(f"GPU Server Error: {data['message']}")
        
        return response["results"] # Returns the best bitstring

    def run_server(self, run_type: str, payload: str, **kwargs) -> Union[Dict[str, Any], list[Dict[str, int]]]:
        manifest = {
            "type": run_type,
            "payload": payload,
            "bindings": kwargs.get("bindings"),
            "shots": kwargs.get("shots", 512)
        }
        
        response = requests.post(self.url, json=manifest)
        data = response.json()
        
        if data.get("status") == "error":
            raise RuntimeError(f"GPU Server Error: {data['message']}")
            
        results = data.get("results", [])

        # --- FIX: ANALOG returns the whole bitstring ---
        if run_type == "ANALOG":
            return data # Return the full dict so bridge.py can .get("status")

        # --- GATE_BASED logic stays the same ---
        bindings = kwargs.get("bindings")
        if isinstance(bindings, list) and len(bindings) > 1:
            return results
        
        return results[0] if results else {}

    def run_digitial_subprocess(self, run_type: str, payload: Union[str, dict], **kwargs) -> Any:
        """
        Dispatches jobs to the GPU Venv.
        run_type: 'GATE_BASED' or 'ANALOG'
        payload: QASM string (for Gate) or QUBO dict (for Analog)
        """
        manifest = {
            "type": run_type,
            "payload": payload,
            "bindings": kwargs.get("bindings"),
            "shots": kwargs.get("shots", 512),
            "num_reads": kwargs.get("num_reads", 1000)
        }
        logger.info(f"Sent GPU manifest: {manifest.__str__()}")
        process = subprocess.run(
            [self.gpu_python, self.entrypoint],
            input=json.dumps(manifest),
            text=True,
            capture_output=True
        )

        if process.returncode != 0:
            logger.error(f"Engine Venv Error: {process.stderr}")
            raise RuntimeError(f"GPU Engine failed: {process.stderr}")

        return json.loads(process.stdout)

class MinimizerCommand:
    def __init__(self, qaoa_layers=2, shots=4096, total_energy_gain_scalar=1.0, seed=67, maxiter=1024, global_samples=1024):
        self.qaoa_layers_p = qaoa_layers
        self.shots = shots
        self.total_energy_gain_scalar = total_energy_gain_scalar
        self.seed = seed
        self.maxiter = maxiter
        self.global_samples = global_samples
        self.executor = QuantumEngineCommand()

    def _build_q_matrix(self, qubo_dict: dict, n: int) -> np.ndarray:
        Q = np.zeros((n, n))
        for (i, j), coeff in qubo_dict.items():
            if i < n and j < n:
                Q[i, j] = coeff
        return Q

    # src/quantum/models/minimizer.py

    def _cost_function_batched(self, batch_params, qasm_str, qubo_dict, constant):
        """
        batch_params: List[np.ndarray] -> [[g1, b1], [g2, b2], [g3, b3]]
        """
        batch_params = np.atleast_2d(batch_params)
        p = batch_params.shape[1] // 2
        bindings_list = []
        
        for point in batch_params:
            # Match the SERVER'S expectation: a dict with 'betas' and 'gammas' lists
            payload_entry = {
                "gammas": point[:p].tolist(),
                "betas": point[p:].tolist()
            }
            bindings_list.append(payload_entry)

        # Note: Using 'bindings' here because your server code 
        # uses manifest.get("bindings", [])
        response = self.executor.run_server(
            run_type="GATE_BASED",
            payload=qasm_str,
            bindings=bindings_list, 
            shots=self.shots
        )

        # --- THE ROBUST UNPACKER ---
        energies = []
            # Since run_digital now returns results[0] for single or results (list) for batch:
        if isinstance(response, list):
            # We have our list of count dicts!
            for counts in response:
                # If the server wrapped counts in a 'counts' key, use .get(), 
                # otherwise use the dict directly.
                actual_counts = counts.get("counts", counts) if isinstance(counts, dict) else counts
                energy = self._calculate_expectation(actual_counts, qubo_dict) + constant
                energies.append(energy)
        else:
            # Single result fallback
            actual_counts = response.get("counts", response) if isinstance(response, dict) else response
            energies.append(self._calculate_expectation(actual_counts, qubo_dict) + constant)
        return energies
    
    def _calculate_expectation(self, counts: dict, qubo_dict: dict) -> float:
        """
        Computes the average energy from a dictionary of measurement counts.
        counts: {'00': 512, '11': 512, ...}
        qubo_dict: {(0,0): 1.0, (1,1): 1.0, (0,1): 0.5}
        """
        total_energy = 0.0
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            # OpenQASM bitstrings are often reversed (LSB at index 0)
            # We reverse it back to match index i to bitstring[i]
            bits = [int(b) for b in bitstring[::-1]]
            
            # Calculate Energy for this specific bitstring
            string_energy = 0.0
            for (i, j), coeff in qubo_dict.items():
                # If i == j, it's a linear term: coeff * x_i
                # If i != j, it's a quadratic term: coeff * x_i * x_j
                string_energy += coeff * bits[i] * bits[j]
            
            # Weight by the frequency of this result
            total_energy += string_energy * count

        return total_energy / total_shots
    
    def _calculate_energy(self, counts: dict | str, qubo_dict: dict, constant: float) -> float:
        """
        Calculates the expectation value <H> from measurement counts.
        counts: {'0101': 12, '1100': 40, ...}
        qubo_dict: {(0, 1): 1.0, (1, 2): -0.5, ...} (The interaction matrix)
        """
        # If counts is a string (e.g., '1010'), convert it to a dict format
        if isinstance(counts, str):
            counts = {counts: self.shots} 

        total_energy = 0.0
        # Now .values() will definitely work
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            # 1. Convert bitstring '0101' to list of integers [0, 1, 0, 1]
            # Note: Qiskit usually orders bits as [q_n, ..., q_0]
            nodes = [int(bit) for bit in reversed(bitstring)]
            
            # 2. Calculate energy of this specific configuration
            config_energy = 0.0
            for (i, j), weight in qubo_dict.items():
                # For Binary variables, cost is weight * x_i * x_j
                if i < len(nodes) and j < len(nodes):
                    config_energy += weight * nodes[i] * nodes[j]
            
            # 3. Add to weighted average
            total_energy += (config_energy + constant) * count

        # Return the expectation value (mean energy)
        return total_energy / total_shots

    def _cost_function(self, 
        angles: np.ndarray, 
        qasm_str: str, 
        qubo_dict: dict, 
        constant: float) -> float:
        if np.isnan(angles).any():
            logger.warning("L-BFGS-B attempted to pass NaN angles. Recovering with high penalty.")
            # Return a massive energy value to tell the optimizer 'Go Back!'
            return 1e6
        p = len(angles) // 2
        bindings = {
            "gammas": angles[:p].tolist(),
            "betas": angles[p:].tolist()
        }

        # Resolve Pylance reportArgumentType: Passing qasm_str (payload)
        try:
            result_counts = self.executor.run_server(payload=qasm_str, bindings=bindings, shots=self.shots, run_type="GATE_BASED")
            final_energy = self._calculate_energy(result_counts, qubo_dict, constant)
            return float(final_energy)
        except Exception as e:
            logger.error(f"Cost Function Crash at p={p}!")
            logger.error(f"Offending Angles: {angles.tolist()}")
            raise e

    def _gradient_batched(self, params, qasm_str, qubo_dict, constant):
        
        # For 4th digit accuracy, central difference is a must
        epsilon = 0.00005
        
        # Generate (params + eps) AND (params - eps)
        plus_points = [params + (np.eye(len(params))[i] * epsilon) for i in range(len(params))]
        minus_points = [params - (np.eye(len(params))[i] * epsilon) for i in range(len(params))]
        
        all_points = plus_points + minus_points
        energies = self._cost_function_batched(all_points, qasm_str, qubo_dict, constant)
        
        n = len(params)
        plus_energies = energies[:n]
        minus_energies = energies[n:]
        
        # Central difference formula: (f(x+h) - f(x-h)) / 2h
        grad = [(plus_energies[i] - minus_energies[i]) / (2 * epsilon) for i in range(n)]
        grad = np.array(grad)

        # CRITICAL: Replace NaNs or Infs with 0.0 before returning to SciPy
        if np.isnan(grad).any() or np.isinf(grad).any():
            logger.warning(f"Invalid gradient detected at {params}. Zeroing out.")
            return np.zeros_like(params)

        return np.clip(grad, -5.0, 5.0)

    def minimize_cost_global_batched(self, qasm_str: str, qubo_dict: dict, constant: float, p: int):
        """
        1. Global Exploration: Batch-sample the landscape using Sobol sequences.
        2. Local Refinement: Start a local search from the best discovered point.
        """
        # --- PHASE 1: GLOBAL BATCH SAMPLING ---
        n_samples = self.global_samples  # Number of points to test at once
        sampler = qmc.Sobol(d=2*p, scramble=True, rng=self.seed)
        sample_points = sampler.random(n_samples)
        
        # Scale Sobol points from [0, 1] to [0, pi]
        scaled_points = sample_points * np.pi 
        
        logger.info(f"Blasting GPU with batch of {n_samples} Sobol points...")
        
        # Use the 'bindings_list' logic we discussed for the GPU Server
        # This sends ONE request to the server
        energies = self._cost_function_batched(scaled_points, qasm_str, qubo_dict, constant)
        
        # Find the best starting candidate
        best_idx = np.argmin(energies)
        x0 = scaled_points[best_idx]
        x0 = x0 + np.random.normal(0, 0.05, len(x0))
        x0 = np.clip(x0, 0, np.pi) # Keep within bounds
        
        if energies[best_idx] <= (1.0 / self.shots):
            logger.info("Global sampling hit the theoretical minimum. Skipping refinement.")
            return OptimizeResult(
                x=x0, 
                fun=energies[best_idx], 
                status=0, 
                success=True, 
                message="Global minimum reached"
            )
        
        logger.info(f"Global best found at E={energies[best_idx]:.4f}. Starting local refinement...")

        # --- PHASE 2: LOCAL REFINEMENT ---
        # Now we switch to sequential mode because local search depends on the previous step
        res = minimize(
            self._cost_function, # Your standard 1-by-1 cost function
            x0=x0,
            args=(qasm_str, qubo_dict, constant),
            method='L-BFGS-B',
            bounds=[(0, np.pi)] * (2 * p),
            jac=self._gradient_batched,
            options={
                'maxiter': self.maxiter, 
                'ftol': 1e-7,  # Loosen tolerance for quantum noise
                'gtol': 1e-7,   # Prevent infinite loops on flat gradients} 
                'eps': 1e-3
            }
        )
        
        return res


    def minimize_cost_global(self, qasm_str: str, qubo_dict: dict, constant: float, p: int) -> OptimizeResult:
        bounds = [(0, np.pi)] * (2 * p)
        return shgo(
            self._cost_function, 
            bounds, 
            args=(qasm_str, qubo_dict, constant),
            n=1024, iters=5, sampling_method='sobol',
            options={'seed': self.seed},
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': self.maxiter}}
        )

    def minimize_cost_local(self, qasm_str: str, qubo_dict: dict, constant: float, x0: np.ndarray, p_level: int) -> OptimizeResult:
        res = minimize(
            self._cost_function,
            x0=x0,
            args=(qasm_str, qubo_dict, constant),
            jac=self._gradient_batched,  # <--- CRITICAL: One RTT per iteration
            method='L-BFGS-B',
            bounds=[(0, np.pi)] * (2 * p_level),
            options={'maxiter': 15, 'ftol': 1e-6} 
        )
        return res
        
    