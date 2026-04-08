import jijmodeling as jm
from typing import Dict, Any
from qamomile.optimization.binary_model.model import BinaryModel
from  qamomile.optimization.qaoa import QAOAConverter # Fix: Moved from .connectivity
from ommx.v1 import Instance
from qamomile.qiskit.transpiler import QiskitTranspiler
from utils.cache import LandscapeCacheProxy, generate_identifier
from dataclasses import dataclass
import numpy as np
from src.utils.logging_mod import logging

logger = logging.getLogger(__name__)

@dataclass
class QUBOProduct:
    qubo_dict: dict[tuple[int, int], float]
    constant: float
    params: Dict[str, Any]
    hash: str
    
class InstantiatorCommand:
    """
    Takes in symbolic problem,
    optimizes them and cache for later use
    outputs quantum models to be execute.
    
    Intermediate Representations:
        - QUBO
        - ExecutableProgram(Qiskit) 
    
    """
    def __init__(self, p_max: int = 4, maxiter: int = 200):
        self.p_max = p_max
        self.maxiter = maxiter
        self.transpiler = QiskitTranspiler()
        self.cache_manager = LandscapeCacheProxy()
    
    def get_qubo_prod(self, 
        problem: jm.Problem,            # from QProblemBuilder
        expected_data: list[str],
        instance_data: Dict[str, Any]
        ):
        """
        Filters and Validates data to match the strict Jijmodeling schema.
        """
        # Define structural keys that MUST affect the hash
        structural_metadata = ["n_bits", "n_assets", "k_cardinality"]
        
        # Keep math-required keys OR structural keys
        filtered_data = {
            key: value for key, value in instance_data.items() 
            if key in expected_data or key in structural_metadata
        }

        # 2. INJECTION: Add neutral defaults for expected keys that are missing
        for key in expected_data:
            if key not in filtered_data:
                logger.warning(f"⚠️ Injecting shape-aware fallback for: {key}")
                
                # We need the number of assets to create the correct shapes
                # Try to infer it from 'mu', or use a fallback if mu is also missing
                n_assets = len(instance_data.get("mu", []))
                
                if key == "sigma":
                    # Must be a 2D matrix (Identity matrix is a safe neutral)
                    filtered_data[key] = np.eye(n_assets).tolist() 
                elif key == "mu" or key == "esg":
                    # Must be a 1D vector
                    filtered_data[key] = np.zeros(n_assets).tolist()
                else:
                    # Scalars are fine as floats
                    filtered_data[key] = 0.0

        try:
            # 3. EVALUATE: Use the scrubbed 'filtered_data' instead of 'instance_data'
            eval_data = {k: v for k, v in filtered_data.items() if k in expected_data}
            instance = problem.eval(eval_data)
            
            qubo_dict, q_const = instance.to_qubo()
            
            # Use the FULL filtered_data (including n_bits) for the hash
            data_hash = generate_identifier(filtered_data)
            
            logger.info(f"✅ QUBO Compiled Successfully. Hash: {data_hash}")
            return QUBOProduct(qubo_dict, q_const, filtered_data, data_hash)

        except jm._jijmodeling.ModelingError as e:
            logger.error(f"❌ Strict Eval Failed. Expected symbols: {expected_data}")
            logger.error(f"Filtered keys provided: {list(filtered_data.keys())}")
            raise e

    def get_qiskit_program(self, qubo_product: QUBOProduct, level_p: int):
        instance = BinaryModel.from_qubo(qubo_product.qubo_dict)
        converter = QAOAConverter(instance)
        return converter.transpile(self.transpiler, p=level_p)
