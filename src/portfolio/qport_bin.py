from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, OrderedDict, Any, cast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pypfopt import EfficientFrontier, expected_returns, risk_models
from src.utils.logging_mod import logging
from src.utils.portfolio import Portfolio, PortfolioResult
import qamomile.circuit as qmc
import inspect
from qiskit_aer import AerSimulator
from qamomile.qiskit.transpiler import QiskitTranspiler
from  qamomile.optimization.qaoa import QAOAConverter # Fix: Moved from .connectivity
from qamomile.optimization.binary_model.model import BinaryModel
import jijmodeling as jm
from qiskit.primitives import StatevectorSampler
from src.quantum_engine.quantum_executor import LocalExecutor
from qamomile.optimization.binary_model.model import BinaryModel
from scipy.optimize import minimize
import random
from src.portfolio.qport_base import BaseQuantumPortfolio

import random
logger = logging.getLogger(__name__)

class QportBinary(BaseQuantumPortfolio):
    def __init__(self, name="BinaryQuantum", num_assets_to_select: int = 3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = num_assets_to_select

    def _get_instance_data(self, mu, sigma):
        return {"mu": mu, "sigma": sigma, "K": self.k, "lam_card": self.lam_card}

    def _build_symbolic_problem(self, n_assets: int) -> jm.Problem:
        problem = jm.Problem(self.name)
        
        # The Decorated API allows you to use Pythonic for-loops/generators
        @problem.update
        def _(problem: jm.DecoratedProblem):
            # Define all Placeholders (No hardcoded self.k here)
            mu_ph = problem.Placeholder("mu", dtype=float, ndim=1)
            sigma_ph = problem.Placeholder("sigma", dtype=float, ndim=2)
            
            # Placeholders for our specific portfolio logic
            K = problem.Natural("K")  # Target number of assets
            lam_card = problem.Placeholder("lam_card", dtype=int) # Penalty multiplier
            
            # Decision Variable
            N = mu_ph.len_at(0)
            x = problem.BinaryVar("x", shape=(N,))
            
            # Objective Function
            returns = jm.sum(mu_ph[i] * x[i] for i in range(n_assets))
            risk = jm.sum(
                x[i] * sigma_ph[i, j] * x[j] 
                for i in range(n_assets) for j in range(n_assets)
            )
            
            # We must explicitly add the penalty to the objective for QAOA 
            # to see the variables and generate qubits.
            cardinality_penalty = lam_card * (jm.sum(x[i] for i in range(n_assets)) - K)**2
            
            problem += self.lam * risk - (1 - self.lam) * returns + cardinality_penalty

            # Explicit Constraint (Still good practice for OMMX metadata)
            problem += problem.Constraint("cardinality", jm.sum(x[i] for i in range(n_assets)) == K)

        return problem

    def _decode_results(self, best_bitstring, n_assets):
        return [1 if b == 1 else 0 for b in best_bitstring]