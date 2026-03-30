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

logger = logging.getLogger(__name__)
        
class QportInteger(BaseQuantumPortfolio):
    def __init__(self, name: str="IntegerQuantum",lam_div: float = 5.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lam_div = lam_div

    def _get_instance_data(self, mu, sigma):
        return {"mu": mu, "sigma": sigma, "K_lots": 2*len(mu)}

    def _build_symbolic_problem(self, n_assets: int) -> jm.Problem:
        problem = jm.Problem(self.name)
        
        # The Decorated API allows you to use Pythonic for-loops/generators
        @problem.update
        def _(problem: jm.DecoratedProblem):
            # 1. Define all Placeholders (No hardcoded self.k here)
            mu_ph = problem.Placeholder("mu", dtype=float, ndim=1)
            sigma_ph = problem.Placeholder("sigma", dtype=float, ndim=2)
            K_lots = problem.Natural("K_lots") # Total number of lots allowed
            
            # Decision Variable: Now Integer instead of Binary
            # shape=(n_assets,), lower_bound=0, upper_bound=3 (for 2-qubit encoding)
            x = problem.IntegerVar("x", shape=(n_assets,), lower_bound=0, upper_bound=3)
            
            # Objective: Same structure, but x[i] can now be 0, 1, 2, or 3
            returns = jm.sum(mu_ph[i] * x[i] for i in range(n_assets))
            risk = jm.sum(
                x[i] * sigma_ph[i, j] * x[j] 
                for i in range(n_assets) for j in range(n_assets)
            )
            
            # Penalty: Total lots must equal K_lots
            lot_penalty = self.lam_card * (jm.sum(x[i] for i in range(n_assets)) - K_lots)**2
            
            problem += self.lam * risk - (1 - self.lam) * returns + lot_penalty
        return problem

    def _decode_results(self, bits, n_assets):
        # Binary decoding: 2 bits per asset
        return [bits[2*i]*2 + bits[2*i+1] for i in range(n_assets)]