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
class QportDiversification(BaseQuantumPortfolio):
    def __init__(
        self,
        name="IntegerDiversification",
        lam_div: float = 50.0, 
        upper_bound: int = 3, 

        **kwargs
    ):
        """
        Integer-based portfolio with a penalty for lack of diversity.
        Upper bound of 3 implies 2 bits per asset in the QUBO.
        """
        super().__init__(name=name, **kwargs)
        self.lam_div = lam_div
        self.upper_bound = upper_bound

    def _get_instance_data(self, mu: np.ndarray, sigma: np.ndarray) -> dict:
        # K_lots is set to 2*N as a heuristic for moderate allocation
        return {
            "mu": mu, 
            "sigma": sigma, 
            "K_lots": 2 * len(mu), 
            "lam_div": self.lam_div,
            "lam_card": self.lam_card
        }

    def _build_symbolic_problem(self, n_assets: int) -> jm.Problem:
        problem = jm.Problem(self.name)
        
        @problem.update
        def _(p: jm.DecoratedProblem):
            # 1. Define Variables and Placeholders
            mu_ph = p.Placeholder("mu", dtype=float, ndim=1)
            sigma_ph = p.Placeholder("sigma", dtype=float, ndim=2)
            K_lots = p.Natural("K_lots")
            L_div = p.Placeholder("lam_div", dtype=float)
            L_card = p.Placeholder("lam_card", dtype=float)
            
            # Integer variable (0 to 3)
            x = p.IntegerVar("x", shape=(n_assets,), lower_bound=0, upper_bound=self.upper_bound)
            
            # 2. Objective Terms
            returns = jm.sum(mu_ph[i] * x[i] for i in range(n_assets))
            risk = jm.sum(x[i] * sigma_ph[i, j] * x[j] for i in range(n_assets) for j in range(n_assets))
            
            # 3. Penalties
            # Cardinality: Ensures total number of "lots" matches target
            lot_penalty = L_card * (jm.sum(x[i] for i in range(n_assets)) - K_lots)**2
            
            # Diversification: Penalizes high concentration in single assets (sum of squares)
            div_penalty = L_div * jm.sum(x[i]**2 for i in range(n_assets))
            
            # Combine based on risk_coefficient (self.lam)
            p += self.lam * risk - (1 - self.lam) * returns + lot_penalty + div_penalty
            
        return problem

    def _decode_results(self, best_bitstring: list[int], n_assets: int) -> list[int]:
        """
        Decodes the flat bitstring into integer lots.
        Assumes 2 bits per asset for an upper bound of 3.
        """
        decoded_lots = []
        # Each asset i is represented by bits at [2*i] and [2*i + 1]
        for i in range(n_assets):
            # Binary to Integer: (bit_0 * 2^1) + (bit_1 * 2^0)
            val = best_bitstring[2*i] * 2 + best_bitstring[2*i+1]
            decoded_lots.append(int(val))
        return decoded_lots