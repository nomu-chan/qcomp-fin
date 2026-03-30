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
from typing import Mapping
from abc import ABC, abstractmethod
from qamomile.circuit.transpiler.executable import ExecutableProgram
from src.quantum_engine.quantum_executor import LocalExecutor

logger = logging.getLogger(__name__)
class BaseQuantumPortfolio(Portfolio, ABC):
    """
    Template Method pattern for Quantum Portfolios.
    Handles the common infrastructure: 
        1. Scaling
        2. QUBO
        3. QAOA
        4. Classical Optimization
        5. Efficiency Frontier
    """
    def __init__(self, name: str, risk_coefficient: float = 0.5, p: int = 2, 
                 lam_card: float = 20.0, seed: int = 67, maxiter: int = 200, k = 3):
        super().__init__(name)
        self.lam = risk_coefficient
        self.p = p
        self.lam_card = lam_card
        self.seed = seed
        self.maxiter = maxiter
        self.scalar_gain = 10.0
        self.k = 3
        self.shots = 256
        self.transpiler = QiskitTranspiler()

    @abstractmethod
    def _build_symbolic_problem(self, n_assets: int) -> jm.Problem:
        """Subclasses define their specific math problem here."""
        pass

    @abstractmethod
    def _decode_results(self, best_bitstring: list[int], n_assets: int) -> list[int | float]:
        """Subclasses define how to turn bits back into asset selections/lots."""
        pass

    def _compute_moments(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        mu = expected_returns.mean_historical_return(prices)
        covariance = pd.DataFrame(risk_models.CovarianceShrinkage(prices).ledoit_wolf())
        return mu, covariance

    def _scale_data(self, mu: pd.Series[float], covariance: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        std_scaler = StandardScaler()
        mu_scaled = std_scaler.fit_transform(mu.to_numpy().reshape(-1, 1)).flatten()
        cov_scaled = std_scaler.fit_transform(covariance.values)
        return mu_scaled * self.scalar_gain, cov_scaled * self.scalar_gain

    def _cost_function(self, angles: np.ndarray, exec_obj, executor, qubo_dict: dict, constant: float) -> float:
        # Integrated logic to handle dynamic bitstring widths
        p = len(angles) // 2
        samples = exec_obj.sample(executor, shots=self.shots, 
                                  bindings={"gammas": angles[:p].tolist(), "betas": angles[p:].tolist()})
        result_data = samples.result().results
        
        actual_n_bits = len(result_data[0][0])
        Q = np.zeros((actual_n_bits, actual_n_bits))
        for (i, j), coeff in qubo_dict.items():
            if i < actual_n_bits and j < actual_n_bits: Q[i, j] = coeff
                

        total_energy = sum((np.array(bits).T @ Q @ np.array(bits) + constant) * count 
                           for bits, count in result_data)
        return total_energy / sum(count for _, count in result_data)
    
    @abstractmethod
    def _get_instance_data(self, mu_scaled, cov_scaled) -> Mapping[str, int | float]:
        pass 

    def _setup_problem(self, prices: pd.DataFrame):
        mu, covariance = self._compute_moments(prices)
        mu_scaled, cov_scaled = self._scale_data(mu, covariance)
        problem = self._build_symbolic_problem(len(mu))
        instance = problem.eval(self._get_instance_data(mu_scaled, cov_scaled))
        qubo_dict, constant = instance.to_qubo()
        return qubo_dict, constant, mu, covariance
        
    def _convert_qubo_to_circuit(self, qubo_dict, constant):
        model = BinaryModel.from_qubo(qubo_dict, constant=constant)
        converter = QAOAConverter(model)
        exec_obj = converter.transpile(self.transpiler, p=self.p)
        executor = LocalExecutor(seed=self.seed)
        return exec_obj, executor
    
    def _minimize_cost(self, 
        exec_obj: ExecutableProgram, 
        executor: LocalExecutor, 
        qubo_dict: dict[tuple[int, int], float], 
        constant: float):
        return minimize(self._cost_function, 
            np.random.uniform(0, np.pi, 2 * self.p),
            args=(exec_obj, executor, qubo_dict, constant), 
            method='COBYLA')
    
    def _run_simulation_samples(self, 
        exec_obj: ExecutableProgram, 
        executor: LocalExecutor, 
        min_result, 
        mu: pd.Series[Any]):
        final_samples = exec_obj.sample(executor, 
            shots=1024, 
            bindings={
                "gammas": min_result.x[:self.p].tolist(), 
                "betas": min_result.x[self.p:].tolist()
        })
        best_bits, _ = max(final_samples.result().results, key=lambda x: x[1])
        selections = self._decode_results(best_bits, len(mu))
        return selections
    
    def _set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def run(self, prices: pd.DataFrame) -> PortfolioResult:
        self._set_seed()
        qubo_dict, constant, mu, covariance = self._setup_problem(prices)
        exec_obj, executor = self._convert_qubo_to_circuit(qubo_dict, constant)
        min_result = self._minimize_cost(exec_obj, executor, qubo_dict, constant)
        selections = self._run_simulation_samples(exec_obj, executor, min_result, mu)
        
        # 5. Hybrid Classical Refinement
        return self._finalize_portfolio(selections, mu, covariance, prices.columns)
    
    def _finalize_portfolio(
        self, 
        selections: list[int | float], 
        mu: pd.Series, 
        cov: pd.DataFrame, 
        columns: pd.Index
    ) -> PortfolioResult:
        """
        Refines quantum selections into a valid financial portfolio.
        Flow: Filter Assets -> Efficient Frontier -> (Fallback) Equal Weight -> (Crisis) Top K.
        """
        weights_series = pd.Series(0.0, index=columns)
        rf_rate = 0.02  # Benchmark risk-free rate
        
        # Extract identified assets (where lots > 0 or bit == 1)
        selected_assets = [columns[i] for i, val in enumerate(selections) if val > 0]
        
        # 2. Strategy: Attempt Modern Portfolio Theory (MPT) Optimization
        if len(selected_assets) > 0:
            mu_subset = mu[selected_assets]
            cov_subset = cov.loc[selected_assets, selected_assets]
            
            try:
                # We use weight_bounds=(0, 1) for a long-only portfolio
                ef = EfficientFrontier(mu_subset, cov_subset, weight_bounds=(0, 1))
                
                # Maximize Sharpe Ratio
                ef.max_sharpe(risk_free_rate=rf_rate)
                cleaned_weights = ef.clean_weights()
                
                weights_series.update(pd.Series(cleaned_weights))
                logger.info(f"Hybrid Success: EF optimized {len(selected_assets)} assets.")
                
            except Exception as e:
                # If MPT fails (e.g. non-positive definite matrix or negative returns)
                # We use the proportions suggested by the Quantum model (especially for Integer lots)
                logger.warning(f"EF Optimization failed. Falling back to Quantum Proportions. Error: {e}")
                total_val = sum(selections)
                if total_val > 0:
                    for i, val in enumerate(selections):
                        weights_series[columns[i]] = val / total_val
                else:
                    # Equal weights for the selected subset
                    weights_series[selected_assets] = 1.0 / len(selected_assets)
        
        else:
            # The Quantum Model picked nothing
            logger.error("CRITICAL: QAOA selected 0 assets. Executing Top-K Return Recovery.")
            # We pick the top 'K' assets by historical return to ensure a non-empty portfolio
            top_k_indices = mu.nlargest(self.k if hasattr(self, 'k') else 3).index
            weights_series[top_k_indices] = 1.0 / len(top_k_indices)

        portfolio_return = np.dot(weights_series, mu)
        w_v = weights_series.to_numpy()
        portfolio_volatility = np.sqrt(np.dot(w_v.T, np.dot(cov.to_numpy(), w_v)))
        sharpe = (portfolio_return - rf_rate) / (portfolio_volatility + 1e-9)

        return PortfolioResult(
            method_name=self.name,
            weights=OrderedDict({str(k): float(v) for k, v in weights_series.to_dict().items()}),
            expected_return=float(portfolio_return),
            volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe),
            config=OrderedDict({"assets_considered": len(selected_assets)})
        )