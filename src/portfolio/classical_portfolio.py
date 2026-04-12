from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, OrderedDict, Any, cast
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from src.utils.logging_mod import logging
from src.portfolio.portfolio_base import PortfolioBase, PortfolioResult
from src.financial_context.command import FinancialContextCommand
from src.portfolio.quantum_portfolio import QportHyperparameterProduct
from pypfopt import DiscreteAllocation
import numpy as np
import gurobipy as gp
logger = logging.getLogger(__name__)

class ClassicalPortfoilioBase(PortfolioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand, hparams: QportHyperparameterProduct):
        super().__init__(name, financial_context)
        self.hparams = hparams
        
    def calculate_qubo_energy(self, weights: OrderedDict[str, int], mu: np.ndarray, sigma: np.ndarray, ):
        """
        Calculates the Hamiltonian energy of a weight vector to compare 
        Classical vs Quantum performance.
        """
        w = np.array(list(weights.values()))
        
        # 1. Risk Component (Quadratic)
        risk = self.hparams.lambda_risk * (w.T @ sigma @ w)
        
        # 2. Reward Component (Linear)
        reward = self.hparams.lambda_reward * (mu.T @ w)
        
        # 3. Cardinality Penalty
        # We count how many assets are 'active' (weight > epsilon)
        active_count = np.sum(w > 1e-4)
        cardinality_penalty = self.hparams.lambda_cardinality * (active_count - self.hparams.k_cardinality)**2
        
        # Total Energy (Lower is better)
        total_energy = risk - reward + cardinality_penalty
        
        return float(total_energy)


class IdealClassicalPortfolio(ClassicalPortfoilioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand, hparams: QportHyperparameterProduct):
        super().__init__(name, financial_context, hparams)

    def run(self) -> PortfolioResult:
        # 1. Get tickers and moments
        context = self.financial_context.get_context()
        tickers = context.tickers  # This is the list of names ["AAPL", "BTC", etc.]
        mu, covariance = context.get_moments(False, False)

        # 2. Ensure mu has ticker names as the index
        # PyPortfolioOpt uses the index of mu to label the weights
        if not isinstance(mu, pd.Series):
            mu = pd.Series(mu, index=tickers)
        
        # 3. Optimize
        ef = EfficientFrontier(mu, covariance)
        ef.max_sharpe()
        
        # clean_weights() now returns { "AAPL": 0.5, "BTC": 0.5 ... }
        raw_weights = ef.clean_weights()
        
        # 4. Cast to OrderedDict for the PortfolioResult
        cleaned_weights = OrderedDict({str(k): float(v) for k, v in raw_weights.items()})
        
        performance = ef.portfolio_performance(verbose=False)
        expected_return = float(cast(Any, performance[0]))
        volatility = float(cast(Any, performance[1]))
        sharpe_ratio = float(cast(Any, performance[2]))
        config = OrderedDict({
            "mu":mu,
            "covariance": covariance,
        })
        return PortfolioResult(
            method_name=self.name,
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            config=config
        )

class DiscretePortfolio(ClassicalPortfoilioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand, hparams: QportHyperparameterProduct, k_assets: int = 10):
        super().__init__(name, financial_context, hparams)
        self.k_assets = k_assets  # Accept the cardinality hyperparameter

    def weights_to_bitstring(self, weights: OrderedDict) -> list[int]:
        """
        Converts classical weights back into the binary bitstring 
        used by the QUBO model.
        """
        n_bits = self.hparams.n_bits
        total_units = (2**n_bits) - 1
        bitstring = []

        for ticker, weight in weights.items():
            # 1. Convert weight back to the integer 'unit' count (0 to 7 for 3-bits)
            # We round to avoid floating point noise from the classical solver
            units = int(round(weight * total_units))
            
            # 2. Convert integer to binary list (e.g., 3 -> [1, 1, 0] for 3 bits)
            # We use format(units, f'0{n_bits}b') to get '011' and reverse it 
            # so index 0 is 2^0, index 1 is 2^1, etc.
            binary_str = format(units, f'0{n_bits}b')[::-1] 
            unit_bits = [int(b) for b in binary_str]
            
            bitstring.extend(unit_bits)

        return bitstring
    
        
    def _calculate_energy(self, bitstring: list[int], qubo_dict: dict) -> float:
        """Calculates the QUBO energy for a specific bitstring: x^T Q x"""
        energy = 0.0
        for (i, j), coeff in qubo_dict.items():
            if bitstring[i] == 1 and bitstring[j] == 1:
                energy += coeff
        return energy

    def map_landscape_ruggedness(self, optimal_bitstring, qubo_dict: dict, radius: int = 1) -> Dict[str, Any]:
        """
        Probes the neighborhood of the optimal bitstring to measure ruggedness.
        """
        logger.info(f"op_bitstring: {optimal_bitstring}")
        if isinstance(optimal_bitstring, str):
            base_bits = [int(b) for b in optimal_bitstring if b in "01"]
        else:
            base_bits = list(optimal_bitstring)

        base_energy = self._calculate_energy(base_bits, qubo_dict)
        
        neighbor_energies = []
        n_vars = len(optimal_bitstring)

        # Flip bits to explore neighbors at Hamming Distance 1
        for i in range(n_vars):
            neighbor = base_bits[:]
            neighbor[i] = 1 - neighbor[i] # Flip bit
            neighbor_energies.append(self._calculate_energy(neighbor, qubo_dict))

        # Metrics
        avg_neighbor_energy = np.mean(neighbor_energies)
        energy_std = np.std(neighbor_energies)
        # Ruggedness Coefficient: High Std/Mean ratio indicates a 'spiky' landscape
        ruggedness_coeff = energy_std / abs(base_energy) if base_energy != 0 else 0

        logger.info(f"Landscape Ruggedness Report:")
        logger.info(f" - Optimal Energy: {base_energy:.4f}")
        logger.info(f" - Mean Neighbor Energy (Dist 1): {avg_neighbor_energy:.4f}")
        logger.info(f" - Energy Std Dev: {energy_std:.4f}")
        logger.info(f" - Ruggedness Coefficient: {ruggedness_coeff:.4f}")

        return {
            "base_energy": base_energy,
            "neighbor_energies": str(neighbor_energies),
            "mean_neighborhood_energy": avg_neighbor_energy,
            "ruggedness": ruggedness_coeff
        }
        
    def build_qubo(self):
        """
        Constructs the QUBO dictionary for the Portfolio Optimization problem.
        Returns: {(i, j): coefficient}
        """
        context = self.financial_context.get_context()
        mu, sigma = context.get_moments(False, False)
        tickers = context.tickers
        
        n_assets = len(tickers)
        n_bits = self.hparams.n_bits
        total_bits = n_assets * n_bits
        
        # Pre-calculate unit values for the binary expansion: [1, 2, 4, ...]
        unit_values = [2**k for k in range(n_bits)]
        
        qubo = {}

        # 1. RISK & REWARD (The Quadratic & Linear terms)
        for i in range(n_assets):
            for k in range(n_bits):
                # Global index for asset i, bit k
                idx_ik = i * n_bits + k
                
                # --- Linear Term (Reward + Self-Risk) ---
                # Reward: -lambda_reward * mu_i * 2^k
                reward_contribution = -self.hparams.lambda_reward * mu[i] * unit_values[k]
                
                # Risk Self-Interaction: lambda_risk * sigma_ii * (2^k)^2
                risk_self = self.hparams.lambda_risk * sigma[i, i] * (unit_values[k]**2)
                
                qubo[(idx_ik, idx_ik)] = qubo.get((idx_ik, idx_ik), 0) + reward_contribution + risk_self

                # --- Quadratic Term (Cross-Asset Risk) ---
                for j in range(n_assets):
                    for m in range(n_bits):
                        idx_jm = j * n_bits + m
                        
                        # Avoid double counting self-interaction and only do i < j for efficiency
                        if idx_ik < idx_jm:
                            # Interaction: 2 * lambda_risk * sigma_ij * 2^k * 2^m
                            # (The '2' comes from the symmetry of the sigma matrix)
                            risk_interaction = 2 * self.hparams.lambda_risk * sigma[i, j] * unit_values[k] * unit_values[m]
                            qubo[(idx_ik, idx_jm)] = qubo.get((idx_ik, idx_jm), 0) + risk_interaction

        # 2. CARDINALITY PENALTY (Optional)
        # Note: Implementing a strict cardinality in bits usually requires indicator variables.
        # For a 'soft' cardinality on units, we use: lambda_card * (sum(w_i) - K_target)^2
        if self.hparams.lambda_cardinality > 0:
            k_target = self.hparams.k_cardinality
            l_card = self.hparams.lambda_cardinality
            
            # Expanding (sum(2^k * x_ik) - K)^2
            # = sum(2^k * x_ik)^2 - 2*K*sum(2^k * x_ik) + K^2
            for i in range(n_assets):
                for k in range(n_bits):
                    idx_ik = i * n_bits + k
                    
                    # Linear part of the square: l_card * ((2^k)^2 - 2 * K * 2^k)
                    qubo[(idx_ik, idx_ik)] += l_card * (unit_values[k]**2 - 2 * k_target * unit_values[k])
                    
                    for j in range(n_assets):
                        for m in range(n_bits):
                            idx_jm = j * n_bits + m
                            if idx_ik < idx_jm:
                                # Quadratic part: 2 * l_card * 2^k * 2^m
                                qubo[(idx_ik, idx_jm)] += 2 * l_card * unit_values[k] * unit_values[m]

        return qubo
    
    def run(self) -> PortfolioResult:
        # 1. Get tickers and moments
        context = self.financial_context.get_context()
        tickers = context.tickers  # This is the list of names ["AAPL", "BTC", etc.]
        mu, covariance = context.get_moments(False, False)

        # 2. Ensure mu has ticker names as the index
        # PyPortfolioOpt uses the index of mu to label the weights
        if not isinstance(mu, pd.Series):
            mu = pd.Series(mu, index=tickers)
        
        # 3. Optimize
        ef = EfficientFrontier(mu, covariance)
        ef.max_sharpe()
        
        # clean_weights() now returns { "AAPL": 0.5, "BTC": 0.5 ... }
        raw_weights = ef.clean_weights()
    
        # NEW: Match the 'n_bits' granularity
        # If n_bits=3, you have 7 discrete 'slots' per asset (2^3 - 1)
        # 1. Force total_units to float to avoid casting errors during division
        total_units = float((2**self.hparams.n_bits) - 1)

        # 2. Ensure latest_prices is a Series of floats
        # PyPortfolioOpt internals sometimes choke if these are ints
        prices_series = pd.Series([1.0] * len(tickers), index=tickers, dtype=float)

        da = DiscreteAllocation(
            raw_weights, 
            latest_prices=prices_series, 
            total_portfolio_value=total_units # type: ignore
        )

        # This is usually where the 'divide' ufunc error happens 
        allocation, leftover = da.greedy_portfolio()
        
        # Convert allocation (e.g., {"AAPL": 3}) back to weights (3/7)
        cleaned_weights = OrderedDict({t: float(allocation.get(t, 0) / total_units) for t in tickers})
        w_arr = np.array(list(cleaned_weights.values()))
        expected_return = np.dot(w_arr, mu)
        volatility = np.sqrt(np.dot(w_arr.T, np.dot(covariance, w_arr)))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        config = OrderedDict({
            "mu":mu,
            "covariance": covariance,
        })
        return PortfolioResult(
            method_name=self.name,
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            config=config
        )
        
        
class DiscreteMILPPortfolio(ClassicalPortfoilioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand, hparams: QportHyperparameterProduct):
        super().__init__(name, financial_context, hparams)

    def weights_to_bitstring(self, weights: OrderedDict) -> list[int]:
        n_bits = self.hparams.n_bits
        total_units = (2**n_bits) - 1
        bitstring = []
        for ticker, weight in weights.items():
            units = int(round(weight * total_units))
            # Format to binary, reverse so index 0 is least significant bit
            binary_str = format(units, f'0{n_bits}b')[::-1] 
            bitstring.extend([int(b) for b in binary_str])
        return bitstring
    
    def _calculate_energy(self, bitstring: list[int], qubo_dict: dict) -> float:
        energy = 0.0
        for (i, j), coeff in qubo_dict.items():
            if bitstring[i] == 1 and bitstring[j] == 1:
                energy += coeff
        return energy

    def map_landscape_ruggedness(self, optimal_bitstring, qubo_dict: dict) -> Dict[str, Any]:
        if isinstance(optimal_bitstring, str):
            base_bits = [int(b) for b in optimal_bitstring if b in "01"]
        else:
            base_bits = list(optimal_bitstring)

        base_energy = self._calculate_energy(base_bits, qubo_dict)
        neighbor_energies = []
        n_vars = len(base_bits)

        for i in range(n_vars):
            neighbor = base_bits[:]
            neighbor[i] = 1 - neighbor[i] 
            neighbor_energies.append(self._calculate_energy(neighbor, qubo_dict))

        avg_neighbor_energy = np.mean(neighbor_energies)
        energy_std = np.std(neighbor_energies)
        ruggedness_coeff = energy_std / abs(base_energy) if base_energy != 0 else 0

        return {
            "base_energy": base_energy,
            "neighbor_energies": str(neighbor_energies),
            "mean_neighborhood_energy": avg_neighbor_energy,
            "ruggedness": ruggedness_coeff
        }
        
    def build_qubo(self):
        context = self.financial_context.get_context()
        mu, sigma = context.get_moments(False, False)
        n_assets = len(context.tickers)
        n_bits = self.hparams.n_bits
        unit_values = [2**k for k in range(n_bits)]
        
        qubo = {}
        # 1. RISK & REWARD
        for i in range(n_assets):
            for k in range(n_bits):
                idx_ik = i * n_bits + k
                # Linear: -Reward + Risk Self-Interaction
                reward_term = -self.hparams.lambda_reward * mu[i] * unit_values[k]
                risk_self = self.hparams.lambda_risk * sigma[i, i] * (unit_values[k]**2)
                qubo[(idx_ik, idx_ik)] = qubo.get((idx_ik, idx_ik), 0) + reward_term + risk_self

                for j in range(n_assets):
                    for m in range(n_bits):
                        idx_jm = j * n_bits + m
                        if idx_ik < idx_jm:
                            # Cross-Asset Risk: 2 * lambda * sigma * bit_i * bit_j
                            risk_interaction = 2 * self.hparams.lambda_risk * sigma[i, j] * unit_values[k] * unit_values[m]
                            qubo[(idx_ik, idx_jm)] = qubo.get((idx_ik, idx_jm), 0) + risk_interaction

        # 2. CARDINALITY (Quadratic Expansion)
        if self.hparams.lambda_cardinality > 0:
            K = self.hparams.k_cardinality
            L = self.hparams.lambda_cardinality
            # (sum(w_i) - K)^2 = sum(w_i^2) + 2*sum(w_i*w_j) - 2K*sum(w_i) + K^2
            for i in range(n_assets):
                for k in range(n_bits):
                    idx_ik = i * n_bits + k
                    # Linear: L * (2^k^2 - 2 * K * 2^k)
                    qubo[(idx_ik, idx_ik)] += L * (unit_values[k]**2 - 2 * K * unit_values[k])
                    for j in range(n_assets):
                        for m in range(n_bits):
                            idx_jm = j * n_bits + m
                            if idx_ik < idx_jm:
                                # Quadratic: 2 * L * 2^k * 2^m
                                qubo[(idx_ik, idx_jm)] += 2 * L * unit_values[k] * unit_values[m]
        return qubo
    def solve_exact_gurobi(self, qubo_dict, n_vars):
        model = gp.Model("Exact_MIQP")
        model.setParam('OutputFlag', 0)
        x = model.addVars(n_vars, vtype=gp.GRB.BINARY, name="x")
        
        obj = gp.QuadExpr()
        # FIX for reportAttributeAccessIssue: Ensure we call .items() on the argument
        # previously you might have been accidentally referencing 'self' or the class
        for (i, j), coeff in qubo_dict.items():
            obj += coeff * x[i] * x[j]
        
        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.optimize()
        
        # Access the solution (X attribute)
        return [int(x[i].X) for i in range(n_vars)]
    
    def run(self) -> PortfolioResult:
        context = self.financial_context.get_context()
        mu, sigma = context.get_moments(False, False)
        n_assets = len(context.tickers)
        n_bits = self.hparams.n_bits
        
        qubo_dict = self.build_qubo()
        total_bits = n_assets * n_bits

        # FIX for reportCallIssue: Ensure internal methods are called via 'self'
        optimal_bitstring = self.solve_exact_gurobi(qubo_dict, total_bits)

        # Decode weights
        decoded_weights = OrderedDict()
        scaling = float((2**n_bits) - 1)
        for i, ticker in enumerate(context.tickers):
            bits = optimal_bitstring[i*n_bits : (i+1)*n_bits]
            units = sum(b * (2**k) for k, b in enumerate(bits))
            decoded_weights[ticker] = units / scaling

        # Normalized Performance
        w_arr = np.array(list(decoded_weights.values()))
        if np.sum(w_arr) > 0:
            w_arr = w_arr / np.sum(w_arr)
            
        exp_ret = np.dot(w_arr, mu)
        vol = np.sqrt(np.dot(w_arr.T, np.dot(sigma, w_arr)))
        
        # Ensure this call also has its required arguments (optimal_bitstring, qubo_dict)
        landscape_stats = self.map_landscape_ruggedness(optimal_bitstring, qubo_dict)

        return PortfolioResult(
            method_name=self.name,
            weights=decoded_weights,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=exp_ret / vol if vol > 0 else 0,
            config=OrderedDict({
                "ruggedness": landscape_stats["ruggedness"],
                "base_energy": landscape_stats["base_energy"],
                "n_bits": n_bits
            })
        )