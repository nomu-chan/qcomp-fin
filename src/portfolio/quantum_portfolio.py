from src.portfolio.portfolio_base import PortfolioBase, PortfolioResult
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, OrderedDict, Any
import pandas as pd
import numpy as np
from src.financial_context.command import FinancialContext, FinancialContextCommand
from src.symbolics.hamiltonian_modelling import QuantumProblemModelingBuilder, IntegerQProb
from src.quantum.middleware.instantiator import InstantiatorCommand
from src.quantum.middleware.minimizer import MinimizerCommand
from src.quantum.middleware.bridge import ModelBridgeCommand
from src.symbolics.hamiltonians import RewardMuIntegerStrategy, RiskCovarianceIntegerStrategy, CardinalityBinaryStrategy, TransactionCostStrategy
from src.utils.logging_mod import get_logging
import numpy as np
import matplotlib.pyplot as plt
from src.portfolio.portfolio_base import QMethod
logger = get_logging(__name__)

@dataclass
class QportHyperparameterProduct:
    p_qaoa_layers: int = 4
    max_iterations: int = 1024
    shots: int = 1024
    seed: int = 67
    samples: int = 1024
    k_cardinality: int = 2
    total_energy_scalar: float = 1.0
    lambda_risk: float = 0.5
    lambda_reward: float = 0.5
    lambda_cardinality: float = 100.0
    lambda_turnover: float = 0
    lambda_esg: float = 1.0
    n_bits: int = 4
    mu_scalar: int = 10
    prev_weights: float | None = None


class QuantumPortfolioComposite(PortfolioBase, ABC):
    def __init__(self, name: str, 
        proxy: FinancialContextCommand,
        symbolics: QuantumProblemModelingBuilder,
        hparaproduct: QportHyperparameterProduct
        ) -> None:
        super().__init__(
            name=name, 
            financial_context=proxy
        )
        self.hparaproduct = hparaproduct
        self.proxy = proxy
        self.symbolics = symbolics 
        self.instantiator = InstantiatorCommand(
            p_max=hparaproduct.p_qaoa_layers, 
            maxiter= hparaproduct.max_iterations
        )
        self.minimizer = ModelBridgeCommand(layers_p=hparaproduct.p_qaoa_layers, shots=hparaproduct.shots, maxiter=hparaproduct.max_iterations)
        self.context = self.financial_context.get_context(False, False)
        logger.info(f"Moments of mu and cov {self.context.get_moments()}")
        self.problem, self.expected_variables = self.symbolics.build()
    
    def _plot_landscape(self, data):
        plt.figure(figsize=(8, 5))
        plt.hist(data['neighbor_energies'], bins=20, alpha=0.7, label='Neighbor Energies')
        plt.axvline(data['base_energy'], color='r', linestyle='dashed', label='Global Minima (Annealer)')
        plt.title("Energy Landscape Neighborhood (Hamming Distance = 1)")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
    def decode_quantum_result(self, bitstring, n_assets=10, n_bits=2):
        # Reshape bitstring into (n_assets, n_bits)
        bits_per_asset = np.array(bitstring).reshape(n_assets, n_bits)
        
        # Convert binary to integer weights
        powers_of_2 = 2**np.arange(n_bits)
        weights = np.dot(bits_per_asset, powers_of_2)
        
        # Calculate relative percentage
        total_weight = np.sum(weights)
        percentages = (weights / total_weight) * 100 if total_weight > 0 else weights
        
        print("--- 🎯 Quantum Asset Selection ---")
        for i, w in enumerate(weights):
            if w > 0:
                # FIX: Use the ticker name from context instead of "Asset #{i}"
                ticker_name = self.context.tickers[i] 
                print(f"{ticker_name}: {int(w)} units ({percentages[i]:.2f}%)")
    
    def run_quantum_annealing_analysis(self):
        
        instance = self.symbolics.get_instance_data(
            proxy = self.context,
            k_target = self.hparaproduct.k_cardinality,
            lambda_risk = self.hparaproduct.lambda_risk,
            lambda_reward = self.hparaproduct.lambda_reward,
            lambda_cardinality= self.hparaproduct.lambda_cardinality,
            lambda_esg=self.hparaproduct.lambda_esg,
            n_bits=self.hparaproduct.n_bits,
            lambda_turnover=self.hparaproduct.lambda_turnover
        )
        qprod = self.instantiator.get_qubo_prod(self.problem, self.expected_variables, instance)
        qubo_dict = qprod.qubo_dict

        # Log the energy statistics
        linear_biases = [v for k, v in qubo_dict.items() if k[0] == k[1]]
        quadratic_biases = [v for k, v in qubo_dict.items() if k[0] != k[1]]
        logger.info(f"n_bits: {self.hparaproduct.n_bits}")
        logger.info(f"QUBO Energy Scale - Max Linear: {max(linear_biases) if linear_biases else 0}, Max Quadratic: {max(quadratic_biases) if quadratic_biases else 0}")
        res = self.minimizer.minimize_analog(qprod)
        logger.info(res)
        
        if isinstance(res, dict):
            # Get the key with the highest count (usually the first/only one)
            bitstring = max(res, key=res.get) 
        else:
            bitstring = res

        # Now bitstring is '10000000000000100010'
        k = self.map_landscape_ruggedness(bitstring, qubo_dict, 1)
        return None
        
    def _calculate_energy(self, bitstring: list[int], qubo_dict: dict) -> float:
        """Calculates the QUBO energy for a specific bitstring: x^T Q x"""
        energy = 0.0
        for (i, j), coeff in qubo_dict.items():
            if bitstring[i] == 1 and bitstring[j] == 1:
                energy += coeff
        return energy

    def map_landscape_ruggedness(self, optimal_bitstring, qubo_dict: dict, radius: int = 1):
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
    
    def run(self, mode: QMethod = QMethod.GATE_BASED) -> PortfolioResult:
        """
        Main entry point. 
        Supports 'analog' (Annealing) and 'gate-based' (QAOA/Warming).
        """
        # 1. Prepare Instance Data
        instance = self.symbolics.get_instance_data(
            proxy=self.context,
            k_target=self.hparaproduct.k_cardinality,
            lambda_risk=self.hparaproduct.lambda_risk,
            lambda_reward=self.hparaproduct.lambda_reward,
            lambda_cardinality=self.hparaproduct.lambda_cardinality,
            lambda_esg=self.hparaproduct.lambda_esg,
            n_bits=self.hparaproduct.n_bits,
            prev_weights=self.hparaproduct.prev_weights
        )
        qprod = self.instantiator.get_qubo_prod(self.problem, self.expected_variables, instance)
        config_add: Dict[str, Any] = {} # Initialize as empty dict
    
        if mode == QMethod.ANALOG_BASED:
            logger.info("Executing Analog Annealing Path...")
            result_dict = self.minimizer.minimize_analog(qprod)
            
            # 1. Robustly get the bitstring. 
            # Ensure it is a string of '0's and '1's
            raw_key = list(result_dict.keys())[0]
            if isinstance(raw_key, (list, tuple, np.ndarray)):
                bitstring = "".join(map(str, raw_key))
            else:
                bitstring = str(raw_key)
                
            logger.info(f"Bitstring identified: {bitstring}")

            # 2. Calculate selections with safety slicing
            decisions = []
            n_assets = len(self.context.tickers)
            n_bits = self.hparaproduct.n_bits
            
            for i in range(n_assets):
                start = i * n_bits
                end = start + n_bits
                chunk = bitstring[start : end]
                
                if not chunk:
                    logger.warning(f"Empty chunk for asset {i}. Check bitstring length!")
                    decisions.append(0)
                    continue
                
                # Use [::-1] if your QUBO builder puts the least significant bit first
                decisions.append(int(chunk, 2))
                
            config_add = self.map_landscape_ruggedness(bitstring, qprod.qubo_dict, 1)                
            config_add['prob_success'] = 1.0
        else:
            logger.info("Executing Gate-based/Warming Path...")
            min_result = self.minimizer.minimize_with_warming(qprod)
            decisions = self.minimizer.sample_best_configuration(qprod, min_result)
            bitstring = "".join(map(str, decisions)) 
            prob_success = 0.0
            if hasattr(min_result, 'eigenstate'):
                prob_success = min_result.eigenstate.get(bitstring, 0.0)
            elif hasattr(min_result, 'samples'):
                # For shot-based results, find the sample that matches our 'decisions'
                for sample in min_result.samples():
                    if "".join(map(str, sample.configuration)) == bitstring:
                        prob_success = sample.probability
                        break
            
            config_add = self.map_landscape_ruggedness(bitstring, qprod.qubo_dict, 1)
            config_add['prob_success'] = prob_success
            
        logger.info(f"Decision: {decisions}")
        selections = self.symbolics.decode_results(decisions)
        
        # 4. Finalize (Calculates Return, Vol, Sharpe)
        return self._finalize_portfolio(selections=selections, context=self.context, mode=mode, additional=config_add)

class QPortRiskRewardCardinality(QuantumPortfolioComposite):
    def __init__(self, 
        name: str, 
        proxy: FinancialContextCommand, 
        hparaproduct: QportHyperparameterProduct) -> None:
        symbolics = IntegerQProb(
            proxy.get_context().tickers,
            "QuantumPortfolio/Integer/RiskRewardCardinality",
            hparaproduct.n_bits
            ).add_strategy_list([
            RewardMuIntegerStrategy(),
            RiskCovarianceIntegerStrategy(),
            CardinalityBinaryStrategy()
        ])
        super().__init__(name, proxy, symbolics, hparaproduct)
        

class QPortRiskRewardCardinalityTurnover(QuantumPortfolioComposite):
    def __init__(self, 
        name: str, 
        proxy: FinancialContextCommand, 
        hparaproduct: QportHyperparameterProduct) -> None:
        symbolics = IntegerQProb(
            proxy.get_context().tickers,
            "QuantumPortfolio/Integer/RiskRewardCardinality",
            hparaproduct.n_bits
            ).add_strategy_list([
            RewardMuIntegerStrategy(),
            RiskCovarianceIntegerStrategy(),
            CardinalityBinaryStrategy()
        ])
        super().__init__(name, proxy, symbolics, hparaproduct)
        
class QPortRiskReward(QuantumPortfolioComposite):
    def __init__(self, 
        name: str, 
        proxy: FinancialContextCommand, 
        hparaproduct: QportHyperparameterProduct) -> None:
        symbolics = IntegerQProb(
            proxy.get_context().tickers,
            "QuantumPortfolio/Integer/RiskReward",
            hparaproduct.n_bits
            ).add_strategy_list([
            RewardMuIntegerStrategy(),
            RiskCovarianceIntegerStrategy(),
            CardinalityBinaryStrategy()
        ])
        super().__init__(name, proxy, symbolics, hparaproduct)
    
    
