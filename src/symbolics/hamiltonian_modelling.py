import jijmodeling as jm
from src.symbolics.hamiltonians import HamiltonianStrategy, HamiltonianPlaceholders
from src.symbolics.decisions import BinarySelectionStrategy, IntegerWeightingStrategy, ChoiceStrategy
from src.financial_context.context import FinancialContext
from typing import Dict, Any, Union, OrderedDict
from abc import ABC, abstractmethod
from utils.logging_mod import get_logging
import numpy as np
import pandas as pd
logger = get_logging(__name__)

class QuantumProblemModelingBuilder(ABC):
    def __init__(self, 
        tickers: list[str], 
        decision_strategy: ChoiceStrategy, 
        name: str = "BasePortfolio"
    ):
        self.tickers = tickers
        self.name = name
        self._problem = jm.Problem(name)
        self.strategies: list[HamiltonianStrategy] = []
        self.decision_strategy: ChoiceStrategy = decision_strategy
        self.n_assets = len(tickers)
    def add_strategy(self, strategy: HamiltonianStrategy):
        self.strategies.append(strategy)
        return self
    
    def add_strategy_list(self, strategy_list: list[HamiltonianStrategy]):
        self.strategies.extend(strategy_list)
        return self
    
    @abstractmethod
    def decode_results(self, decisions):
        return self.decision_strategy.decode_expression(decisions)
    
    def get_instance_data(self, 
        proxy: FinancialContext, 
        k_target: int, 
        max_units_per_asset: int = 10,        
        lambda_risk: float = 0.5,
        lambda_reward: float = 0.5, 
        lambda_cardinality: float = 0.1,
        lambda_esg: float = 1.0,
        n_bits: int = 1,
        lambda_turnover=0.1, 
        prev_weights=None
    ) -> Dict[str, Any]:
        """
        Maps Proxy data to the Symbolic Placeholders.
        """
        # Fetch the pre-scaled moments from your 2026 FinancialContext
        mu_scaled, sigma_scaled = proxy.get_moments(True, True, True)
        esg_scaled = proxy.esg_scores().values
        # Build the instance dictionary using your Constants
        if prev_weights is None:
            prev_weights = np.zeros(self.n_assets)
        instance_data = {
            # Financial Data
            HamiltonianPlaceholders.MU: mu_scaled,
            HamiltonianPlaceholders.SIGMA: sigma_scaled,
            # Governance
            HamiltonianPlaceholders.ESG: esg_scaled,
            # Constraints & Hyperparameters
            HamiltonianPlaceholders.K_TARGET_ASSETS: k_target,
            HamiltonianPlaceholders.MAX_UNITS: max_units_per_asset,
            # Global Orchestrator Weight (Risk/Return balance)
            HamiltonianPlaceholders.LAMBDA_RISK: lambda_risk,
            HamiltonianPlaceholders.LAMBDA_REWARD: lambda_reward,
            HamiltonianPlaceholders.LAMBDA_ESG: lambda_esg,
            HamiltonianPlaceholders.LAMBDA_CARDINALITY: lambda_cardinality,
            HamiltonianPlaceholders.N_ASSETS: self.n_assets,
            HamiltonianPlaceholders.N_INTEGER_BITS: n_bits,
            HamiltonianPlaceholders.PREV_X: prev_weights,
            HamiltonianPlaceholders.LAMBDA_TURNOVER: lambda_turnover,
        }
        logger.info("Instance Data:")
        logger.info(instance_data)
        return instance_data
    
    def build(self) -> tuple[jm.Problem, list[str]]:
        strategies: list[str] = []
        
        @self._problem.update
        def _(problem: jm.DecoratedProblem):
            # 1. Capture the strategy_output (contains asset_expressions)
            strategy_out_base, problem, x = self.decision_strategy.build_expression(problem)
            
            for strategy in self.strategies:
                # 2. Pass strategy_out_base as the 4th argument (matching our new ABC)
                strategy_out, problem = strategy.build_expression(
                    x, 
                    problem, 
                    self.n_assets, 
                    strategy_out_base  # <--- This fixes the Pylance error
                )
                

                problem += strategy_out.expression
                strategies.extend(strategy_out.tags)
                logger.debug(f"Strategy: {strategy_out.tags} Expression Added")
        self.variables = strategies
        logger.debug("Symbolic Problem Composed.")
        return self._problem, strategies
    
class BinaryQProb(QuantumProblemModelingBuilder):
    def __init__(self, tickers: list[str], name: str = "BinaryPortfolio"):
        super().__init__(tickers, BinarySelectionStrategy(len(tickers)), name)

    def decode_results(self, decisions: list[int]):
        return self.decision_strategy.decode_expression(decisions)
    
class IntegerQProb(QuantumProblemModelingBuilder):
    def __init__(self, tickers: list[str], name: str = "IntegerPortfolio", n_bits=8):
        super().__init__(tickers, IntegerWeightingStrategy(len(tickers), n_bits), name)
    
    def decode_results(self, decisions: Dict[tuple[Any, Any], int]):
        return self.decision_strategy.decode_expression(decisions)
        
    
    