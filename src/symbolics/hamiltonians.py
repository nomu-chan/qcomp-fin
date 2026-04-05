from abc import ABC, abstractmethod
import jijmodeling as jm
from src.financial_context.context import FinancialContext
import pandas as pd

from dataclasses import dataclass
from abc import ABC, abstractmethod
import jijmodeling as jm


class HamiltonianPlaceholders:
    # Variables
    X_BINARY = "x_selection"
    W_INTEGER = "w_weights"
    # Finance
    MU = "mu"       # Asset Returns (Linear)
    SIGMA = "sigma" # Risk / Covariance (Quadratic)
    
    # Sustainability
    ESG = "esg"
    # Cardinality (Number of Assets)
    K_TARGET_ASSETS = "K_cardinality"              
    # Constraint IDs
    CARD_LIMIT = "cardinality_limit"
    MAX_UNITS = "max_units"
    
    # Force Strengths
    LAMBDA_RISK = "lambda_risk"
    LAMBDA_REWARD = "lambda_reward"
    LAMBDA_CARDINALITY = "lambda_cardinality"
    LAMBDA_ESG = "lambda_esg"
    
    # constants
    N_ASSETS = "n_assets"
    N_INTEGER_BITS = "n_bits"

    
@dataclass
class StrategyOutput:
    expression: jm.Expression
    tags: list[str]
    asset_expressions: list[jm.Expression] | None = None # List of weighted asset sums

class HamiltonianStrategy(ABC):
    @abstractmethod
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Added to base contract
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        raise NotImplementedError("Please implement the Hamiltonian.")

class RewardMuBinaryStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        mu = problem.Placeholder(HamiltonianPlaceholders.MU, shape=(n_assets,), dtype=float)
        lambda_reward = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_REWARD, dtype=float)
        sum_expr = -jm.sum([mu[i] * x[i] for i in range(n_assets)]) * lambda_reward
        placeholders = [HamiltonianPlaceholders.MU, HamiltonianPlaceholders.LAMBDA_REWARD]
        return StrategyOutput(sum_expr, placeholders), problem

class RiskCovarianceBinaryStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        sigma = problem.Placeholder(HamiltonianPlaceholders.SIGMA, shape=(n_assets, n_assets), dtype=float)
        lambda_risk = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_RISK, dtype=float)

        risk_expr = jm.sum([
            x[i] * sigma[i, j] * x[j] 
            for i in range(n_assets) for j in range(n_assets)
        ]) * lambda_risk
        placeholders = [HamiltonianPlaceholders.SIGMA, HamiltonianPlaceholders.LAMBDA_RISK]
        return StrategyOutput(risk_expr, placeholders), problem
    
class CardinalityBinaryStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        # 1. Define Placeholders (Data Inputs)
        k = problem.Placeholder(HamiltonianPlaceholders.K_TARGET_ASSETS, dtype=int)
        lambda_cardinality = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_CARDINALITY, dtype=float)
        
        # 2. Define the Soft Penalty (Added to Objective Function)
        asset_sum = jm.sum([x[i] for i in range(n_assets)])
        penalty_expr = lambda_cardinality * (asset_sum - k)**2
        
        # problem += problem.Constraint(HamiltonianPlaceholders.CARD_LIMIT, asset_sum == k)
        
        placeholders = [
            HamiltonianPlaceholders.LAMBDA_CARDINALITY, 
            HamiltonianPlaceholders.K_TARGET_ASSETS
        ]
        
        return StrategyOutput(penalty_expr, placeholders), problem
    
class ESGBinaryStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        esg = problem.Placeholder(HamiltonianPlaceholders.ESG, shape=(n_assets,), dtype=float)
        lambda_esg = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_ESG, dtype=float)
        sum_expr = - jm.sum([esg[i] * x[i] for i in range(n_assets)]) * lambda_esg
        placeholders = [HamiltonianPlaceholders.ESG, HamiltonianPlaceholders.LAMBDA_ESG]
        return StrategyOutput(sum_expr, placeholders), problem
    
class RewardMuIntegerStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        if not strategy_output.asset_expressions:
             raise ValueError("Integer Reward Strategy requires pre-defined asset weight expressions!")
        
        mu = problem.Placeholder(HamiltonianPlaceholders.MU, shape=(n_assets,), dtype=float)
        lambda_reward = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_REWARD, dtype=float)
        
        # Guard against None and use weighted asset expressions
        assets = strategy_output.asset_expressions if strategy_output.asset_expressions else []
        
        sum_expr = -jm.sum([
            mu[i] * assets[i] for i in range(n_assets)
        ]) * lambda_reward
        
        return StrategyOutput(sum_expr, [HamiltonianPlaceholders.MU, HamiltonianPlaceholders.LAMBDA_REWARD]), problem
    
class RiskCovarianceIntegerStrategy(HamiltonianStrategy):
    def build_expression(self,
        x: jm.DecisionVar,
        problem: jm.DecoratedProblem,
        n_assets: int,
        strategy_output: StrategyOutput, # Fixed position
        input: int | float | None = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem]:
        sigma = problem.Placeholder(HamiltonianPlaceholders.SIGMA, shape=(n_assets, n_assets), dtype=float)
        lambda_risk = problem.Placeholder(HamiltonianPlaceholders.LAMBDA_RISK, dtype=float)
        
        assets = strategy_output.asset_expressions if strategy_output.asset_expressions else []

        risk_expr = jm.sum([
            assets[i] * sigma[i, j] * assets[j] 
            for i in range(n_assets) for j in range(n_assets)
        ]) * lambda_risk
        
        return StrategyOutput(risk_expr, [HamiltonianPlaceholders.SIGMA, HamiltonianPlaceholders.LAMBDA_RISK]), problem