from abc import ABC, abstractmethod
import jijmodeling as jm
from src.financial_context.context import FinancialContext
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass
import jijmodeling as jm
from src.symbolics.hamiltonians import StrategyOutput, HamiltonianPlaceholders
from src.utils.logging_mod import get_logging
logger = get_logging(__name__)
class ChoiceStrategy(ABC):
    def __init__(self, n_assets: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        
    def set_n_assets(self, n_assets: int):
        self.n_assets = n_assets
        
    @abstractmethod
    def build_expression(self,
        problem: jm.DecoratedProblem,
        input = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem, jm.DecisionVar]:
        raise NotImplementedError("Please implement the decision variable.")
    
    @abstractmethod
    def decode_expression(self, decisions: list[int] | Dict[tuple[Any, Any], int]) -> list[int]:
        raise NotImplementedError("Please implement way to decode decision variable")
    
class BinarySelectionStrategy(ChoiceStrategy):
    def __init__(self, n_assets: int,) -> None:
        super().__init__(n_assets)
    
    def build_expression(self,
        problem: jm.DecoratedProblem,
        input = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem, jm.DecisionVar]:
        # Only Binary. No Integers allowed.
        x = problem.BinaryVar(HamiltonianPlaceholders.X_BINARY, shape=(self.n_assets,))
        
        # We return the selection sum for Cardinality checks
        selection_expr = jm.sum([x[i] for i in range(self.n_assets)])
        return StrategyOutput(selection_expr, [HamiltonianPlaceholders.X_BINARY]), problem, x
    
    def decode_expression(self, decisions: list[int] | Dict[tuple[Any, Any], int]) -> list[int]:
        # Pylance is happy because we accept the full Union
        if isinstance(decisions, dict):
            raise ValueError("Not a binary list.")
            
        return [1 if b == 1 else 0 for b in decisions]
    
class IntegerWeightingStrategy(ChoiceStrategy):
    def __init__(self, n_assets: int, n_bits: int = 2) -> None:
        super().__init__(n_assets)
        self.n_bits = n_bits # e.g., 2 bits = 0-3 units, 3 bits = 0-7 units

    def build_expression(self,
        problem: jm.DecoratedProblem,
        input = None
    ) -> tuple[StrategyOutput, jm.DecoratedProblem, jm.DecisionVar]:
        x = problem.BinaryVar(HamiltonianPlaceholders.W_INTEGER, shape=(self.n_assets, self.n_bits))
        
        asset_weights = [
            jm.sum([(2**b) * x[i_idx, b] for b in range(self.n_bits)])
            for i_idx in range(self.n_assets)
        ] # List form
        
        # 2. Total units (for cardinality)
        total_units_expr = jm.sum([
            jm.sum([(2**b) * x[i_idx, b] for b in range(self.n_bits)])
            for i_idx in range(self.n_assets) # Inside here, its treated as generator DO NOT TOUCH
        ])
        
        # 3. Return the StrategyOutput
        output = StrategyOutput(
            expression=total_units_expr,
            tags=[HamiltonianPlaceholders.W_INTEGER],
            asset_expressions=asset_weights
        )
        
        return output, problem, x
        
    # Match the base class signature exactly
    # def decode_expression(self, decisions: list[int] | Dict[tuple[Any, Any], int]) -> list[int]:
    #     weights = [0] * self.n_assets

    #     # Add a type guard to handle the Union
    #     if not isinstance(decisions, dict):
    #         raise ValueError("IntegerWeightingStrategy expects a result dictionary from the solver.")

    #     for key, bit_val in decisions.items():
    #         if bit_val == 1:
    #             # We cast to int or handle the tuple properly
    #             flat_idx = int(key[0]) 
                
    #             asset_idx = flat_idx // self.n_bits
    #             bit_idx = flat_idx % self.n_bits
                
    #             weights[asset_idx] += (2 ** bit_idx)
                
    #     return weights
    
    def decode_expression(self, decisions: list[int] | dict[Any, int]) -> list[int]:
        weights = [0] * self.n_assets
            # If it's a list of ints where values are > 1, it's already decoded!
        if isinstance(decisions, list) and any(v > 1 for v in decisions):
            return decisions

        # CASE 1: Dictionary Input (Can be JijModeling tuples OR a raw Analog bitstring)
        if isinstance(decisions, dict):
            # Check if the dictionary contains a long bitstring (Analog Path)
            # e.g., {'00001000...': 1}
            first_key = list(decisions.keys())[0]
            logger.info(f"first_key: {first_key}")

            if isinstance(first_key, str) and len(first_key) > self.n_bits:
                # Handle the Analog Bitstring we saw in the logs
                for asset_idx in range(self.n_assets):
                    chunk = first_key[asset_idx * self.n_bits : (asset_idx + 1) * self.n_bits]
                    # Note: Binary strings are usually read right-to-left for power of 2, 
                    # but int(chunk, 2) works if '1000' is meant to be 8.
                    weights[asset_idx] = int(chunk, 2)
            
            else:
                # Handle Digital/Gate-based (Dictionary with tuples (idx,))
                for key, bit_val in decisions.items():
                    if bit_val == 1:
                        flat_idx = int(key[0]) 
                        asset_idx = flat_idx // self.n_bits
                        bit_idx = flat_idx % self.n_bits
                        if asset_idx < self.n_assets:
                            weights[asset_idx] += (2 ** bit_idx)

        # CASE 2: Flat list (Fallback)
        elif isinstance(decisions, list):
            for i, bit_val in enumerate(decisions):
                if bit_val == 1:
                    asset_idx = i // self.n_bits
                    bit_idx = i % self.n_bits
                    if asset_idx < self.n_assets:
                        weights[asset_idx] += (2 ** bit_idx)

        return weights