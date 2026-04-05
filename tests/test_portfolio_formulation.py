import pytest
import numpy as np
from unittest.mock import MagicMock
from src.financial_context.context import FinancialContext
from src.symbolics.hamiltonians import HamiltonianStrategy, HamiltonianPlaceholders, RewardMuBinaryStrategy, RiskCovarianceBinaryStrategy, CardinalityBinaryStrategy, ESGBinaryStrategy
from src.symbolics.hamiltonian_modelling import QuantumProblemModelingBuilder, BinaryQProb, IntegerQProb
import jijmodeling as jm
import pandas as pd
from typing import Dict, Any, Union, OrderedDict

def create_mock_proxy():
    proxy = MagicMock(spec=FinancialContext)
    proxy.tickers = ["AAPL", "MSFT", "TSLA"]
    proxy.get_moments.return_value = (np.random.rand(3), np.random.rand(3, 3))
    proxy.esg_scores.return_value = pd.Series([80, 70, 90], index=proxy.tickers)
    return proxy

class TestPortfolioCompilation:
    # Use @staticmethod or remove 'self' if not needed for the fixture
    @pytest.fixture
    def mock_proxy(self):
        return create_mock_proxy()

    def test_binary_portfolio_compilation(self, mock_proxy):
        """Checks if the Binary Universe compiles with Risk and Cardinality."""
        portfolio = BinaryQProb(mock_proxy.tickers)
        portfolio.add_strategy(RewardMuBinaryStrategy())
        portfolio.add_strategy(CardinalityBinaryStrategy())
        
        problem = portfolio.build()
        assert isinstance(problem, jm.Problem)

        data = portfolio.get_instance_data(mock_proxy, k_target=2)
        assert data[HamiltonianPlaceholders.K_TARGET_ASSETS] == 2
        assert HamiltonianPlaceholders.MU in data

    def test_integer_portfolio_compilation(self, mock_proxy):
        """Checks if the Integer Universe compiles with Covariance and ESG."""
        portfolio = IntegerQProb(mock_proxy.tickers)
        portfolio.add_strategy(RiskCovarianceBinaryStrategy())
        portfolio.add_strategy(ESGBinaryStrategy())
        
        problem = portfolio.build()
        data = portfolio.get_instance_data(mock_proxy, k_target=1, max_units_per_asset=15)
        assert data[HamiltonianPlaceholders.MAX_UNITS] == 15


def run_sense_test():
    tickers = ["BTC", "ETH", "SOL"]
    
    print("\n--- RUNNING BINARY COMPILATION TEST ---")
    bp = BinaryQProb(tickers, name="EqualWeight_Test")
    bp.add_strategy(RewardMuBinaryStrategy())
    bp.add_strategy(CardinalityBinaryStrategy())
    
    model = bp.build()
    print(f"Model '{model.name}' built successfully.")
    
    # QAOA/Simulator style output: A flat list of bits
    mock_bits = [1, 1, 0] 
    
    # This now uses 'self.decision_strategy' internally
    result = bp.decode_results(mock_bits)
    
    print(f"Input Bits:  {mock_bits}")
    print(f"Decoded Res: {result}") 
    # Expected: [1, 1, 0]