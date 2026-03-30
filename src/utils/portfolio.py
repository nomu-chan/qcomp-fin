from abc import ABC, abstractmethod
from src.utils.logging_mod import get_logging
from dataclasses import dataclass
from typing import Dict, List, OrderedDict, Any
import pandas as pd
import numpy as np

@dataclass
class PortfolioResult:
    method_name: str        
    weights: OrderedDict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    esg_score: float = 0.0  
    energy: float | None = None
    success_prob: float | None = None
    config: OrderedDict[str, Any] | None = None
    def __str__(self) -> str:
        # Format weights: "TICKER: 10.00%"
        weight_str = "\n".join([f"  {k}: {v:.2%}" for k, v in self.weights.items() if v > 0])
        
        # Format config: "key: value"
        config_str = ""
        if self.config:
            config_str = "\n".join([f"  {k}: {v}" for k, v in self.config.items()])

        return "\n".join([
            f"--- Portfolio Result: {self.method_name} ---",
            f"Weights:",
            weight_str if weight_str else "  (No assets selected)",
            f"Expected Return:  {self.expected_return:.2%}",
            f"Annual Volatility: {self.volatility:.2%}",
            f"Sharpe Ratio:      {self.sharpe_ratio:.4f}",
            f"ESG Score:         {self.esg_score:.2f}",
            f"Energy:            {self.energy if self.energy is not None else 'N/A'}",
            f"Success Prob:      {f'{self.success_prob:.2%}' if self.success_prob else 'N/A'}",
            f"Config details:",
            config_str if config_str else "  (None)",
            "------------------------------------------"
        ])
# Defunct Code
# for ticker, weight in classical.weights.items():
#         if weight > 0:
#             print(f"  {ticker}: {weight:.3f}")
logger = get_logging(__name__)

class Portfolio(ABC):
    def __init__(self, name: str, seed: int = 67):
        self.name = name
        self.seed = seed
        pass

    @abstractmethod
    def run(cls, prices: pd.DataFrame) -> PortfolioResult:
        logger.info(f"Running {cls.name}...")
        pass
    
    def get_qubo_matrix_as_df(self, qubo_dict: dict, n_qubits: int) -> pd.DataFrame:
        """
        Converts the QUBO dictionary to a dense DataFrame for reporting.
        """
        Q = np.zeros((n_qubits, n_qubits))
        for (i, j), coeff in qubo_dict.items():
            # QUBOs are symmetric; we fill both sides for a clear heatmap-style view
            Q[i, j] = coeff
            if i != j:
                Q[j, i] = coeff
                
        # Labeling qubits: 0-19 are Assets (2 per asset), 20-29 are Slacks
        columns = [f"q{i}" for i in range(n_qubits)]
        return pd.DataFrame(Q, index=columns, columns=columns)