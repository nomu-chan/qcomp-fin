from dataclasses import dataclass
from typing import List
from pathlib import Path


DATAPATH = Path("./data/")
@dataclass(frozen=True)
class ProjectConfig:
    tickers: List[str]
    period: str = "5y"
    interval: str = "1d"
    risk_aversion: float = 2.0      # Risk aversion parameter for the QUBO model (higher means more risk-averse)
    target_cardinality: int = 5     # Target number of stocks out of default tickers 
    penalty_strength: float = 10.0   # Penalty strength for the cardinality constraint in the QUBO model


# 16 default tickers from the S&P 500, representing a mix of tech and other sectors.
DEFAULT_TICKERS = [
    "AAPL", # Apple Inc.
    "MSFT", # Microsoft
    "NVDA", # NVIDIA
    "GOOGL", # Google
    "AMZN", # Amazon
    "META", # Meta
    "ADBE", # Adobe
    "CRM", # Salesforce
    "ORCL", # Oracle
    "INTC", # Intel
    "JPM", 
    "GS",
    "XOM", 
    "CVX",
    "GLD", 
    "TLT"
]

DEFAULT_PROJECT_CONFIG = ProjectConfig(tickers=DEFAULT_TICKERS)
