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
    # --- TIER 1: The "Small" Core (N=4) ---
    # 1 Tech, 1 Finance, 1 Energy, 1 Safe Haven
    "AAPL", "JPM", "XOM", "GLD", 

    # --- TIER 2: The "Medium" Expansion (N=8) ---
    # Adds Cloud/Software, Investment Banking, Oil Major, and Treasuries
    "MSFT", "GS", "CVX", "TLT",

    # --- TIER 3: The "Large" Full Market (N=16) ---
    # Fills in the remaining high-volatility Tech and Semi-conductors
    "NVDA", "GOOGL", "AMZN", "META", "ADBE", "CRM", "ORCL", "INTC"
]

DEFAULT_PROJECT_CONFIG = ProjectConfig(tickers=DEFAULT_TICKERS)
