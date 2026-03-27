from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    tickers: List[str]
    period: str = "5y"
    interval: str = "1d"
    # Risk aversion parameter for the QUBO model (higher means more risk-averse)
    risk_aversion: float = 2.0
    # Target number of stocks out of default tickers 
    target_cardinality: int = 5
    # Penalty strength for the cardinality constraint in the QUBO model
    penalty_strength: float = 10.0


# 10 default tickers from the S&P 500, representing a mix of tech and other sectors.
def default_tickers() -> List[str]:
    return [
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
    ]


def default_config() -> ProjectConfig:
    return ProjectConfig(tickers=default_tickers())
